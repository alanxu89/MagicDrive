from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

from diffusers.models.attention_processor import Attention
from diffusers.models.attention import BasicTransformerBlock, AdaLayerNorm
from diffusers.models.controlnet import zero_module


def _ensure_kv_is_int(view_pair: dict):
    """yaml key can be int, while json cannot. We convert here.
    """
    new_dict = {}
    for k, v in view_pair.items():
        new_value = [int(vi) for vi in v]
        new_dict[int(k)] = new_value
    return new_dict


class GatedConnector(nn.Module):

    def __init__(self, dim) -> None:
        super().__init__()
        data = torch.zeros(dim)
        self.alpha = nn.parameter.Parameter(data)

    def forward(self, inx):
        # as long as last dim of input == dim, pytorch can auto-broad
        return F.tanh(self.alpha) * inx


class BasicMultiviewVideoTransformerBlock(BasicTransformerBlock):

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
        # multi_view
        neighboring_view_pair: Optional[Dict[int, List[int]]] = None,
        neighboring_attn_type: Optional[str] = "add",
        zero_module_type="zero_linear",
        # temporal
        frames=16,
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, dropout,
                         cross_attention_dim, activation_fn,
                         num_embeds_ada_norm, attention_bias,
                         only_cross_attention, double_self_attention,
                         upcast_attention, norm_elementwise_affine, norm_type,
                         final_dropout)

        self.neighboring_view_pair = _ensure_kv_is_int(neighboring_view_pair)
        self.neighboring_attn_type = neighboring_attn_type
        self.n_frame = frames

        # multiview attention
        self.norm4 = (AdaLayerNorm(dim, num_embeds_ada_norm)
                      if self.use_ada_layer_norm else nn.LayerNorm(
                          dim, elementwise_affine=norm_elementwise_affine))
        self.attn4 = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        if zero_module_type == "zero_linear":
            # NOTE: zero_module cannot apply to successive layers.
            self.connector = zero_module(nn.Linear(dim, dim))
        elif zero_module_type == "gated":
            self.connector = GatedConnector(dim)
        elif zero_module_type == "none":
            # TODO: if this block is in controlnet, we may not need zero here.
            self.connector = lambda x: x
        else:
            raise TypeError(f"Unknown zero module type: {zero_module_type}")

        # ST-Attn from Tune-A-Video
        self.norm5 = (AdaLayerNorm(dim, num_embeds_ada_norm)
                      if self.use_ada_layer_norm else nn.LayerNorm(
                          dim, elementwise_affine=norm_elementwise_affine))
        self.attn5 = SparseCausalAttention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=None,
            upcast_attention=upcast_attention,
        )

        # temporal attention
        self.norm6 = (AdaLayerNorm(dim, num_embeds_ada_norm)
                      if self.use_ada_layer_norm else nn.LayerNorm(
                          dim, elementwise_affine=norm_elementwise_affine))
        self.attn6 = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )

    @property
    def new_module(self):
        ret = {
            "norm4": self.norm4,
            "attn4": self.attn4,
        }
        if isinstance(self.connector, nn.Module):
            ret["connector"] = self.connector
        return ret

    @property
    def n_cam(self):
        return len(self.neighboring_view_pair)

    def _construct_multiview_attn_input(self, norm_hidden_states):
        B = len(norm_hidden_states)
        # reshape, key for origin view, value for ref view
        hidden_states_in1 = []
        hidden_states_in2 = []
        cam_order = []
        if self.neighboring_attn_type == "add":
            for key, values in self.neighboring_view_pair.items():
                for value in values:
                    hidden_states_in1.append(norm_hidden_states[:, key])
                    hidden_states_in2.append(norm_hidden_states[:, value])
                    cam_order += [key] * B
            # N*2*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "concat":
            for key, values in self.neighboring_view_pair.items():
                hidden_states_in1.append(norm_hidden_states[:, key])
                hidden_states_in2.append(
                    torch.cat(
                        [norm_hidden_states[:, value] for value in values],
                        dim=1))
                cam_order += [key] * B
            # N*B, H*W, head*dim
            hidden_states_in1 = torch.cat(hidden_states_in1, dim=0)
            # N*B, 2*H*W, head*dim
            hidden_states_in2 = torch.cat(hidden_states_in2, dim=0)
            cam_order = torch.LongTensor(cam_order)
        elif self.neighboring_attn_type == "self":
            hidden_states_in1 = rearrange(norm_hidden_states,
                                          "b n l ... -> b (n l) ...")
            hidden_states_in2 = None
            cam_order = None
        else:
            raise NotImplementedError(
                f"Unknown type: {self.neighboring_attn_type}")
        return hidden_states_in1, hidden_states_in2, cam_order

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ):
        """
        hidden_states: [b*f*n, channel, height, width], where f=frames, n=num_cams
        """
        # Notice that normalization is always applied before the real computation in the following blocks.
        # *********  1. Self-Attention  *********
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states,
                timestep,
                class_labels,
                hidden_dtype=hidden_states.dtype)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states
            if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # *********  2. Cross-Attention  *********
        if self.attn2 is not None:
            norm_hidden_states = (self.norm2(hidden_states, timestep)
                                  if self.use_ada_layer_norm else
                                  self.norm2(hidden_states))
            # TODO (Birch-San): Here we should prepare the encoder_attention mask correctly
            # prepare attention mask here

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states
        # print(f"hidden_states after Cross Attention {hidden_states.shape}")

        # *********  S-T Attention  *********
        norm_hidden_states = self.norm5(hidden_states)
        attn_output = self.attn5(norm_hidden_states,
                                 encoder_hidden_states=encoder_hidden_states
                                 if self.only_cross_attention else None,
                                 video_length=self.n_frame,
                                 num_cams=self.n_cam)
        hidden_states = attn_output + hidden_states
        # print(f"hidden_states after S-T Attention {hidden_states.shape}")

        # *********  multi-view cross attention  *********
        norm_hidden_states = (self.norm4(hidden_states, timestep)
                              if self.use_ada_layer_norm else
                              self.norm4(hidden_states))
        # batch dim first, cam dim second
        norm_hidden_states = rearrange(norm_hidden_states,
                                       '(b f n) ... -> (b f) n ...',
                                       f=self.n_frame,
                                       n=self.n_cam)
        B = len(norm_hidden_states)
        # key is query in attention; value is key-value in attention
        hidden_states_in1, hidden_states_in2, cam_order = self._construct_multiview_attn_input(
            norm_hidden_states, )
        # print(hidden_states_in1.shape, hidden_states_in1.shape)
        # attention
        attn_raw_output = self.attn4(
            hidden_states_in1,
            encoder_hidden_states=hidden_states_in2,
            **cross_attention_kwargs,
        )
        # final output
        if self.neighboring_attn_type == "self":
            attn_output = rearrange(attn_raw_output,
                                    'b (n l) ... -> b n l ...',
                                    n=self.n_cam)
        else:
            attn_output = torch.zeros_like(norm_hidden_states)
            for cam_i in range(self.n_cam):
                attn_out_mv = rearrange(attn_raw_output[cam_order == cam_i],
                                        '(n b) ... -> b n ...',
                                        b=B)
                attn_output[:, cam_i] = torch.sum(attn_out_mv, dim=1)
        attn_output = rearrange(attn_output, 'b n ... -> (b n) ...')
        # apply zero init connector (one layer)
        attn_output = self.connector(attn_output)
        # short-cut
        hidden_states = attn_output + hidden_states

        # *********  temporal attention  *********
        norm_hidden_states = (self.norm6(hidden_states, timestep)
                              if self.use_ada_layer_norm else
                              self.norm6(hidden_states))
        seq_len = norm_hidden_states.shape[1]
        norm_hidden_states = rearrange(norm_hidden_states,
                                       "(b f n) s c -> (b n s) f c",
                                       f=self.n_frame,
                                       n=self.n_cam)
        # print(f"norm_hidden_states: {norm_hidden_states.shape}")
        attn_output = self.attn6(norm_hidden_states)
        attn_output = rearrange(attn_output,
                                '(b n s) f c -> (b f n) s c',
                                s=seq_len,
                                f=self.n_frame,
                                n=self.n_cam)
        hidden_states = attn_output + hidden_states

        # *********  3. Feed-forward  *********
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (
                1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class SparseCausalAttention(Attention):

    def forward(self,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                video_length=None,
                num_cams=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(
                1, 2)).transpose(1, 2)

        # print(f"hidden_states: {hidden_states.shape}")
        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.head_to_batch_dim(query)
        # print(f"query: {query.shape}")

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # print(f"encoder_hidden_states: {encoder_hidden_states.shape}")

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0

        key = rearrange(key,
                        "(b f n) d c -> (b n) f d c",
                        f=video_length,
                        n=num_cams)
        key = torch.cat(
            [key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
        key = rearrange(key, "(b n) f d c -> (b f n) d c", n=num_cams)

        value = rearrange(value,
                          "(b f n) d c -> (b n) f d c",
                          f=video_length,
                          n=num_cams)
        value = torch.cat(
            [value[:, [0] * video_length], value[:, former_frame_index]],
            dim=2)
        value = rearrange(value, "(b n) f d c -> (b f n) d c", n=num_cams)

        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)
        # print(f"key/value: {key.shape, value.shape}")

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length),
                                       value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads,
                                                                  dim=0)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False)
        # print(f"out hidden_states: {hidden_states.shape}")

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.out_dim)
        hidden_states = hidden_states.to(query.dtype)
        # print(f"out hidden_states: {hidden_states.shape}")

        # attention, what we cannot get enough of
        # if self._use_memory_efficient_attention_xformers:
        #     hidden_states = self._memory_efficient_attention_xformers(
        #         query, key, value, attention_mask)
        #     # Some versions of xformers return output in fp32, cast it back to the dtype of the input
        #     hidden_states = hidden_states.to(query.dtype)
        # else:
        #     if self._slice_size is None or query.shape[
        #             0] // self._slice_size == 1:
        #         hidden_states = self._attention(query, key, value,
        #                                         attention_mask)
        #     else:
        #         hidden_states = self._sliced_attention(query, key, value,
        #                                                sequence_length, dim,
        #                                                attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
