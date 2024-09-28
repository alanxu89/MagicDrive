# MagicDrive multi-view video generation with first-person view depth and semantic_map inputs

This repo is based on [MagicDrive3D](https://github.com/flymin/MagicDrive3D) on **3D scene generation**, but has been extended to video generation and with first-person view depth map and semantic map as control inputs. 

## changes
We have made a lot of changes to the original architecture in order to generate consistent multi-view videos given the new input conditions, including:
- A new nuScenes data processing pipeline to generate [depth map](tools/fpv_nuscenes/process_depth.py) using Depth-Anything-V2 and [fpv semantic map](tools/fpv_nuscenes/process_semantic_map.py) with the nuScenes map api for each camera and each frame of a scene. We then iterate over the available scenes and collect relevant information to a [csv file](tools/fpv_nuscenes/process_frames.py) for further data processing. 
- We create a new [DatasetFromCSV](magicdrive/dataset/fpv_nuscenes_dataset.py) class to load scene data into pytorch tensors, including rgb, depth, semantic, text description, etc. 
- A newly designed [fpv_runner](magicdrive/runner/fpv_runner.py) where we wrap a MultiControlNet and stable diffusion Unet into a model, and load the 6D data (b, c, f, n, h, w) into the the model for diffusion training, with f as the data frames and n as the camera views.
- Model architucture changes: The controlnet in the original MagicDrive3D is being replaced by a MultiControlNet defined in the diffusers library, where it accepts depth and semantic_map as conditional inputs. The overall structure of the [Unet](magicdrive/networks/unet_2d_condition_fpv.py) remains unchanged, but all the BasicTransformerBlock are being replaced by the [BasicMultiviewVideoTransformerBlock](magicdrive/networks/blocks_video.py#L36). 
- The BasicMultiviewVideoTransformerBlock adds three types of attentions: (1) [SparseCausalAttention](magicdrive/networks/blocks_video.py#L378) between frame i and frame 0 and frame i-1; (2) cross-view attention between neighboring camera views; (3) temporal attention along the frame axis of video data. 
- A new pipeline for video generation with depth and map inputs, as detailed in [pipeline_fpv_controlnet](magicdrive/pipeline/pipeline_fpv_controlnet.py).

The default branch is now the alanxu/fpv branch where all the changes are made, while the main branch is just a direct copy of the [MagicDrive3D](https://github.com/flymin/MagicDrive3D) repo without any modifications. 

## results
Here we show some of the preliminary results after training on 142 scenes (6 key frames for each scene) with 2 A800-80G GPUs for about 8 hours. The scene description is from nuScenes. Each image below is divided into four quarters, with:
- upper-left for original frames
- lower-left for generated frames
- upper-right for depth frames
- lower-right for semantic_map frames. 

We can clearly see that consistent multi-view video frames are generated with our model. 

A driving scene image at boston-seaport. Wait at intersection, truck, peds, turn right, parking lot.
![scene1](assets/img_gens/img0.png)

A driving scene image at singapore-onenorth. Intersection, peds, waiting vehicle, parked motorcycle at parking lot.
![scene2](assets/img_gens/img1.png)

A driving scene image at singapore-onenorth. Many peds, parked buses, parked cars.
![scene3](assets/img_gens/img2.png)

A driving scene image at singapore-onenorth. Turn left, motorcycle driving, ped sitting.
![scene4](assets/img_gens/img3.png)

