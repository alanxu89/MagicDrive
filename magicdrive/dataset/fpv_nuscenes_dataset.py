import os
import time

import pandas as pd
import numpy as np
import torch
import torchvision


def read_image_files_into_array(file_list, data_root, img_size):
    rgb_list = []
    for fpath in file_list:
        image_array = torchvision.io.read_image(os.path.join(
            data_root, fpath))  # [C, H, W]
        resize = torchvision.transforms.Resize(img_size)
        image_array = resize(image_array)
        # for depth image if only 1 channel, we repeat to get 3
        if image_array.shape[0] == 1:
            image_array = image_array.repeat(3, 1, 1)
        rgb_list.append(image_array)
    rgb = torch.stack(rgb_list)  # [B, C, H, W]
    return rgb


class DatasetFromCSV(torch.utils.data.Dataset):

    def __init__(self,
                 csv_path,
                 num_frames=16,
                 frame_interval=1,
                 transform=None,
                 img_size=None,
                 root=None):
        self.df = pd.read_csv(os.path.join(root, csv_path))
        (self.scene_col, self.timestamp_col, self.camera_col, self.mode_col,
         self.filepath_col, self.location_col,
         self.description_col) = self.df.columns

        self.scene_names = self.df[self.scene_col].unique()
        self.n_cams = len(list(self.df[self.camera_col].unique()))
        self.n_modes = len(list(self.df[self.mode_col].unique()))

        self.transform = transform
        self.img_size = img_size
        self.root = root

        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.num_real_frames = 1 + (num_frames - 1) * frame_interval

    def getitem(self, index):
        t0 = time.time()
        # get a scene
        # and sort by timestamp
        # and then by camera ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        # and then by modes ['depth', 'map', 'rgb']
        scene_name = self.scene_names[index]
        scene_df = self.df[self.df[self.scene_col] == scene_name].sort_values(
            by=[self.timestamp_col, self.camera_col, self.mode_col])
        scene_description = scene_df[self.description_col].iloc[0]

        all_files = scene_df[self.filepath_col]
        all_data = read_image_files_into_array(all_files, self.root,
                                               self.img_size)  # [B, C, H, W]
        if self.transform is not None:
            all_data = self.transform(all_data)

        C, H, W = all_data.shape[-3:]
        all_data = torch.reshape(
            all_data,
            (-1, self.n_cams, self.n_modes, C, H, W))  # [T, N, M, C, H, W]

        # Sampling video frames
        # total_frames = len(all_data)
        # start_frame_ind = random.randint(0, total_frames - self.num_real_frames)
        start_frame_ind = 0
        end_frame_ind = start_frame_ind + self.num_real_frames
        frame_indice = np.arange(start_frame_ind,
                                 end_frame_ind,
                                 step=self.frame_interval,
                                 dtype=int)
        all_data = all_data[frame_indice]

        all_data = all_data.permute(3, 0, 1, 2, 4, 5)  # [C, T, N, M, H, W]

        return {
            "rgb": all_data[:, :, :, 2],
            "semantic_map": all_data[:, :, :, 1],
            "depth": all_data[:, :, :, 0],
            "text": scene_description,
            "video_id": scene_name,
        }

    def __getitem__(self, index):
        for _ in range(5):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.scene_names)


class PreprocessedDatasetFromCSV(torch.utils.data.Dataset):

    def __init__(self,
                 csv_path,
                 num_frames=None,
                 root=None,
                 preprocessed_dir=None):

        self.csv_path = csv_path
        if (not os.path.exists(csv_path)) and (root is not None):
            self.csv_path = os.path.join(root, csv_path)

        self.preprocessed_dir = preprocessed_dir
        if (not os.path.exists(preprocessed_dir)) and (root is not None):
            # relative dir => absolute dir
            self.preprocessed_dir = os.path.join(root, preprocessed_dir)

        self.num_frames = num_frames

        self.df = pd.read_csv(self.csv_path)
        self.scene_col = self.df.columns[0]
        self.scene_names = self.df[self.scene_col].unique()

    def getitem(self, index):
        t0 = time.time()
        # corresponds to the "video_id" in DatasetFromCSV output
        video_id = self.scene_names[index]

        preprocessed_data_path = os.path.join(self.preprocessed_dir,
                                              f"{video_id}.pt")
        data = torch.load(preprocessed_data_path)
        if self.num_frames is not None:
            data['x'] = data['x'][:, :self.num_frames]

        return data

    def __getitem__(self, index):
        for _ in range(5):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.scene_names)
