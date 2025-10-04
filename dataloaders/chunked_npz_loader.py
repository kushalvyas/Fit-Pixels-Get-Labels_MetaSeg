import numpy as np
import torch
import torch.nn as nn
import os, os.path as osp
import glob


class ChunkedLoader(torch.utils.data.Dataset):
    def __init__(self, path, mode):
        super(ChunkedLoader, self).__init__()
        self.path = path
        self.mode = mode
        self.base_path= osp.join(self.path, self.mode)

        self.all_files = glob.glob(osp.join(self.base_path, "*.npz"))
        self.length = len(self.all_files)
    def __len__(self):
        return self.length
        

    def __getitem__(self, idx):
        npz_file = self.all_files[idx]

        data = np.load(npz_file)

        img = data['img']
        features = data['features']
        seg_int = data['seg_int']
        seg_onehot = data['seg_onehot']

        img_tensor = torch.from_numpy(img).float()
        features_tensor = torch.from_numpy(features).float()
        seg_int_tensor = torch.from_numpy(seg_int).long()
        seg_onehot_tensor = torch.from_numpy(seg_onehot).long()

        return {
            'img': img_tensor,
            'features': features_tensor,
            'seg_int': seg_int_tensor,
            'seg_onehot': seg_onehot_tensor
        }