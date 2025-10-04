import torch
import torch.nn as nn

import torch
import numpy as np
import json
import os, os.path as osp
import sys
from datetime import datetime
import cv2
import skimage.filters, skimage.transform
import nibabel as nib


def resize_image(image, resolution, mode='nearest'):
    
    h, w = image.shape[:2]
    if h > w:
        padd_left = (h - w) // 2
        padd_right = h - w - padd_left
        image = np.pad(image, ((0, 0), (padd_left, padd_right)), mode='constant')
    elif w > h:
        padd_top = (w - h) // 2
        padd_bottom = w - h - padd_top
        image = np.pad(image, ((padd_top, padd_bottom), (0, 0)), mode='constant')
    
    h, w = image.shape[:2]
    if h != resolution or w != resolution:
        image = cv2.resize(image, resolution, interpolation=cv2.INTER_NEAREST if mode == 'nearest' else cv2.INTER_LINEAR)

    return image


def random_augment(x, y,):
    # Random flip
    choice_flip = [True,False][np.random.randint(0, 2)]
    if choice_flip:
        x = np.flip(x, axis=0)
        y = np.flip(y, axis=0)
    
    choice_rot = [1, -1, 0][np.random.randint(0, 3)]
    x = np.rot90(x, k=choice_rot)
    y = np.rot90(y, k=choice_rot)
    return x, y


from skimage.transform import rescale, rotate
def random_rotate(x, y, angle_rot):    
    # angle_rot =  15 # in degrees
    angle = np.random.uniform(low=-angle_rot, high=angle_rot)
    x = rotate(x, angle, resize=False, preserve_range=True, mode="constant")
    y = rotate(y, angle, resize=False, order=0, preserve_range=True, mode="constant")
    return x, y


def random_translate(x, y, offset):
    
    translation = np.random.randint(low=-offset, high=offset)
    x = np.roll(x, translation, axis=0)
    y = np.roll(y, translation, axis=0)
    return x, y


def seglabels_to_onehot(seglabels, num_classes=5):
    # classes = np.unique(seglabels.flatten()).tolist()
    # print(len(classes), seglabels.shape)
    onehot = np.zeros(shape=(seglabels.shape[0], seglabels.shape[1], num_classes), dtype=np.int32)
    for c in range(num_classes):
        # if c == 0:
        #     continue
        row, cols = np.where(seglabels == c)
        onehot[row, cols, int(c)] = 1
    return np.array(onehot)


class TorchMRIDataloader(torch.utils.data.Dataset):
    def __init__(self, json_file, mode=None, num_classes=4, resolution=192, transforms=None, config={}, coords=None):
        super(TorchMRIDataloader, self).__init__()
        assert num_classes == 4 or num_classes == 24, "Only 4 or 24 classes are supported"
        self.json_file = json_file
        self.mode = mode
        self.coords=coords
        self.config = config
        self.resolution = resolution
        self.transforms = transforms
        self.num_classes = num_classes
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.build()

    def build(self):
        with open(self.json_file, 'r') as f:
            _data = json.load(f)
        
        self.data = _data[self.mode]
        if self.config.get('N_samples', None) is not None:
            self.data = self.data[:self.config['N_samples']]
        self.length = len(self.data)

    def __len__(self):
        return self.length
    
    def read_nib_file(self, file_path):
        img = nib.load(file_path).get_fdata()
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        return img
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        img_path = data_dict['img'] 
        seg_path = data_dict[f'seg{self.num_classes}'] 

        img = self.read_nib_file(img_path)
        seg = self.read_nib_file(seg_path)

        img = np.clip(img, 0, None)

        img = resize_image(img, self.resolution, mode='bilinear')
        seg = resize_image(seg, self.resolution, mode='nearest')
        
        if self.config.get('augment', True):
            img, seg = random_augment(img, seg)

        seg_integer = seg.copy()
        seg_labels = seglabels_to_onehot(seg_integer, num_classes=self.num_classes + 1)

        nc = seg_labels.shape[-1]
        img_tensor = torch.from_numpy(np.array(img.reshape(-1, 1))).float() # C x H x W
        seg_labels_tensor = torch.from_numpy(np.array(seg_labels.reshape(-1, nc))).long() # C x H x W
        seg_integer_tensor = torch.from_numpy(np.array(seg_integer.reshape(-1, 1))).long()  # C x H x W

        return {
            'img' : img_tensor,
            'seg_onehot' : seg_labels_tensor,
            'seg_integer' : seg_integer_tensor,
            'coords' : self.coords,
            'resolution' : torch.from_numpy(np.array(self.resolution)).reshape(1, -1),
            'seg' : seg_labels_tensor,
            'seg_integer' : seg_integer_tensor, 
        }
    



class TorchMRIDataloaderAugment(torch.utils.data.Dataset):
    def __init__(self, json_file, mode='train', num_classes=4, resolution=192, transforms=None, config={}, coords=None, rotate=None,translate=None):
        super(TorchMRIDataloaderAugment, self).__init__()
        assert num_classes == 4 or num_classes == 24, "Only 4 or 24 classes are supported"
        self.json_file = json_file
        self.mode = mode
        self.coords=coords
        self.config = config
        self.resolution = resolution
        self.rotate = rotate
        self.translate = translate
        self.transforms = transforms
        self.num_classes = num_classes
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.build()

    def build(self):
        with open(self.json_file, 'r') as f:
            _data = json.load(f)
        
        self.data = _data[self.mode]
        if self.config.get('N_samples', None) is not None:
            self.data = self.data[:self.config['N_samples']]
        self.length = len(self.data)

    def __len__(self):
        return self.length
    
    def read_nib_file(self, file_path):
        img = nib.load(file_path).get_fdata()
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        return img
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        img_path = data_dict['img'] 
        seg_path = data_dict[f'seg{self.num_classes}'] 

        img = self.read_nib_file(img_path)
        seg = self.read_nib_file(seg_path)

        img = np.clip(img, 0, None)

        img = resize_image(img, self.resolution, mode='bilinear')
        seg = resize_image(seg, self.resolution, mode='nearest')
        
        if self.rotate:
            img, seg = random_rotate(img, seg, self.rotate)

        if self.translate:
            img, seg = random_translate(img, seg, self.translate)

        seg_integer = seg.copy()
        seg_labels = seglabels_to_onehot(seg_integer, num_classes=self.num_classes + 1)

        nc = seg_labels.shape[-1]
        img_tensor = torch.from_numpy(np.array(img.reshape(-1, 1))).float() # C x H x W
        seg_labels_tensor = torch.from_numpy(np.array(seg_labels.reshape(-1, nc))).long() # C x H x W
        seg_integer_tensor = torch.from_numpy(np.array(seg_integer.reshape(-1, 1))).long()  # C x H x W

        return {
            'img' : img_tensor,
            'seg_onehot' : seg_labels_tensor,
            'seg_integer' : seg_integer_tensor,
            'coords' : self.coords,
            'resolution' : torch.from_numpy(np.array(self.resolution)).reshape(1, -1),
            'seg' : seg_labels_tensor,
            'seg_integer' : seg_integer_tensor, 
        }