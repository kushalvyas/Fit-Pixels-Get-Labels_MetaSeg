import torch
import torch.utils
import torch.utils.data as torch_data

import numpy as np
import nibabel as nib
import json
import os, os.path as osp

import libINR.utils.coords
import torch.nn.functional as F



class TorchMRI3D_Dataloader(torch.utils.data.Dataset):
    def __init__(self, json_file, mode='train', num_classes=4, resolution=None, transforms=None, config={}, coords=None, skip_pixels=1.0):
        super(TorchMRI3D_Dataloader, self).__init__()
        assert num_classes == 4 or num_classes == 24 or num_classes == 35, "Only 4 or 24 classes are supported"
        self.json_file = json_file
        self.mode = mode
        self.coords=coords
        self.config = config
        self.resolution = resolution
        self.transforms = transforms
        self.num_classes = num_classes
        self.skip_pixels = skip_pixels
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
    
    def read_nib_volume(self, file_path):
        img = nib.load(file_path)
        # print('img shape = ', img.get_fdata().shape)
        return img.get_fdata()[:, 16:-16, 12:-12] # crops
    
    def get_coords(self, h, w, d):
        xx = torch.linspace(-1, 1, h)
        yy = torch.linspace(-1, 1, w)
        zz = torch.linspace(-1, 1, d)

        coords = torch.meshgrid(xx, yy, zz)
        coords = torch.stack(coords, dim=-1)
        return coords


    def sample_from_3d(self, volume, segmentation):
        # volume = np.reshape(volume, (-1, 1))
        # segmentation = np.reshape(segmentation, (-1, 1))
        # # selection_indices = np.random.choice(np.arange(volume.shape[0]), int(self.sample_fraction * volume.shape[0]), replace=False)
        # selection_indices = np.arange(volume.shape[0])[::int(1/self.sample_fraction)]
        # volume = volume[selection_indices,...]
        # segmentation = segmentation[selection_indices,...]
        # return volume, segmentation, selection_indices

        h,w,d = volume.shape

        coords_mtx = self.get_coords(h,w,d)
        volume_sampled = volume[::self.skip_pixels, ::self.skip_pixels, ::self.skip_pixels]
        segmentation_sampled = segmentation[::self.skip_pixels, ::self.skip_pixels, ::self.skip_pixels]
        coords_sampled = coords_mtx[::self.skip_pixels, ::self.skip_pixels, ::self.skip_pixels, ...]
        
        # #random
        # random_h = np.random.choice(np.arange(h), int(h*self.sample_fraction), replace=False)
        # random_w = np.random.choice(np.arange(w), int(h*self.sample_fraction), replace=False)
        # random_d = np.random.choice(np.arange(d), int(h*self.sample_fraction), replace=False)
        # volume_sampled = volume[random_h, random_w, random_d]
        # segmentation_sampled = segmentation[random_h, random_w, random_d]
        # coords_sampled = coords_mtx[random_h, random_w, random_d, ...]

        vs, ss, cs = volume_sampled.reshape(-1, 1), segmentation_sampled.reshape(-1, 1), coords_sampled.reshape(-1, 3)
        assert vs.shape[0] == ss.shape[0] == cs.shape[0], 'Shape mismatch for vs, ss, and cs'
        return vs, ss, cs

    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        vol_path = data_dict['img'] # key is img. but its actually volume
        seg_path = data_dict[f'seg{self.num_classes}'] 

        volume = self.read_nib_volume(vol_path)
        segmentation_integers = self.read_nib_volume(seg_path)
        coords = self.coords.clone()
        # print('coords shape = ', coords.shape, 'volume shape = ', volume.shape, 'seg shape = ', segmentation_integers.shape)
        if self.skip_pixels != 1.0:
            volume, segmentation_integers, coords = self.sample_from_3d(volume, segmentation_integers)
        else:
            coords = self.get_coords(volume.shape[0], volume.shape[1], volume.shape[2]).reshape(-1, 3)
            volume = volume.reshape(-1, 1)
            segmentation_integers = segmentation_integers.reshape(-1, 1)
            
        
        seg_integer = segmentation_integers.copy()  # H X W X D
        seg_onehot_labels = F.one_hot(torch.from_numpy(seg_integer).long(), 
                                      num_classes=self.num_classes + 1).float() # H x W x D x NUM_CLASSES
        
        data_dict = {
            'img': torch.from_numpy(volume),
            'seg': seg_onehot_labels.squeeze(-2),
            'seg_integer': torch.from_numpy(seg_integer),
            'coords' : coords,
            'resolution': volume.shape,
        }
        return data_dict
        

import glob
import time

class CLFFeature(torch.utils.data.Dataset):
    def __init__(self, path, mode):
        super(CLFFeature, self).__init__()
        self.path = path
        self.mode = mode

        self.all_files = glob.glob(osp.join(self.path, self.mode ,'*.pth'))
        self.length = len(self.all_files)

    def __len__(self):  
        return self.length
    
    def __getitem__(self, idx):
        st = time.time()
        data = torch.load(self.all_files[idx])
        et = time.time()
        # print('time to load = ' ,et-st)

        return data
    


class TorchMRI3D_DataloaderFast(torch.utils.data.Dataset):
    def __init__(self, json_file, mode='train', num_classes=4, resolution=None, transforms=None, config={}, coords=None, skip_pixels=1.0):
        super(TorchMRI3D_DataloaderFast, self).__init__()
        assert num_classes == 4 or num_classes == 24 or num_classes == 35, "Only 4 or 24 classes are supported"
        self.json_file = json_file
        self.mode = mode
        self.coords=coords
        self.config = config
        self.resolution = resolution
        self.transforms = transforms
        self.num_classes = num_classes
        self.skip_pixels = skip_pixels
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
    
    def read_nib_volume(self, file_path):
        img = nib.load(file_path)
        # print('img shape = ', img.get_fdata().shape)
        return img.get_fdata()[:, 16:-16, 12:-12] # crops
    
    def get_coords(self, h, w, d):
        xx = torch.linspace(-1, 1, h)
        yy = torch.linspace(-1, 1, w)
        zz = torch.linspace(-1, 1, d)

        coords = torch.meshgrid(xx, yy, zz)
        coords = torch.stack(coords, dim=-1)
        return coords


    def sample_from_3d(self, volume, segmentation):
        # volume = np.reshape(volume, (-1, 1))
        # segmentation = np.reshape(segmentation, (-1, 1))
        # # selection_indices = np.random.choice(np.arange(volume.shape[0]), int(self.sample_fraction * volume.shape[0]), replace=False)
        # selection_indices = np.arange(volume.shape[0])[::int(1/self.sample_fraction)]
        # volume = volume[selection_indices,...]
        # segmentation = segmentation[selection_indices,...]
        # return volume, segmentation, selection_indices

        h,w,d = volume.shape

        coords_mtx = self.get_coords(h,w,d)
        volume_sampled = volume[::self.skip_pixels, ::self.skip_pixels, ::self.skip_pixels]
        segmentation_sampled = segmentation[::self.skip_pixels, ::self.skip_pixels, ::self.skip_pixels]
        coords_sampled = coords_mtx[::self.skip_pixels, ::self.skip_pixels, ::self.skip_pixels, ...]
        
        # #random
        # random_h = np.random.choice(np.arange(h), int(h*self.sample_fraction), replace=False)
        # random_w = np.random.choice(np.arange(w), int(h*self.sample_fraction), replace=False)
        # random_d = np.random.choice(np.arange(d), int(h*self.sample_fraction), replace=False)
        # volume_sampled = volume[random_h, random_w, random_d]
        # segmentation_sampled = segmentation[random_h, random_w, random_d]
        # coords_sampled = coords_mtx[random_h, random_w, random_d, ...]

        # vs, ss, cs = volume_sampled.reshape(-1, 1), segmentation_sampled.reshape(-1, 1), coords_sampled.reshape(-1, 3)
        # assert vs.shape[0] == ss.shape[0] == cs.shape[0], 'Shape mismatch for vs, ss, and cs'
        # return vs, ss, cs

        return volume_sampled, segmentation_sampled, coords_sampled

        
        
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        vol_path = data_dict['img'] # key is img. but its actually volume
        seg_path = data_dict[f'seg{self.num_classes}'] 

        volume = self.read_nib_volume(vol_path)
        segmentation_integers = self.read_nib_volume(seg_path)
        coords = self.coords.clone()
        # print('coords shape = ', coords.shape, 'volume shape = ', volume.shape, 'seg shape = ', segmentation_integers.shape)
        if self.skip_pixels != 1.0:
            volume, segmentation_integers, coords = self.sample_from_3d(volume, segmentation_integers)
        else:
            coords = self.get_coords(volume.shape[0], volume.shape[1], volume.shape[2])#.reshape(-1, 3)
            volume = volume#.reshape(-1, 1)
            segmentation_integers = segmentation_integers#.reshape(-1, 1)
            
        
        seg_integer = segmentation_integers.copy()  # H X W X D
        seg_onehot_labels = F.one_hot(torch.from_numpy(seg_integer).long(), 
                                      num_classes=self.num_classes + 1).float() # H x W x D x NUM_CLASSES
        
        data_dict = {
            'img': torch.from_numpy(volume),
            'seg': seg_onehot_labels,
            'seg_integer': torch.from_numpy(seg_integer),
            'coords' : coords,
            'resolution': volume.shape,
        }
        return data_dict
        

class TorchMRI3D_Dataloader_SR(torch.utils.data.Dataset):
    def __init__(self, json_file, mode='train', num_classes=4, resolution=None, transforms=None, config={}, coords=None, skip_pixels=1.0):
        super(TorchMRI3D_Dataloader_SR, self).__init__()
        assert num_classes == 4 or num_classes == 24 or num_classes == 35, "Only 4 or 24 classes are supported"
        self.json_file = json_file
        self.mode = mode
        self.coords=coords
        self.config = config
        self.resolution = resolution
        self.transforms = transforms
        self.num_classes = num_classes
        self.skip_pixels = skip_pixels
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
    
    def read_nib_volume(self, file_path):
        img = nib.load(file_path)
        # print('img shape = ', img.get_fdata().shape)
        return img.get_fdata()[:, 16:-16, 12:-12] # crops
    
    def get_coords(self, h, w, d):
        xx = torch.linspace(-1, 1, h)
        yy = torch.linspace(-1, 1, w)
        zz = torch.linspace(-1, 1, d)

        coords = torch.meshgrid(xx, yy, zz)
        coords = torch.stack(coords, dim=-1)
        return coords


    def sample_from_3d(self, volume, segmentation):
        # volume = np.reshape(volume, (-1, 1))
        # segmentation = np.reshape(segmentation, (-1, 1))
        # # selection_indices = np.random.choice(np.arange(volume.shape[0]), int(self.sample_fraction * volume.shape[0]), replace=False)
        # selection_indices = np.arange(volume.shape[0])[::int(1/self.sample_fraction)]
        # volume = volume[selection_indices,...]
        # segmentation = segmentation[selection_indices,...]
        # return volume, segmentation, selection_indices

        h,w,d = volume.shape

        coords_mtx = self.get_coords(h,w,d)
        volume_sampled = volume[::self.skip_pixels, ::self.skip_pixels, ::self.skip_pixels]
        segmentation_sampled = segmentation[::self.skip_pixels, ::self.skip_pixels, ::self.skip_pixels]
        coords_sampled = coords_mtx[::self.skip_pixels, ::self.skip_pixels, ::self.skip_pixels, ...]
        
        # #random
        # random_h = np.random.choice(np.arange(h), int(h*self.sample_fraction), replace=False)
        # random_w = np.random.choice(np.arange(w), int(h*self.sample_fraction), replace=False)
        # random_d = np.random.choice(np.arange(d), int(h*self.sample_fraction), replace=False)
        # volume_sampled = volume[random_h, random_w, random_d]
        # segmentation_sampled = segmentation[random_h, random_w, random_d]
        # coords_sampled = coords_mtx[random_h, random_w, random_d, ...]

        vs, ss, cs = volume_sampled.reshape(-1, 1), segmentation_sampled.reshape(-1, 1), coords_sampled.reshape(-1, 3)
        assert vs.shape[0] == ss.shape[0] == cs.shape[0], 'Shape mismatch for vs, ss, and cs'
        return vs, ss, cs


        
        
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        vol_path = data_dict['img'] # key is img. but its actually volume
        seg_path = data_dict[f'seg{self.num_classes}'] 

        volume = self.read_nib_volume(vol_path)
        segmentation_integers = self.read_nib_volume(seg_path)
        coords = self.coords.clone()

        coords_hr = self.get_coords(volume.shape[0], volume.shape[1], volume.shape[2]).clone().reshape(-1, 3)
        volume_hr = volume.copy().reshape(-1, 1)
        segmentation_integers_hr = segmentation_integers.copy().reshape(-1, 1)
        # print('coords shape = ', coords.shape, 'volume shape = ', volume.shape, 'seg shape = ', segmentation_integers.shape)
        if self.skip_pixels != 1.0:
            volume, segmentation_integers, coords = self.sample_from_3d(volume, segmentation_integers)
        else:
            coords = self.get_coords(volume.shape[0], volume.shape[1], volume.shape[2]).reshape(-1, 3)
            volume = volume.reshape(-1, 1)
            segmentation_integers = segmentation_integers.reshape(-1, 1)
            
        
        seg_integer = segmentation_integers.copy()  # H X W X D
        seg_onehot_labels = F.one_hot(torch.from_numpy(seg_integer).long(), 
                                      num_classes=self.num_classes + 1).float() # H x W x D x NUM_CLASSES
        
        seg_onehot_labels_hr = F.one_hot(torch.from_numpy(segmentation_integers_hr).long(), 
                                      num_classes=self.num_classes + 1).float() # H x W x D x NUM_CLASSES
        
        data_dict = {
            'img': torch.from_numpy(volume),
            'seg': seg_onehot_labels.squeeze(-2),
            'seg_integer': torch.from_numpy(seg_integer),
            'coords' : coords,
            'resolution': volume.shape,
            'coords_hr' : coords_hr,
            'img_hr': torch.from_numpy(volume_hr),
            'seg_hr': seg_onehot_labels_hr.squeeze(-2),
            'seg_integer_hr': torch.from_numpy(segmentation_integers_hr),
        }
        return data_dict
        