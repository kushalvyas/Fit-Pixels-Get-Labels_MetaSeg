import torch
import torch.nn as nn
import numpy as np
import glob
import os, os.path as osp
from collections import OrderedDict, defaultdict
import nibabel as nib

def load_nii_file(filename):
    data = nib.load(filename)
    return data.get_fdata()[...,0]

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


def random_augment(x, y):
    # Random flip
    choice_flip = [True,False][np.random.randint(0, 2)]
    if choice_flip:
        x = np.flip(x, axis=0)
        y = np.flip(y, axis=0)
    
    choice_rot = [1, -1, 0][np.random.randint(0, 3)]
    x = np.rot90(x, k=choice_rot)
    y = np.rot90(y, k=choice_rot)
    return x, y

def pad_resize(im, res):
    assert isinstance(res, list), "resolution must be a list"
    return np.pad(im, ((0, res[0] - im.shape[0]), (0, res[1] - im.shape[1])), mode='constant', constant_values=0)
    
def load_img_seg_pairs(x_paths, y_paths, resolution=None):
    for im_path, label_path in zip(x_paths, y_paths):
        img = load_nii_file(im_path)
        seg = load_nii_file(label_path)
        if resolution is not None:
            img = pad_resize(img, resolution)
            seg = pad_resize(seg, resolution)

        yield img, seg


def get_seg_label_coord_lookup(seg):
    seg = seg.flatten()
    seg_label_coord = defaultdict(list)
    for i, s in enumerate(seg):
        seg_label_coord[s].append(i)
    

    return seg_label_coord


def get_random_triplet_indexes(seg_coord_lookup, N):
    classwise_triplet_indexes = defaultdict(lambda x: defaultdict(list))
    for k in seg_coord_lookup.keys():
        neg_keys = [key for key in seg_coord_lookup.keys() if key != k]
        classwise_triplet_indexes[k] = {'anchor': [], 'positive': [], 'negative': []}
        for i in range(N):
            pos_anch_pair = np.random.choice(seg_coord_lookup[k], size=2)
            anchor, positive = pos_anch_pair
            classwise_triplet_indexes[k]['anchor'].append(anchor)
            classwise_triplet_indexes[k]['positive'].append(positive)
            neg_sample_key = np.random.choice(neg_keys)
            neg = np.random.choice(seg_coord_lookup[neg_sample_key],size=1)[0]
            classwise_triplet_indexes[k]['negative'].append(neg)
            

    return classwise_triplet_indexes
