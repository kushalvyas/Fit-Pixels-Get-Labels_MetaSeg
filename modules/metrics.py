import os, os.path as osp
import numpy as np
from scipy.spatial.distance import dice, jaccard

from torchmetrics.functional.segmentation import generalized_dice_score

def multiclass_dice_score(segmentation, gt, num_classes):
    # print(segmentation.shape, gt.shape, num_classes)
    assert segmentation.shape == gt.shape, f"Segmentation and ground truth must have the same shape. {segmentation.shape=} {gt.shape=}"
    assert segmentation.shape[-1] == num_classes, f"Segmentation must have the same number of classes as the number of classes. {segmentation.shape=}  {num_classes=} "
    if segmentation.shape[0] != 1:
        segmentation = segmentation[None,...]
    if gt.shape[0] != 1:
        gt = gt[None,...]
    
    segmentation = segmentation.permute(0, 3,1,2)
    gt = gt.permute(0, 3,1,2)
    
    dice_scores_per_class = generalized_dice_score(
        segmentation,
        gt,
        include_background=True,
        num_classes=num_classes,
        weight_type='simple',
    )

    return dice_scores_per_class



def multiclass_dice_score_3d(segmentation, gt, num_classes):
    # print(segmentation.shape, gt.shape, num_classes)
    assert segmentation.shape == gt.shape, f"Segmentation and ground truth must have the same shape. {segmentation.shape=} {gt.shape=}"
    assert segmentation.shape[-1] == num_classes, f"Segmentation must have the same number of classes as the number of classes. {segmentation.shape=}  {num_classes=} "
    if segmentation.shape[0] != 1:
        segmentation = segmentation[None,...]
    if gt.shape[0] != 1:
        gt = gt[None,...]
    
    segmentation = segmentation.permute(0,4,1,2,3)
    gt = gt.permute(0,4,1,2,3)
    
    dice_scores_per_class = generalized_dice_score(
        segmentation,
        gt,
        include_background=True,
        num_classes=num_classes,
        weight_type='simple',
    )

    return dice_scores_per_class



import skimage.metrics

def psnr2(x, y):
    return skimage.metrics.peak_signal_noise_ratio(x, y, data_range=1)

import torch
def dice_simple(pred, gt_onehot, num_classes=5):
    pred_probs = torch.nn.functional.softmax(pred, -1)
    pred_probs_logit = torch.nn.functional.one_hot(torch.argmax(pred_probs, dim=-1), num_classes=num_classes)
    score = generalized_dice_score(pred_probs_logit.permute(0,3,1,2), gt_onehot.permute(0,3,1,2), num_classes=num_classes, weight_type='simple')
    return score.mean()


def multiclass_dice_score_3d_for_flat(pred_onehot, gt_onehot, num_classes):
    # num_classes = -1 dim.
    pred_reshaped = pred_onehot.permute(0, 2, 1)
    gt_reshaped = gt_onehot.permute(0, 2, 1)
    dice_scores_per_class = generalized_dice_score(
        pred_reshaped,
        gt_reshaped,
        include_background=True,
        num_classes=num_classes,
        weight_type='simple',
    )
    return dice_scores_per_class