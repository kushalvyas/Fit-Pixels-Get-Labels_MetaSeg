import os, os.path as osp
import numpy as np
import glob
import torch


import torch.nn as nn 
import ipdb
from ipdb import set_trace as debug
from pdb import set_trace as debug

class MSE_and_CELoss(nn.Module):
    def __init__(self, alpha=1., beta=1., device=torch.device("cuda")):
        super(MSE_and_CELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss().to(device)
        self.ce = nn.CrossEntropyLoss(reduction='mean').to(device)

    def forward(self, data_packet):
        gt_img = data_packet['gt']
        gt_seg = data_packet['seg']
        inr_output = data_packet['output']['inr_output']
        segmentation_output = data_packet['output']['segmentation_output']
        seg_probs = segmentation_output
        # seg_probs = nn.functional.softmax(segmentation_output, dim=-1)
        # print(seg_probs[0,0], gt_seg[0,0])
        # print(f'{segmentation_output.shape=}, {seg_probs.shape=}, {gt_seg.shape=}, {inr_output.shape=}, {gt_img.shape=}')
        mse_loss = self.mse(inr_output, gt_img)
        ce_loss = self.ce(seg_probs.permute(0,2,1), gt_seg.permute(0,2,1))
        


        total_loss = self.alpha * mse_loss + self.beta * ce_loss
        # print('mse loss:', mse_loss.item(), 'ce loss:', ce_loss.item())
        return total_loss, {'loss' : total_loss, 
                            'mse_loss' : float(mse_loss.item()), 
                            'ce_loss' : float(ce_loss.item())
                            }
    

class DiceLoss(nn.Module):
    def __init__(self, alpha=1.0, smoothing = 1e-5):
        super(DiceLoss, self).__init__()
        self.smoothing = smoothing
        self.alpha = alpha

    def forward(self, prediction, target):
        pred_probs = nn.functional.softmax(prediction, dim=1)
        target_probs = target # already one hot encoded

        intersection = torch.sum(pred_probs * target_probs)
        dice = (2. * intersection + self.smoothing) / (torch.sum(pred_probs) + torch.sum(target_probs) + self.smoothing)

        return (1 - dice) * self.alpha
    


class DiceLossV2(nn.Module):
    def __init__(self, alpha=1.0, smoothing = 1e-5):
        super(DiceLossV2, self).__init__()
        self.smoothing = smoothing
        self.alpha = alpha

    def forward(self, data_packet):
        # inr_output = data_packet['output']['inr_output']
        prediction = data_packet['output']['segmentation_output']
        target =data_packet['seg']
        pred_probs = nn.functional.softmax(prediction, dim=1)
        target_probs = target # already one hot encoded

        intersection = torch.sum(pred_probs * target_probs)
        dice = (2. * intersection + self.smoothing) / (torch.sum(pred_probs) + torch.sum(target_probs) + self.smoothing)

        return (1 - dice) * self.alpha
    

import monai.losses
import torch.nn.functional as F

class DiceLossMonai(nn.Module):
    def __init__(self, num_classes=5, res=192):
        super(DiceLossMonai, self).__init__()

        self.num_classes = num_classes
        self.res=res
        self.dice_loss = monai.losses.DiceLoss(to_onehot_y=False, softmax=False, include_background=True, reduction='mean')

    def forward(self, data_packet):
        pred = data_packet['output']['segmentation_output'].reshape(-1, self.res, self.res, self.num_classes).permute(0, 3, 1, 2)
        target = data_packet['seg'].reshape(-1, self.res, self.res, self.num_classes).permute(0,3, 1, 2)
        pred_sfx = F.softmax(pred, dim=1)
        dice_loss = self.dice_loss(pred_sfx, target)

        return dice_loss
    
class MSELoss(nn.Module):
    def __init__(self, alpha=1.0, device=torch.device('cuda'), reduction='mean', zero_weight=0.01):
        super(MSELoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.zero_weight = zero_weight
        if self.reduction in ['sum' , 'mean']:
            self.mse_loss = nn.MSELoss().to(device)
        else:
            self.mse_loss = nn.MSELoss(reduction='none').to(device)
    
    def forward(self, data_packet):
        gt_img = data_packet['gt']
        seg_int = data_packet['seg_integer']
        inr_output = data_packet['output']['inr_output']
        if self.reduction in ['mean', 'sum']:
            mse_loss = self.mse_loss(inr_output, gt_img)
        else:
            weights = torch.where(seg_int == 0, self.zero_weight, 1.0)
            mse_loss_main = weights*torch.abs(inr_output - gt_img)**2
            mse_loss = torch.sum(mse_loss_main)/torch.sum(weights)
        return mse_loss * self.alpha



class MSE_CE_Dice_Loss(nn.Module):
    def __init__(self, alpha=1., beta=1., gamma=1., device=torch.device("cuda")):
        super(MSE_CE_Dice_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss().to(device)
        self.ce = nn.CrossEntropyLoss(reduction='mean').to(device)
        self.diceloss = DiceLoss().to(device)

    def forward(self, data_packet):
        gt_img = data_packet['gt']
        gt_seg = data_packet['seg']
        inr_output = data_packet['output']['inr_output']
        segmentation_output = data_packet['output']['segmentation_output']
        seg_probs = segmentation_output
        # seg_probs = nn.functional.softmax(segmentation_output, dim=-1)
        # print(seg_probs[0,0], gt_seg[0,0])
        # print(f'{segmentation_output.shape=}, {seg_probs.shape=}, {gt_seg.shape=}, {inr_output.shape=}, {gt_img.shape=}')
        mse_loss = self.mse(inr_output, gt_img)
        ce_loss = self.ce(seg_probs.permute(0,2,1), gt_seg.permute(0,2,1))
        
        dice_loss = self.diceloss(data_packet['output']['segmentation_output'], data_packet['seg'])

        total_loss = self.alpha * mse_loss + self.beta * ce_loss + self.gamma * dice_loss
        # print('mse loss:', mse_loss.item(), 'ce loss:', ce_loss.item())
        return total_loss, {'loss' : total_loss, 
                            'mse_loss' : float(mse_loss.item()), 
                            'ce_loss' : float(ce_loss.item()),
                            'dice_loss' : float(dice_loss.item())
                            }
    

class ContrastiveLosses(nn.Module):
    def __init__(self, mode):
        super(ContrastiveLosses, self).__init__()
        self.mode = mode
    
    def contrastive_loss(self, data_packet):
        raise NotImplementedError

    def triplet_loss(self, data_packet):
        features = data_packet['output']['inr_features']
        seg_labels_coord_triplets = data_packet['seg_label_coord_triplets_random']

        loss = 0.0
        #  compute positive score between anchor and pos labels
        for key in seg_labels_coord_triplets.keys():
            anchor = seg_labels_coord_triplets[key]['anchor']
            pos = seg_labels_coord_triplets[key]['positive']
            neg = seg_labels_coord_triplets[key]['negative']

            for i in range(len(anchor)):
                anchor_features = features[:, anchor[i], :]
                pos_features = features[:, pos[i], :]
                neg_features = features[:, neg[i], :]

                triplet_score = torch.norm(anchor_features - pos_features) - torch.norm(anchor_features - neg_features)
                loss += nn.functional.relu(triplet_score)

        # print('triplet loss=', loss.item())
        return loss
    
    def kl_div(self, data_packet):
        features = data_packet['output']['inr_features']
        seg_labels_coord_triplets = data_packet['seg_label_coord_triplets_random']

        loss = 0.0
        #  compute positive score between anchor and pos labels
        for key in seg_labels_coord_triplets.keys():
            anchor = seg_labels_coord_triplets[key]['anchor']
            pos = seg_labels_coord_triplets[key]['positive']
            neg = seg_labels_coord_triplets[key]['negative']

            for i in range(len(anchor)):
                anchor_features = features[:, anchor[i], :]
                pos_features = features[:, pos[i], :]
                neg_features = features[:, neg[i], :]

                triplet_score = nn.functional.cosine_similarity(anchor_features.reshape(1, -1), pos_features.reshape(1, -1)) - nn.functional.cosine_similarity(anchor_features.reshape(1, -1), neg_features.reshape(1, -1))
                loss += nn.functional.relu(triplet_score)

        # print('triplet loss=', loss.item())
        return loss


    def forward(self, data_packet):
        if self.mode == 'contrastive':
            return self.contrastive_loss(data_packet)
        elif self.mode == 'kl_div':
            return self.kl_div(data_packet)
        elif self.mode == 'triplet':
            return self.triplet_loss(data_packet)
        else:
            raise NotImplementedError



class MSE_CE_Triplet(nn.Module):
    def __init__(self, alpha=1., beta=1., gamma=1.0, mode='triplet', device=torch.device("cuda")):
        super(MSE_CE_Triplet, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss().to(device)
        self.ce = nn.CrossEntropyLoss(reduction='mean').to(device)
        self.triplet = ContrastiveLosses(mode).to(device)

    def forward(self, data_packet):
        gt_img = data_packet['gt']
        gt_seg = data_packet['seg']
        inr_output = data_packet['output']['inr_output']
        segmentation_output = data_packet['output']['segmentation_output']
        seg_probs = segmentation_output
        # seg_probs = nn.functional.softmax(segmentation_output, dim=-1)
        # print(seg_probs[0,0], gt_seg[0,0])
        # print(f'{segmentation_output.shape=}, {seg_probs.shape=}, {gt_seg.shape=}, {inr_output.shape=}, {gt_img.shape=}')
        mse_loss = self.mse(inr_output, gt_img)
        ce_loss = self.ce(seg_probs.permute(0,2,1), gt_seg.permute(0,2,1))
        
        triplet_loss = self.triplet(data_packet)

        total_loss = self.alpha * mse_loss + self.beta * ce_loss + self.gamma * triplet_loss
        # print('mse loss:', mse_loss.item(), 'ce loss:', ce_loss.item())
        return total_loss, {'loss' : total_loss, 
                            'mse_loss' : float(mse_loss.item()), 
                            'ce_loss' : float(ce_loss.item()), 
                            'triplet_loss' : float(triplet_loss.item())
                    }
    


class LocallysmoothLoss(nn.Module):
    def __init__(self, type='tv', alpha=0.1):
        super(LocallysmoothLoss, self).__init__()

        self.type = type
        self.alpha = alpha

    def forward(self, data_packet):
        if self.type == 'tv':
            loss = self.tv_loss(data_packet)
        elif self.type == 'laplacian':
            loss = self.laplacian_loss(data_packet)
        else:
            raise NotImplementedError
        return loss * self.alpha
    

    def tv_loss(self, data_packet):
        # seg_output = data_packet['output']['segmentation']
        # seg_probs = nn.functional.softmax(seg_output, dim=1)
        B , N, C  = data_packet['output']['inr_output'].shape
        H, W = data_packet['resolution'].flatten()[0], data_packet['resolution'].flatten()[1]
        inr_output = data_packet['output']['inr_output'].reshape(B, H, W, C )
        grad_x = torch.abs(inr_output[:, :, :-1] - inr_output[:, :, 1:])
        grad_y = torch.abs(inr_output[:, :-1, :] - inr_output[:, 1:, :])
        loss = torch.mean(grad_x) + torch.mean(grad_y)
        return loss
    

    def laplacian(self, data_packet):
        inr_output = data_packet['output']['inr_output']
        B , N, C  = data_packet['output']['inr_output'].shape
        H, W = data_packet['resolution']
        grad_x = torch.abs(inr_output[:, :, :-1] - inr_output[:, :, 1:])
        grad_y = torch.abs(inr_output[:, :-1, :] - inr_output[:, 1:, :])
        loss = torch.mean(grad_x) + torch.mean(grad_y)
        return loss
    

class LocallysmoothSegLoss(nn.Module):
    def __init__(self, type='tv', alpha=0.1):
        super(LocallysmoothSegLoss, self).__init__()

        self.type = type
        self.alpha = alpha

    def forward(self, data_packet):
        if self.type == 'tv':
            loss = self.tv_loss(data_packet)
        elif self.type == 'laplacian':
            loss = self.laplacian_loss(data_packet)
        else:
            raise NotImplementedError
        return loss * self.alpha
    

    def tv_loss(self, data_packet):
        seg_output = data_packet['output']['segmentation_output']
        seg_probs = nn.functional.softmax(seg_output, dim=1)
        B , N, C  = seg_probs.shape
        H, W = data_packet['resolution'].flatten()[0], data_packet['resolution'].flatten()[1]
        seg_output = seg_output.reshape(B, H, W, C )
        grad_x = torch.abs(seg_output[:, :, :-1] - seg_output[:, :, 1:])
        grad_y = torch.abs(seg_output[:, :-1, :] - seg_output[:, 1:, :])
        loss = torch.mean(grad_x) + torch.mean(grad_y)
        return loss

    

class FocalSemanticLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, device=torch.device('cuda')):
        super(FocalSemanticLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = CrossEntropyLoss().to(device)

    def forward(self, data_packet):
        gt_seg = data_packet['seg']
        segmentation_output = data_packet['output']['segmentation_output']
        seg_integers = torch.argmax(gt_seg, dim=-1)
        seg_prob_log = nn.functional.log_softmax(segmentation_output, dim=-1) # B X N X Num_Classes

        grouped_seg_log_probs = torch.gather(seg_prob_log, 
                                                dim=-1, 
                                                index=seg_integers.unsqueeze(-1)).reshape(-1)
        
        scaled_grouped_probs = (1 - torch.exp(grouped_seg_log_probs)) ** self.gamma
        loss = -1 * scaled_grouped_probs * grouped_seg_log_probs

        return loss.mean()
        
class FocalSemanticLossCE(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, device=torch.device('cuda')):
        super(FocalSemanticLossCE, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = CrossEntropyLoss(reduction='none').to(device)
    
    def forward(self, data_packet):
        gt_seg = data_packet['seg']
        segmentation_output = data_packet['output']['segmentation_output']
        seg_integers = torch.argmax(gt_seg, dim=-1)
        seg_prob_log = nn.functional.log_softmax(segmentation_output, dim=-1) # B X N X Num_Classes

        grouped_seg_log_probs = torch.gather(seg_prob_log, 
                                                dim=-1, 
                                                index=seg_integers.unsqueeze(-1)).reshape(-1)
        
        scaled_grouped_probs = (1 - torch.exp(grouped_seg_log_probs)) ** self.gamma
        loss = -1 * scaled_grouped_probs * grouped_seg_log_probs

        return loss.mean()


class CrossEntropyLoss(nn.Module):
    def __init__(self, alpha=1.0, reduction='mean', device=torch.device('cuda')):
        super(CrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction).to(device)

    def forward(self, data_packet):

        gt_seg = data_packet['seg']
        segmentation_output = data_packet['output']['segmentation_output']
        seg_probs = segmentation_output
        # print(f'ce loss, {gt_seg.shape=} {seg_probs.shape=}')
        # seg_probs = nn.functional.softmax(segmentation_output, dim=-1)
        # print(seg_probs[0,0], gt_seg[0,0])
        # print(f'{segmentation_output.shape=}, {seg_probs.shape=}, {gt_seg.shape=}, {inr_output.shape=}, {gt_img.shape=}')
        ce_loss = self.ce_loss(seg_probs.permute(0,2,1), gt_seg.permute(0,2,1))
        
        return ce_loss * self.alpha



class LossFunction(nn.Module):
    def __init__(self, loss_funcs={'mse': MSELoss(alpha=1.0)},):
        super(LossFunction, self).__init__()
        self.loss_funcs = loss_funcs

    def forward(self, data_packet):
        total_loss = 0.0
        loss_dict = {}
        for key, loss_func in self.loss_funcs.items():
            loss_val = loss_func(data_packet)
            total_loss +=  loss_val
            loss_dict[key] = float(loss_val.item())
        
        return total_loss, loss_dict
    



# class FocalSemanticLoss(nn.Module):
#     def __init__(self, alpha=1.0, gamma=2.0, device=torch.device('cuda')):
#         super(FocalSemanticLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.ce_loss = CrossEntropyLoss().to(device)

#     def forward(self, data_packet):
#         gt_seg = data_packet['seg']
#         segmentation_output = data_packet['output']['segmentation_output']
#         seg_integers = torch.argmax(gt_seg, dim=-1)
#         seg_prob_log = nn.functional.log_softmax(segmentation_output, dim=-1) # B X N X Num_Classes

#         grouped_seg_log_probs = torch.gather(seg_prob_log, 
#                                                 dim=-1, 
#                                                 index=seg_integers.unsqueeze(-1)).reshape(-1)
        
#         scaled_grouped_probs = (1 - torch.exp(grouped_seg_log_probs)) ** self.gamma
#         loss = -1 * scaled_grouped_probs * grouped_seg_log_probs

#         return loss.mean()

class FeatureSimilarityLoss(nn.Module):
    def __init__(self, alpha=1, temperature=0.1, normalize_features=True, num_classes=5,resolution=192, eps=1e-8):
        super(FeatureSimilarityLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.eps = eps
        self.resolution = resolution
        self.num_classes=num_classes
        self.normalize_features = normalize_features

    def forward(self, data_packet):
        gt_seg = data_packet['seg']
        seg_integers = torch.argmax(gt_seg, dim=-1)
        seg_integers = seg_integers.reshape(-1)
        N = torch.numel(seg_integers)
        labels = seg_integers.view(N).unsqueeze(0)

        features = data_packet['output']['inr_features']
        # features = b x n x f

        # classwise_features = {}
        # for class_val in range(self.num_classes):
        #     feature_vectors = torch.gather(
        #         features, 
        #         dim=1,
        #         index=torch.where(seg_integers == class_val)
        #     )
        #     classwise_features[class_val] = feature_vectors
        
        
        features = features.reshape(N, -1) # N x C
        if self.normalize_features:
            features = nn.functional.normalize(features, dim=1) # so c-dim is normalized


        sim_mtx = torch.matmul(features, features.t())/self.temperature
        
        mask = torch.eq(labels, labels.t()).float() # [N x N]
        mask = mask - torch.eye(N, device=features.device) # remove self similarity

        sim_max, _ = torch.max(sim_mtx, dim=1, keepdim=True)
        sim_matrix = sim_mtx - sim_max.detach()

        # exp
        exp_sim = torch.exp(sim_matrix)
        denom = exp_sim.sum(dim=1)
        numerator = (exp_sim * mask).sum(dim=1)

        loss = -torch.log((numerator + self.eps)/(denom + self.eps)).mean()
        

        valid_pixels = (mask.sum(dim=1) > 0).float()
        loss = (loss * valid_pixels).sum()/(valid_pixels.sum() + self.eps)

        return loss