import numpy as np
import os, os.path as osp
import torch

from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
import torch.nn as nn
from torch.nn import functional as F


import pdb
import math

import numpy as np

import torch
from torch import nn

from modules.base import BaseINR
    
class SineLayer(nn.Module):
    '''
        See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
        discussion of omega_0.
    
        If is_first=True, omega_0 is a frequency factor which simply multiplies
        the activations before the nonlinearity. Different signals may require
        different omega_0 in the first layer - this is a hyperparameter.
    
        If is_first=False, then the weights will be divided by omega_0 so as to
        keep the magnitude of activations constant, but boost gradients to the
        weight matrix (see supplement Sec. 1.5)
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0, init_weights=True):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        if init_weights:
            self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class INR(BaseINR):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, 
                 out_features, outermost_linear=True,
                 first_omega_0=30, hidden_omega_0=30., scale=10.0,
                 pos_encode=False, sidelength=512, fn_samples=None,
                 use_nyquist=True):
        super().__init__()
        self.pos_encode = pos_encode
        self.nonlin = SineLayer
            
        self.net_w_layers = []
        self.net_w_layers.append(self.nonlin(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0,
                                  scale=scale))

        for i in range(hidden_layers):
            self.net_w_layers.append(self.nonlin(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))

        if outermost_linear:
            dtype = torch.float
            final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
            
            with torch.no_grad():
                const = np.sqrt(6/hidden_features)/max(hidden_omega_0, 1e-12)
                final_linear.weight.uniform_(-const, const)
                    
            self.net_w_layers.append(final_linear)
        else:
            self.net_w_layers.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0,
                                      scale=scale))
        
        self.net = nn.Sequential(*self.net_w_layers)
    
    def forward(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)
            
        output = self.net(coords)
                    
        return output
    
    def forward_w_features(self, coords):
        if self.pos_encode:
            coords = self.positional_encoding(coords)
            
        features = []
        x = coords.clone()
        for layer in self.net_w_layers:
            x = layer(x)
            features.append(x)

        output = x.clone()                 
        return output, features

    def load_weights(self, weights):
        self.load_state_dict(weights)



class SegmentationModel(nn.Module):
    def __init__(self, inr_config, segmentation_config):
        super(SegmentationModel, self).__init__()
        seg_layers = [nn.Linear(inr_config['hidden_features'], segmentation_config['hidden_features'][0]), nn.LeakyReLU()]
        for i in range(1, len(segmentation_config['hidden_features'])):
            seg_layers.append(nn.Linear(segmentation_config['hidden_features'][i-1], segmentation_config['hidden_features'][i]))
            # seg_layers.append(nn.BatchNorm1d(segmentation_config['hidden_features'][i]))
            seg_layers.append(nn.LeakyReLU())
        seg_layers.append(nn.Linear(segmentation_config['hidden_features'][-1], segmentation_config['output_features']))
        self.segmentation_head = nn.Sequential(*seg_layers)

    def forward(self, features):
        return self.segmentation_head(features)

class SirenSegINR(nn.Module):
    def __init__(self, inr_type, inr_config, segmentation_config, normalize_features=False):
        super(SirenSegINR, self).__init__()

        # self.inr = libINR.models.make_inr_model(inr_type, **inr_config)
        self.inr = INR(**inr_config)
        # seg_layers = [nn.Linear(inr_config['hidden_features'], segmentation_config['hidden_features'][0]), nn.LeakyReLU()]
        # for i in range(1, len(segmentation_config['hidden_features'])):
        #     seg_layers.append(nn.Linear(segmentation_config['hidden_features'][i-1], segmentation_config['hidden_features'][i]))
        #     seg_layers.append(nn.LeakyReLU())
        # seg_layers.append(nn.Linear(segmentation_config['hidden_features'][-1], segmentation_config['output_features']))
        # self.segmentation_head = nn.Sequential(*seg_layers)
        self.normalize_features = normalize_features
        self.segmentation_head = SegmentationModel(inr_config, segmentation_config) 

        print(self.inr)
        print(self.segmentation_head)


    def forward(self, coords):
        inr_output, inr_features = self.inr.forward_w_features(coords)
        penultimate_features = inr_features[-2]
        segmentation_output = self.segmentation_head(penultimate_features)
        if self.normalize_features:
            penultimate_features = F.normalize(penultimate_features, dim=-1)
        return {'inr_output' : inr_output,
                'inr_features' : penultimate_features,
                'segmentation_output' : segmentation_output}
    
    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return self.state_dict()



class SirenINR(nn.Module):
    def __init__(self, inr_type, inr_config, segmentation_config, normalize_features=False):
        super(SirenINR, self).__init__()

        self.inr = INR(**inr_config)
        self.normalize_features = normalize_features

        print(self.inr)

    def forward(self, coords):
        inr_output, inr_features = self.inr.forward_w_features(coords)
        return {'inr_output' : inr_output}
    
    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return self.state_dict()
