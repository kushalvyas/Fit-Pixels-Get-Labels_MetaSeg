"""Custom MAML implementation for MetaSeg
author: Kushal Vyas
"""
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

import logging
from copy import deepcopy


class INRMetaLearner():
    def __init__(self, model, inner_steps, config={}, custom_loss_fn=None, outer_optimizer='adam', inner_loop_loss_fn=None):
        super(INRMetaLearner, self).__init__()
        self.model = model
        self.inner_steps = inner_steps
        self.config = config

        self.model_params = OrderedDict({
            k: v.clone().detach().requires_grad_() for k, v in self.model.named_parameters()
        })

        self.outer_optimizer = outer_optimizer
        self.configure_optimizers()

        self.loss_fn = custom_loss_fn if custom_loss_fn is not None else self.loss_fn_mse
        self.inner_loop_loss_fn = inner_loop_loss_fn

    def configure_optimizers(self):
        lr = self.config.get("outer_lr", 1e-4)
        if self.outer_optimizer == 'adam':
            self.opt_thetas = torch.optim.Adam(self.model_params.values(), lr=lr)
        else:
            self.opt_thetas = torch.optim.SGD(self.model_params.values(), lr=lr)

    def get_parameters(self, copy=True):
        return OrderedDict({
            k: v.clone().detach() if copy else v for k, v in self.model_params.items()
        })

    def get_inr_parameters(self, key_prefix="inr", copy=True):
        return OrderedDict({
            k: v.clone().detach() if copy else v for k, v in self.model_params.items() if key_prefix in k
        })

    def get_segmentation_parameters(self, key_prefix='segmentation', copy=True):
        return OrderedDict({
            k: v.clone().detach() if copy else v for k, v in self.model_params.items() if key_prefix in k
        })

    def set_parameters(self, params):
        self.model_params.update(params)

    def mse_loss(self, x, y):
        return nn.functional.mse_loss(x, y)

    def loss_fn_mse(self, data_packet):
        mse_loss = self.mse_loss(data_packet['output']['inr_output'], data_packet['gt'])
        return mse_loss, {'mse_loss': float(mse_loss)}

    def squeeze_output(self, output, gt):
        if isinstance(output, dict):
            for k, v in output.items():
                if v.dim() != gt.dim():
                    output[k] = v.squeeze(1)
        elif isinstance(output, torch.Tensor):
            if output.dim() != gt.dim():
                output = output.squeeze(1)
        return output

    def adam_update(self, params, grads, opt_step, adam_m, adam_v, lr=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8):
        updated_params = OrderedDict()
        for ((k,p), g) in zip(params.items(), grads):
            adam_m[k] = beta1 * adam_m[k] + (1 - beta1) * g
            adam_v[k] = beta2 * adam_v[k] + (1 - beta2) * (g**2)

            m_hat = adam_m[k] / (1 - beta1 ** (opt_step))
            v_hat = adam_v[k] / (1 - beta2 ** (opt_step))
            v_hat = torch.clamp(v_hat, min=1e-12)
            updated_p = p - lr * m_hat / (torch.sqrt(v_hat) + epsilon)
    
            updated_params[k] = updated_p.requires_grad_()
            
        return updated_params, adam_m, adam_v

    def inner_loop(self, coords, data_packet, create_graph=True):
        gt = data_packet['gt']
        params = OrderedDict((k, v.clone()) for k, v in self.model_params.items())  # initial θ
        
        adam_m = {k: torch.zeros_like(v) for k, v in params.items()}
        adam_v = {k: torch.zeros_like(v) for k, v in params.items()}

        for i in range(self.inner_steps):
            output = torch.func.functional_call(self.model, params, coords)
            output = self.squeeze_output(output, gt)
            data_packet['output'] = output

            if self.inner_loop_loss_fn is not None:
                loss, _ = self.inner_loop_loss_fn(data_packet)
            else:
                loss, _ = self.loss_fn(data_packet)
            if torch.isnan(loss).any():
                print("Loss is NaN, skipping this step")
                break
            grads = torch.autograd.grad(loss, params.values(), create_graph=True)
            
            # Perform Adam update
            lr = self.config.get("inner_lr", 1e-4)
            params, adam_m, adam_v = self.adam_update(params, grads, i+1, adam_m, adam_v, lr=lr)            
            logging.info(f"Inner step: {i+1}/{self.inner_steps}, Loss: {loss.item():.6f}")
        return params

    def outer_loop(self, coords, data_packet):
        self.opt_thetas.zero_grad()
        gt = data_packet['gt']
        adapted_params = self.inner_loop(coords, data_packet, create_graph=True)

        output = torch.func.functional_call(self.model, adapted_params, coords)
        output = self.squeeze_output(output, gt)
        data_packet['output'] = output
        loss, loss_info = self.loss_fn(data_packet)

        loss.backward()
        self.opt_thetas.step()
        return loss, loss_info

    def forward(self, coords, data_packet):
        return self.outer_loop(coords, data_packet)

    def render_inner_loop(self, coords, gt, inner_loop_steps=1):
        model_copy = deepcopy(self.model)
        params = OrderedDict({
            k: v.clone().detach().requires_grad_(True) for k, v in self.model_params.items() if "inr" in k
        })
        opt_render = torch.optim.Adam(params.values(), lr=self.config.get("inner_lr", 1e-4))

        for _ in range(inner_loop_steps):
            opt_render.zero_grad()
            output = torch.func.functional_call(model_copy, params, coords)
            loss = nn.functional.mse_loss(output['inr_output'], gt)
            loss.backward()
            opt_render.step()

        output = self.squeeze_output(output, gt)
        return {'output': output}
