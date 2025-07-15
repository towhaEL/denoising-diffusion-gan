# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

'''
Codes adapted from https://github.com/NVlabs/LSGM/blob/main/util/ema.py
'''
import warnings

import torch
from torch.optim import Optimizer


class EMA:
    def __init__(self, opt, ema_decay):
        self.ema_decay = ema_decay
        self.apply_ema = self.ema_decay > 0.
        self.optimizer = opt

    def step(self, *args, **kwargs):
        retval = self.optimizer.step(*args, **kwargs)

        if not self.apply_ema:
            return retval

        ema, params = {}, {}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]
                if 'ema' not in state:
                    state['ema'] = p.data.clone()

                if p.shape not in params:
                    params[p.shape] = {'idx': 0, 'data': []}
                    ema[p.shape] = []

                params[p.shape]['data'].append(p.data)
                ema[p.shape].append(state['ema'])

            for k in params:
                params[k]['data'] = torch.stack(params[k]['data'], dim=0)
                ema[k] = torch.stack(ema[k], dim=0)
                ema[k].mul_(self.ema_decay).add_(params[k]['data'], alpha=1. - self.ema_decay)

            for p in group['params']:
                if p.grad is None:
                    continue
                idx = params[p.shape]['idx']
                self.optimizer.state[p]['ema'] = ema[p.shape][idx]
                params[p.shape]['idx'] += 1

        return retval

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def swap_parameters_with_ema(self, store_params_in_ema):
        if not self.apply_ema:
            warnings.warn('swap_parameters_with_ema was called when there is no EMA weights.')
            return

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                ema = self.optimizer.state[p]['ema']
                if store_params_in_ema:
                    tmp = p.data.detach().clone()
                    p.data.copy_(ema)
                    self.optimizer.state[p]['ema'] = tmp
                else:
                    p.data.copy_(ema)
