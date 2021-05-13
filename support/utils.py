# Sample-based Monte Carlo Denoising using a Kernel-Splatting Network
# Michaël Gharbi Tzu-Mao Li Miika Aittala Jaakko Lehtinen Frédo Durand
# Siggraph 2019
#
# Copyright (c) 2019 Michaël Gharbi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import argparse
import torch as th


def crop_like(src, tgt):
    src_sz = np.array(src.shape)
    tgt_sz = np.array(tgt.shape)

    # Assumes the spatial dimensions are the last two
    # delta = (src_sz[2:4]-tgt_sz[2:4])
    delta = (src_sz[-2:]-tgt_sz[-2:])
    crop = np.maximum(delta // 2, 0)  # no negative crop
    crop2 = delta - crop

    if (crop > 0).any() or (crop2 > 0).any():
        # NOTE: convert to ints to enable static slicing in ONNX conversion
        src_sz = [int(x) for x in src_sz]
        crop = [int(x) for x in crop]
        crop2 = [int(x) for x in crop2]
        return src[..., crop[0]:src_sz[-2]-crop2[0],
                   crop[1]:src_sz[-1]-crop2[1]]
    else:
        return src

def ToneMap(c, limit=1.5):
    # c: (W, H, C=3)
    luminance = 0.2126 * c[:,:,0] + 0.7152 * c[:,:,1] + 0.0722 * c[:,:,2]
    col = c.copy()
    col[:,:,0] /=  (1.0 + luminance / limit)
    col[:,:,1] /=  (1.0 + luminance / limit)
    col[:,:,2] /=  (1.0 + luminance / limit)
    return col

def LinearToSrgb(c):
    # c: (W, H, C=3)
    kInvGamma = 1.0 / 2.2
    return np.clip(c ** kInvGamma, 0.0, 1.0)

def ToneMapBatch(c):
    # c: (B, C=3, W, H)
    luminance = 0.2126 * c[:,0,:,:] + 0.7152 * c[:,1,:,:] + 0.0722 * c[:,2,:,:]
    col = c.copy()
    col[:,0,:,:] /= (1 + luminance / 1.5)
    col[:,1,:,:] /= (1 + luminance / 1.5)
    col[:,2,:,:] /= (1 + luminance / 1.5)
    col = np.clip(col, 0, None)
    kInvGamma = 1.0 / 2.2
    return np.clip(col ** kInvGamma, 0.0, 1.0)


class BasicArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super(BasicArgumentParser, self).__init__(*args, **kwargs)

        self.add_argument('--sbmc', action='store_true',
                            help='train the Sample-based Kernel Splatting Network (Gharbi et al. 2019).')
        self.add_argument('--p_buf', action='store_true',
                            help='use the multi-bounce path buffers for denoising.')
        self.add_argument('--model_name', type=str, default='tSUNet', 
                            help='name of the model.')
        self.add_argument('--data_dir', type=str, default='./data',
                            help='directory of dataset')
        self.add_argument('--visual', action='store_true',
                            help='use visualizer, otherwise use terminal to monitor the training process.')  
        self.add_argument('-b', '--batch_size', type=int, default=64,
                            help='batch size.')
        self.add_argument('-e', '--num_epoch', type=int, default=100,
                            help='number of epochs.')
        self.add_argument('-v', '--val_epoch', type=int, default=1,
                            help='validate the model every val_epoch epoch.')
        self.add_argument('--vis_iter', type=int, default=4,
                            help='visualize the training dataset every vis_iter iteration.')
        self.add_argument('--start_epoch', type=int, default=0,
                            help='from which epoch to start.')
        self.add_argument('--num_samples', type=int, default=8,
                            help='number of samples to be displayed.')
        self.add_argument('--save', type=str, default='./weights', 
                            help='directory to save the model.')
        self.add_argument('--overfit', action='store_true', 
                            help='launch overfitting test.')
