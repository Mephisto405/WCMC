import os
import sys
import time
import argparse
import matplotlib.pyplot as plt 
from collections import OrderedDict

import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader

import train_kpcn
import train_sbmc
import train_lbmc
from support.utils import crop_like
from support.img_utils import WriteImg
from support.datasets import FullImageDataset
from support.networks import PathNet, weights_init
from support.metrics import RelMSE, RelL1, SSIM, MSE, L1, _tonemap


def tonemap(c, ref=None, kInvGamma=1.0/2.2):
    # c: (W, H, C=3)
    if ref is None:
        ref = c
    luminance = 0.2126 * ref[:,:,0] + 0.7152 * ref[:,:,1] + 0.0722 * ref[:,:,2]
    col = np.copy(c)
    col[:,:,0] /= (1 + luminance / 1.5)
    col[:,:,1] /= (1 + luminance / 1.5)
    col[:,:,2] /= (1 + luminance / 1.5)
    col = np.clip(col, 0, None)
    return np.clip(col ** kInvGamma, 0.0, 1.0)


def load_input(filename, spp, args):    
    if 'KPCN' in args.model_name:
        dataset = FullImageDataset(filename, spp, 'kpcn',
                                   args.use_g_buf, args.use_sbmc_buf, 
                                   args.use_llpm_buf, args.pnet_out_size[0])
    elif 'BMC' in args.model_name:
        dataset = FullImageDataset(filename, spp, 'sbmc',
                                   args.use_g_buf, args.use_sbmc_buf, 
                                   args.use_llpm_buf, 0)
    return dataset


def inference(interface, dataloader, spp, args):
    interface.to_eval_mode()

    H, W = dataloader.dataset.h, dataloader.dataset.w
    PATCH_SIZE = dataloader.dataset.PATCH_SIZE
    out_rad = torch.zeros((3, H, W)).cuda()
    out_path = None

    with torch.no_grad():
        for batch, i_start, j_start, i_end, j_end, i, j in dataloader:
            for k in batch:
                if not batch[k].__class__ == torch.Tensor:
                    continue
                batch[k] = batch[k].cuda(args.device_id)

            start = time.time()
            out, p_buffers = interface.validate_batch(batch)

            pad_h = PATCH_SIZE - out.shape[2]
            pad_w = PATCH_SIZE - out.shape[3]
            if pad_h != 0 and pad_w != 0:
                out = nn.functional.pad(out, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2), 'replicate') # order matters

            if args.use_llpm_buf and (out_path is None):
                if type(p_buffers) == dict:
                    out_path = {}
                    for key in p_buffers:
                        b, s, c, h, w = p_buffers[key].shape
                        out_path[key] = torch.zeros((s, c, H, W)).cuda()
                elif type(p_buffers) == torch.Tensor:
                    b, s, c, h, w = p_buffers.shape
                    out_path = torch.zeros((s, c, H, W)).cuda()
                else:
                    assert False, 'P buffer type not defined.'

            for b in range(out.shape[0]):
                out_rad[:,i_start[b]:i_end[b],j_start[b]:j_end[b]] = out[b,:,i_start[b]-i[b]:i_end[b]-i[b],j_start[b]-j[b]:j_end[b]-j[b]]
                if args.use_llpm_buf:
                    if type(p_buffers) == dict:
                        for key in p_buffers:
                            out_path[key][:,:,i_start[b]:i_end[b],j_start[b]:j_end[b]] = p_buffers[key][b,:,:,i_start[b]-i[b]:i_end[b]-i[b],j_start[b]-j[b]:j_end[b]-j[b]]
                    elif type(p_buffers) == torch.Tensor:
                        out_path[:,:,i_start[b]:i_end[b],j_start[b]:j_end[b]] = p_buffers[b,:,:,i_start[b]-i[b]:i_end[b]-i[b],j_start[b]-j[b]:j_end[b]-j[b]]

    out_rad = out_rad.detach().cpu().numpy().transpose([1, 2, 0])
    if args.use_llpm_buf:
        if type(out_path) == dict:
            for key in out_path:
                out_path[key] = out_path[key].detach().cpu().numpy().transpose([2, 3, 0, 1])
        elif type(out_path) == torch.Tensor:
            out_path = out_path.detach().cpu().numpy().transpose([2, 3, 0, 1])

    return out_rad, out_path


def denoise(args, input_dir, output_dir="../test_suite_2", scenes=None, spps=[8], save_figures=False, rhf=False, quantize=False):
    assert os.path.isdir(input_dir), input_dir
    assert 'KPCN' in args.model_name or 'BMC' in args.model_name, args.model_name

    if scenes is None:
        scenes = []
        for fn in os.listdir(input_dir.replace(os.sep + 'input', os.sep + 'gt')):
            if fn.endswith(".npy"):
                scenes.append(fn)
    num_metrics = 5 * 4 # (RelL2, RelL1, DSSIM, L1, MSE) * (linear, tmap w/o gamma, tmap gamma=2.2, tmap gamma=adaptive)
    results = [[0 for i in range(len(scenes))] for j in range(num_metrics * len(spps))]
    results_input = [[0 for i in range(len(scenes))] for j in range(num_metrics * len(spps))]

    if args.model_name.endswith('.pth'):
        p_model = os.path.join(args.save, args.model_name)
    else:
        p_model = os.path.join(args.save, args.model_name + '.pth')
    ck = torch.load(p_model)

    print(scenes)
    for scene in scenes:
        if not scene.endswith(".npy"):
            scene = scene + '.npy'
        filename = os.path.join(input_dir, scene).replace(os.sep + 'input', os.sep + 'gt')
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
    
    for i, scene in enumerate(scenes):
        if scene.endswith(".npy"):
            scene = scene[:-4]
        print("Scene file: ", scene)
        os.makedirs(os.path.join(output_dir, scene), exist_ok=True)

        for j, spp in enumerate(spps):
            print("Samples per pixel:", spp)
            """
            Denoising
            """
            # Dateload
            filename = os.path.join(input_dir, scene + ".npy")

            dataset = load_input(filename, spp, args)
            
            MSPP = 32# if args.pnet_out_size[0] < 12 else 8
            if spp <= MSPP:
                dataloader = DataLoader(
                    dataset,
                    batch_size=8,
                    num_workers=1
                )
            elif spp <= 64:
                dataloader = DataLoader(
                    dataset,
                    batch_size=4,
                    num_workers=1
                )
            else:
                raise RuntimeError("Try higher spp after investigating your RAM and \
                    GRAM capacity.")
            
            if i == 0 and j == 0:
                datasets = {'train': dataset} # dirty code for now
                if 'SBMC' in args.model_name:
                    interfaces, _ = train_sbmc.init_model(datasets, args)
                elif 'LBMC' in args.model_name:
                    interfaces, _ = train_lbmc.init_model(datasets, args)
                elif 'KPCN' in args.model_name:
                    interfaces, _ = train_kpcn.init_model(datasets, args)
            '''
            if tensorrt:
                engines, contexts = export_and_load_onnx_model(interfaces[0], p_model, dataloader)
                return
            '''
            out_rad, out_path = inference(interfaces[0], dataloader, spp, args)
            
            """
            Post processing
            """
            tgt = dataset.full_tgt
            ipt = dataset.full_ipt
            if out_path is not None:
                if rhf:
                    print('Saving P-buffer as numpy file for RHF-like visualization...')
                    if 'BMC' in args.model_name:
                        print('Shape: ', out_path.shape)
                        np.save(os.path.join(output_dir, 'p_buffer_%s_%s.npy'%(scene, args.model_name)), out_path)
                    elif 'KPCN' in args.model_name:
                        print('Shape: ', out_path['diffuse'].shape)
                        np.save(os.path.join(output_dir, 'p_buffer_%s_%s.npy'%(scene, args.model_name)), out_path['diffuse'])
                    print('Saved.')
                    return
                
                if type(out_path) == dict:
                
                    for key in out_path:
                        out_path[key] = np.clip(np.mean(out_path[key], 2), 0.0, 1.0)
                        assert len(out_path[key].shape) == 3, out_path[key].shape
                        if out_path[key].shape[2] >= 3:
                            out_path[key] = out_path[key][...,:3]
                        else:
                            tmp = np.mean(out_path[key], 2, keepdims=True)
                            out_path[key] = np.concatenate((tmp,) * 3, axis=2)
                        assert out_path[key].shape[2] == 3, out_path[key].shape
                elif type(out_path) == torch.Tensor:
                    out_path = np.clip(np.mean(out_path, 2), 0.0, 1.0)
                    assert len(out_path.shape) == 3, out_path.shape
                    if out_path.shape[2] >= 3:
                        out_path = out_path[...,:3]
                    else:
                        tmp = np.mean(out_path, 2, keepdims=True)
                        out_path = np.concatenate((tmp,) * 3, axis=2)
                    assert out_path.shape[2] == 3, out_path.shape

            # Crop
            valid_size = 72
            crop = (128 - valid_size) // 2
            out_rad = out_rad[crop:-crop, crop:-crop, ...]
            if out_path is not None:
                if type(out_path) == dict:
                    for key in out_path:
                        out_path[key] = out_path[key][crop:-crop, crop:-crop, ...]
                elif type(out_path) == torch.Tensor:
                    out_path = out_path[crop:-crop, crop:-crop, ...]
            tgt = tgt[crop:-crop, crop:-crop, ...]
            ipt = ipt[crop:-crop, crop:-crop, ...]

            # Process the background and emittors which do not require to be denoised
            has_hit = dataset.has_hit[crop:-crop, crop:-crop, ...]
            out_rad = np.where(has_hit == 0, ipt, out_rad)

            """
            Statistics
            """
            err = RelMSE(out_rad, tgt, reduce=False)
            err = err.reshape(out_rad.shape[0], out_rad.shape[1], 3)
            
            # (RelL2, RelL1, DSSIM, L1, MSE) * (linear, tmap w/o gamma, tmap gamma=2.2, tmap gamma=adaptive)
            def linear(x):
                return x

            def tonemap28(x):
                return tonemap(x, kInvGamma = 1/2.8)

            metrics = [RelMSE, RelL1, SSIM, L1, MSE]
            tmaps = [linear, _tonemap, tonemap, tonemap28]
            
            print(RelMSE(tonemap(out_rad), tonemap(tgt)))
            print(RelMSE(tonemap(ipt), tonemap(tgt)))

            for t, tmap in enumerate(tmaps):
                for k, metric in enumerate(metrics):
                    results[(len(metrics) * t + k) * len(spps) + j][i] = metric(tmap(out_rad), tmap(tgt))
                    results_input[(len(metrics) * t + k) * len(spps) + j][i] = metric(tmap(ipt), tmap(tgt))

            """
            Save
            """
            if save_figures:
                t_tgt = tmaps[-1](tgt)
                t_ipt = tmaps[-1](ipt)
                t_out = tmaps[-1](out_rad)
                t_err = np.mean(np.clip(err**0.45, 0.0, 1.0), 2)

                plt.imsave(os.path.join(output_dir, scene, 'target.png'), t_tgt)
                #WriteImg(os.path.join(output_dir, scene, 'target.pfm'), tgt) # HDR image
                plt.imsave(os.path.join(output_dir, scene, 'input_{}.png'.format(spp)), t_ipt)
                #WriteImg(os.path.join(output_dir, scene, 'input_{}.pfm'.format(spp)), ipt)
                plt.imsave(os.path.join(output_dir, scene, 'output_{}_{}.png'.format(spp, args.model_name)), t_out)
                #WriteImg(os.path.join(output_dir, scene, 'output_{}_{}.pfm'.format(spp, args.model_name)), out_rad)
                plt.imsave(os.path.join(output_dir, scene, 'errmap_rmse_{}_{}.png'.format(spp, args.model_name)), t_err, cmap=plt.get_cmap('magma'))
                #WriteImg(os.path.join(output_dir, scene, 'errmap_{}_{}.pfm'.format(spp, args.model_name)), err.mean(2))

    np.savetxt(os.path.join(output_dir, 'results_{}_{}.csv'.format(args.model_name, spps[-1])), results, delimiter=',')
    np.savetxt(os.path.join(output_dir, 'results_input_%d.csv'%(spps[-1])), results_input, delimiter=',')


if __name__ == "__main__":
    class Args(): # just for compatibility with argparse-related functions
        save = '/root/LPM/weights/'
        model_name = 'SBMC_v2.0'
        single_gpu = True
        use_g_buf, use_sbmc_buf, use_llpm_buf = True, True, True
        
        lr_pnet = [1e-4]
        lr_ckpt = True
        pnet_out_size = [3]
        w_manif = [0.1]
        manif_learn = False
        manif_loss = 'FMSE'
        train_branches = False

        disentangle = 'm11r11'

        kpcn_ref = False

        start_epoch = 0
        single_gpu = True
        device_id = 0
        lr_dncnn = 1e-4

        visual = False
        start_epoch = 10
        best_err = 1e4

        kpcn_ref = False
        kpcn_pre = False
        not_save = False
    
    args = Args()

    input_dir = '/mnt/ssd2/iycho/KPCN/test2/input/'
    scenes = ['bathroom_v3', 'bathroom-3_v2', 'car', 'car_v2', 'car_v3', 'chair-room', 'chair-room_v2', 'hookah_v3', 'kitchen-2', 'kitchen-2_v2', 'library-office', 'sitting-room-2']
    spps = [8]
    
    """ Test cases
    # LBMC
    print('LBMC_Path_P3')
    args.model_name = 'LBMC_Path_P3'
    args.pnet_out_size = [3]
    args.disentangle = 'm11r11'
    args.use_g_buf, args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = True, False, True, False
    denoise(args, input_dir, spps=[2,4,8,16,32,64], scenes=scenes, save_figures=True)
    
    print('LBMC_Manifold_P6')
    args.model_name = 'LBMC_Manifold_P6'
    args.pnet_out_size = [6]
    args.disentangle = 'm11r11'
    args.use_g_buf, args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = True, False, True, True
    denoise(args, input_dir, spps=[2,4,8,16,32,64], scenes=scenes, save_figures=True)
    
    print('LBMC_vanilla')
    args.model_name = 'LBMC_vanilla'
    args.pnet_out_size = [0]
    args.disentangle = 'm11r11'
    args.use_g_buf, args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = True, False, False, False
    denoise(args, input_dir, spps=[2,4,8,16,32,64], scenes=scenes, save_figures=True)
    
    # KPCN
    print('KPCN_vanilla')
    args.model_name = 'KPCN_vanilla'
    args.pnet_out_size = [0]
    args.use_llpm_buf, args.manif_learn = False, False
    denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True)
    
    print('KPCN_path')
    args.model_name = 'KPCN_path'
    args.pnet_out_size = [3]
    args.disentangle = 'm11r11'
    args.use_llpm_buf, args.manif_learn = True, False
    denoise(args, input_dir, spps=spps, scenes=scenes, rhf=True)
     
    # SBMC
    print('SBMC_vanilla')
    args.model_name = 'SBMC_vanilla'
    args.pnet_out_size = [0]
    args.disentangle = 'm11r11'
    args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = True, False, False
    denoise(args, input_dir, spps=spps, scenes=scenes, save_figures=True)

    print('SBMC_path')
    args.model_name = 'SBMC_path'
    args.pnet_out_size = [3]
    args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = False, True, False
    denoise(args, input_dir, spps=spps, scenes=scenes, rhf=True)
    
    print('SBMC_Manifold_Naive')
    args.model_name = 'SBMC_Manifold_Naive'
    args.pnet_out_size = [3]
    args.use_sbmc_buf, args.use_llpm_buf, args.manif_learn = False, True, False
    denoise(args, input_dir, spps=spps, scenes=scenes)
    """
