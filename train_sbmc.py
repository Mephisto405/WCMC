# Python
import os
import sys
import time
import visdom
import random
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

# NumPy and PyTorch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Cho et al. dependency
import configs
from support.networks import PathNet
from support.interfaces import SBMCInterface
from support.datasets import MSDenoiseDataset
from train_kpcn import validate_kpcn, train, train_epoch_kpcn
from support.losses import TonemappedRelativeMSE, RelativeMSE, FeatureMSE, GlobalRelativeSimilarityLoss

# Gharbi et al. dependency
sys.path.insert(1, configs.PATH_SBMC)
try:
    from sbmc import Multisteps
except ImportError as error:
    print('Put appropriate paths in the configs.py file.')
    raise
from support.utils import BasicArgumentParser
from ttools.modules.image_operators import crop_like


def init_data(args):
    # Initialize datasets
    datasets = {}
    datasets['train'] = MSDenoiseDataset(args.data_dir, 8, 'sbmc', 'train', args.batch_size, 'random',
        use_g_buf=True, use_sbmc_buf=args.use_sbmc_buf, use_llpm_buf=args.use_llpm_buf, pnet_out_size=0)
    datasets['val'] = MSDenoiseDataset(args.data_dir, 8, 'sbmc', 'val', BS_VAL, 'grid',
        use_g_buf=True, use_sbmc_buf=args.use_sbmc_buf, use_llpm_buf=args.use_llpm_buf, pnet_out_size=0)
    
    # Initialize dataloaders
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        datasets['train'], 
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=False,
    )
    dataloaders['val'] = DataLoader(
        datasets['val'],
        batch_size=BS_VAL,
        num_workers=1,
        pin_memory=False
    )
    return datasets, dataloaders


def init_model(dataset, args):
    interfaces = []

    lr_pnets = args.lr_pnet
    pnet_out_sizes = args.pnet_out_size
    w_manifs = args.w_manif

    tmp = [lr_pnets, pnet_out_sizes, w_manifs]
    for lr_pnet, pnet_out_size, w_manif in list(itertools.product(*tmp)):
        # Initialize models (NOTE: modified for each model)
        models = {}
        print('Train a SBMC network.')
        if args.use_llpm_buf and not args.use_sbmc_buf:
            if args.disentangle in ['m10r01', 'm11r01']:
                n_in = dataset['train'].dncnn_in_size + pnet_out_size // 2
            else:
                n_in = dataset['train'].dncnn_in_size + pnet_out_size
            models['dncnn'] = Multisteps(n_in)
            print('Initialize the SBMC for path descriptors (# of input channels: %d).'%(n_in))

            n_in = dataset['train'].pnet_in_size
            n_out = pnet_out_size
            print('Train a LLPM feature extractor. (# of input channels: %d, # of output channels: %d).'%(n_in, n_out))
            models['backbone'] = PathNet(ic=n_in, outc=n_out)
        elif args.use_sbmc_buf:
            n_in = dataset['train'].dncnn_in_size
            models['dncnn'] = Multisteps(n_in)
            print('Initialize the SBMC for vanilla buffers (# of input channels: %d).'%(n_in))
        elif args.use_g_buf and not args.manif_learn:
            n_in = dataset['train'].dncnn_in_size
            models['dncnn'] = Multisteps(n_in)
            print('Initialize the SBMC only for G-buffer (# of input channels: %d).'%(n_in))
        else:
            raise RuntimeError('No such feature combination defined.')

        # Load pretrained weights
        if len(list(itertools.product(*tmp))) == 1:
            model_fn = os.path.join(args.save, args.model_name + '.pth')
        else:
            model_fn = os.path.join(args.save, '%s_lp%f_pos%d_wgt%f.pth'%(args.model_name, lr_pnet, pnet_out_size, w_manif))
        assert args.start_epoch != 0 or not os.path.isfile(model_fn), 'Model %s already exists.'%(model_fn)
        is_pretrained = (args.start_epoch != 0) and os.path.isfile(model_fn)

        if is_pretrained:
            ck = torch.load(model_fn)
            for model_name in models:
                try:
                    models[model_name].load_state_dict(ck['state_dict_' + model_name])
                except RuntimeError:
                    new_state_dict = OrderedDict()
                    for k, v in ck['state_dict_' + model_name].items():
                        name = k[7:]
                        new_state_dict[name] = v
                    models[model_name].load_state_dict(new_state_dict)
            print('Pretraining weights are loaded.')
        else:
            print('Train models from scratch.')

        # Use GPU parallelism if needed
        if args.single_gpu:
            print('Data Sequential')
            for model_name in models:
                models[model_name] = models[model_name].cuda(args.device_id)
        else:
            print('Data Parallel')
            if torch.cuda.device_count() == 1:
                print('Single CUDA machine detected')
                for model_name in models:
                    models[model_name] = models[model_name].cuda()
            elif torch.cuda.device_count() > 1:
                print('%d CUDA machines detected' % (torch.cuda.device_count()))
                for model_name in models:
                    models[model_name] = nn.DataParallel(models[model_name], output_device=1).cuda()
            else:
                assert False, 'No detected GPU device.'
        
        # Initialize optimizers
        optims = {}
        for model_name in models:
            lr = args.lr_dncnn if 'dncnn' == model_name else lr_pnet
            optims['optim_' + model_name] = optim.Adam(models[model_name].parameters(), lr=lr)
            
            if not is_pretrained:
                continue

            if 'optims' in ck:
                state = ck['optims']['optim_' + model_name].state_dict()
            elif 'optim_' + model_name in ck['params']:
                state = ck['params']['optim_' + model_name].state_dict()
            else:
                print('No state for the optimizer for %s, use the initial optimizer and learning rate.'%(model_name))
                continue

            if not args.lr_ckpt:
                print('Set the new learning rate %.3e for %s.'%(lr, model_name))
                state['param_groups'][0]['lr'] = lr
            else:
                print('Use the checkpoint (%s) learning rate for %s.'%(model_fn, model_name))

            optims['optim_' + model_name].load_state_dict(state)
        
        # Initialize losses (NOTE: modified for each model)
        loss_funcs = {
            'l_recon': TonemappedRelativeMSE(),
            'l_test': RelativeMSE()
        }
        if args.manif_learn:
            if args.manif_loss == 'FMSE':
                loss_funcs['l_manif'] = FeatureMSE()
                print('Manifold loss: FeatureMSE')
            elif args.manif_loss == 'GRS':
                loss_funcs['l_manif'] = GlobalRelativeSimilarityLoss()
                print('Manifold loss: Global Relative Similarity')
        else:
            print('Manifold loss: None (i.e., ablation study)')
        
        # Initialize a training interface (NOTE: modified for each model)
        itf = SBMCInterface(models, optims, loss_funcs, args, args.visual, args.use_llpm_buf, args.manif_learn, w_manif, args.use_sbmc_buf, args.disentangle)
        if is_pretrained:
            print('Use the checkpoint best error %.3e'%(args.best_err))
            itf.best_err = args.best_err
        interfaces.append(itf)
    
    # Initialize a visdom visualizer object
    params = {
        'plots': {},
        'data_device': 1 if torch.cuda.device_count() > 1 and not args.single_gpu else args.device_id,
    }
    if args.visual:
        params['vis'] = visdom.Visdom(server='http://localhost')
    else:
        print('No visual.')
    
    # Make the save directory if needed
    if not os.path.isdir(args.save):
        os.mkdir(args.save)

    return interfaces, params


def main(args):
    # Set random seeds
    random.seed("Inyoung Cho, Yuchi Huo, Sungeui Yoon @ KAIST")
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True  #torch.backends.cudnn.deterministic = True

    # Get ready
    dataset, dataloaders = init_data(args)
    interfaces, params = init_model(dataset, args)
    train(interfaces, dataloaders, params, args)


if __name__ == '__main__':
    """ NOTE: Example Training Scripts """
    """ SBMC Vanilla
        Train a SBMC model from scratch: 
            python train_sbmc.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/ssd3/iycho/KPCN --model_name SBMC_vanilla --desc "SBMC_vanilla" --num_epoch 6 --lr_dncnn 5e-4 --use_sbmc_buf
    """
    
    """ SBMC Manifold
        Train a SBMC model from scratch: 
            python train_sbmc.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/ssd3/iycho/KPCN --model_name SBMC_manifold_FMSE_01_P12_FD1 --desc "SBMC_manifold_FMSE_01_P12_FD1" --num_epoch 6 --lr_dncnn 5e-4 --lr_pnet 1e-4 --manif_loss FMSE --use_llpm_buf --manif_learn --w_manif 0.1 --pnet_out_size 12 --disentangle m10r01
    """

    """ SBMC G-buffer
        Train a SBMC model from scratch: 
            python train_sbmc.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/hdd1/iycho/SBMC --model_name SBMC_G --desc "SBMC_G" --num_epoch 6 --lr_dncnn 5e-4
    """

    BS_VAL = 4 # validation set batch size

    parser = BasicArgumentParser()
    # Basic
    parser.add_argument('--desc', type=str, required=True, 
                        help='short description of the current experiment.')
    parser.add_argument('--single_gpu', action='store_true',
                        help='use only one GPU.')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id')
    parser.add_argument('--lr_ckpt', action='store_true',
                        help='')
    parser.add_argument('--best_err', type=float, required=False)
    parser.add_argument('--use_g_buf', action='store_false')

    # Baseline
    parser.add_argument('--lr_dncnn', type=float, default=1e-4, 
                        help='learning rate of PathNet.')
    parser.add_argument('--use_sbmc_buf', action='store_true',
                        help='use the sbmc-specific buffer.')

    # Manifold module
    parser.add_argument('--use_llpm_buf', action='store_true',
                        help='use the llpm-specific buffer.')
    parser.add_argument('--manif_learn', action='store_true',
                        help='use the manifold learning loss.')
    parser.add_argument('--pnet_out_size', type=int, nargs='+', default=[3], 
                        help='# of channels of outputs of PathNet.')
    parser.add_argument('--lr_pnet', type=float, nargs='+', default=[0.0001], 
                        help='learning rate of PathNet.')
    parser.add_argument('--manif_loss', type=str, required=False,
                        help='`FMSE` or `GRS`')
    parser.add_argument('--w_manif', type=float, nargs='+', default=[0.1], 
                        help='ratio of the manifold learning loss to \
                        the reconstruction loss.')
    parser.add_argument('--disentangle', type=str, default='m11r11',
                        help='`m11r11`, `m10r01`, `m10r11`, or `m11r01`')
    parser.add_argument('--not_save', action='store_true',
                        help='do not save checkpoint (debugging purpose).')

    args = parser.parse_args()
    
    if args.manif_learn and not args.use_llpm_buf:
        raise RuntimeError('The manifold learning module requires a llpm-specific buffer.')
    if args.manif_learn and not args.manif_loss:
        raise RuntimeError('The manifold learning module requires a manifold loss.')
    if not args.manif_learn and args.manif_loss:
        raise RuntimeError('A manifold loss is not necessary when the manifold learning module is opted out.')
    if args.manif_learn and args.manif_loss not in ['GRS', 'FMSE']:
        raise RuntimeError('Argument `manif_loss` should be either `FMSE` or `GRS`')
    if args.disentangle not in ['m11r11', 'm10r01', 'm10r11', 'm11r01']:
        raise RuntimeError('Argument `disentangle` should be either `m11r11`, `m10r01`, `m10r11`, or `m11r01`')
    for s in args.pnet_out_size:
        if args.disentangle != 'm11r11' and s % 2 != 0:
            raise RuntimeError('Argument `pnet_out_size` should be a list of even numbers')
    
    main(args)
