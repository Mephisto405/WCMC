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
from support.datasets import MSDenoiseDataset
from support.utils import BasicArgumentParser
from support.losses import RelativeMSE, FeatureMSE, GlobalRelativeSimilarityLoss
from support.interfaces import KPCNInterface, KPCNRefInterface, KPCNPreInterface

# Gharbi et al. dependency
sys.path.insert(1, configs.PATH_SBMC)
try:
    from sbmc import KPCN
except ImportError as error:
    print('Put appropriate paths in the configs.py file.')
    raise
from ttools.modules.image_operators import crop_like


def train_epoch_kpcn(epoch, interfaces, dataloaders, params, args):
    assert 'train' in dataloaders, "argument `dataloaders` dictionary should contain `'train'` key."
    assert 'data_device' in params, "argument `params` dictionary should contain `'data_device'` key."
    print('[][] Epoch %d' % (epoch))

    for itf in interfaces:
        itf.to_train_mode()

    for batch in tqdm(dataloaders['train'], leave=False, ncols=70):
        # Transfer data from the cpu to gpu memory
        for k in batch:
            if not batch[k].__class__ == torch.Tensor:
                continue
            batch[k] = batch[k].cuda(params['data_device'])

        # Main
        for itf in interfaces:
            itf.preprocess(batch)
            itf.train_batch(batch)
        
    if not args.visual:
        for itf in interfaces:
            itf.get_epoch_summary(mode='train', norm=len(dataloaders['train']))


def validate_kpcn(epoch, interfaces, dataloaders, params, args):
    assert 'val' in dataloaders, "argument `dataloaders` dictionary should contain `'train'` key."
    assert 'data_device' in params, "argument `params` dictionary should contain `'data_device'` key."
    print('[][] Validation (epoch %d)' % (epoch))

    for itf in interfaces:
        itf.to_eval_mode()

    cnt = 0
    summaries = []
    with torch.no_grad():
        for batch in tqdm(dataloaders['val'], leave=False, ncols=70):
            # Transfer data from the cpu to gpu memory
            for k in batch:
                if not batch[k].__class__ == torch.Tensor:
                    continue
                batch[k] = batch[k].cuda(params['data_device'])

            # Main
            for itf in interfaces:
                itf.validate_batch(batch)

    for itr in interfaces:
        summaries.append(itf.get_epoch_summary(mode='eval', norm=len(dataloaders['val'])))

    return summaries


def train(interfaces, dataloaders, params, args):
    print('[] Experiment: `{}`'.format(args.desc))
    print('[] # of interfaces : %d'%(len(interfaces)))
    print('[] Model training start...')
    
    # Start training
    for epoch in range(args.start_epoch, args.num_epoch):
        if len(interfaces) == 1:
            save_fn = args.model_name + '.pth'
        else:
            raise NotImplementedError('Multiple interfaces')

        start_time = time.time()
        train_epoch_kpcn(epoch, interfaces, dataloaders, params, args)
        print('[][] Elapsed time: %d'%(time.time() - start_time))

        for i, itf in enumerate(interfaces):
            tmp_params = params.copy()
            tmp_params['vis'] = None

            state_dict = {
                'description': args.desc, #
                'start_epoch': epoch + 1,
                'model': str(itf.models['dncnn']),
                'params': tmp_params,
                'optims': itf.optims,
                'args': args,
                'best_err': itf.best_err
            }

            for model_name in itf.models:
                state_dict['state_dict_' + model_name] = itf.models[model_name].state_dict()

            if not args.not_save:
                torch.save(state_dict, os.path.join(args.save, 'latest_' + save_fn))

        # Validate models
        if (epoch % args.val_epoch == args.val_epoch - 1):
            print('[][] Validation')
            summaries = validate_kpcn(epoch, interfaces, dataloaders, params, args)

            for i, itf in enumerate(interfaces):
                if summaries[i] < itf.best_err:
                    itf.best_err = summaries[i]

                    tmp_params = params.copy()
                    tmp_params['vis'] = None

                    state_dict = {
                        'description': args.desc, #
                        'start_epoch': epoch + 1,
                        'model': str(itf.models['dncnn']),
                        'params': tmp_params,
                        'optims': itf.optims,
                        'args': args,
                        'best_err': itf.best_err
                    }

                    for model_name in itf.models:
                        state_dict['state_dict_' + model_name] = itf.models[model_name].state_dict()

                    if not args.not_save:
                        torch.save(state_dict, os.path.join(args.save, save_fn))
                        print('[][] Model %s saved at epoch %d.'%(save_fn, epoch))

                print('[][] Model {} RelMSE: {:.3f}e-3 \t Best RelMSE: {:.3f}e-3'.format(save_fn, summaries[i]*1000, itf.best_err*1000))
        
        # Update schedulers
        for key in params:
            if 'sched_' in key:
                params[key].step()
    print('[] Training complete!')


"""
Main Utils
"""
def init_data(args):
    # Initialize datasets
    datasets = {}
    datasets['train'] = MSDenoiseDataset(args.data_dir, 8, 'kpcn', 'train', args.batch_size, 'random',
        use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3)
    datasets['val'] = MSDenoiseDataset(args.data_dir, 8, 'kpcn', 'val', BS_VAL, 'grid',
        use_g_buf=True, use_sbmc_buf=False, use_llpm_buf=args.use_llpm_buf, pnet_out_size=3)
    
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
        if args.train_branches:
            print('Train diffuse and specular branches indenpendently.')
        else:
            print('Post-train two branches of KPCN.')
            
        if args.use_llpm_buf:
            if args.disentangle in ['m10r01', 'm11r01']:
                n_in = dataset['train'].dncnn_in_size - dataset['train'].pnet_out_size + pnet_out_size // 2
            else:
                n_in = dataset['train'].dncnn_in_size - dataset['train'].pnet_out_size + pnet_out_size
            models['dncnn'] = KPCN(n_in)
            print('Initialize KPCN for path descriptors (# of input channels: %d).'%(n_in))

            n_in = dataset['train'].pnet_in_size
            n_out = pnet_out_size
            if args.train_branches:
                print('Train PathNet backbones indenpendently for diffuse and specular branches (# of input channels: %d, # of output channels: %d).'%(n_in, n_out))
            else:
                print('Post-train PathNet backbones.')
            models['backbone_diffuse'] = PathNet(ic=n_in, outc=n_out)
            models['backbone_specular'] = PathNet(ic=n_in, outc=n_out)
        else:
            if args.kpcn_ref:
                n_in = dataset['train'].dncnn_in_size + 3
            else:
                n_in = dataset['train'].dncnn_in_size
            models['dncnn'] = KPCN(n_in)
            print('Initialize KPCN for vanilla buffers (# of input channels: %d).'%(n_in))
        
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
            'l_diffuse': nn.L1Loss(),
            'l_specular': nn.L1Loss(),
            'l_recon': nn.L1Loss(),
            'l_test': RelativeMSE()
        }
        if args.manif_learn:
            if args.manif_loss == 'FMSE':
                loss_funcs['l_manif'] = FeatureMSE(non_local = not args.local)
                print('Manifold loss: FeatureMSE')
            elif args.manif_loss == 'GRS':
                loss_funcs['l_manif'] = GlobalRelativeSimilarityLoss()
                print('Manifold loss: Global Relative Similarity') 
        else:
            print('Manifold loss: None (i.e., ablation study)')

        # Initialize a training interface (NOTE: modified for each model)
        if args.kpcn_ref:
            itf = KPCNRefInterface(models, optims, loss_funcs, args, train_branches=args.train_branches)
        elif args.kpcn_pre:
            itf = KPCNPreInterface(models, optims, loss_funcs, args, manif_learn=args.manif_learn, train_branches=args.train_branches)
        else:
            itf = KPCNInterface(models, optims, loss_funcs, args, visual=args.visual, use_llpm_buf=args.use_llpm_buf, manif_learn=args.manif_learn, w_manif=w_manif, train_branches=args.train_branches, disentanglement_option=args.disentangle)
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


if __name__ == "__main__":
    """ NOTE: Example Training Scripts """
    """ KPCN Vanilla
        Train two branches (i.e., diffuse and specular):
            python cp_train_kpcn.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/ssd3/iycho/KPCN --model_name KPCN_vanilla --desc "KPCN vanilla" --num_epoch 8 --lr_dncnn 1e-4 --train_branches

        Post-joint training ('fine-tuning' according to the original authors): 
            python cp_train_kpcn.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/ssd3/iycho/KPCN --model_name KPCN_vanilla --desc "KPCN vanilla" --num_epoch 10 --lr_dncnn 1e-6 --start_epoch ?
    """

    """ KPCN Manifold
        Train two branches (i.e., diffuse and specular):
            python cp_train_kpcn.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/ssd3/iycho/KPCN --model_name KPCN_manifold_FMSE --desc "KPCN manifold FMSE" --num_epoch 8 --manif_loss FMSE --lr_dncnn 1e-4 --lr_pnet 1e-4 --use_llpm_buf --manif_learn --w_manif 0.1 --train_branches

        Post-joint training:
            python cp_train_kpcn.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/ssd3/iycho/KPCN --model_name KPCN_manifold_FMSE --desc "KPCN manifold FMSE" --num_epoch 10 --manif_loss FMSE --lr_dncnn 1e-6 --lr_pnet 1e-6 --use_llpm_buf --manif_learn --w_manif 0.1 --start_epoch <best pre-training epoch>
    """

    """ KPCN Path (ablation study)
        Train two branches (i.e., diffuse and specular):
            python cp_train_kpcn.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/ssd3/iycho/KPCN --model_name KPCN_path --desc "KPCN ablation study" --num_epoch 8 --lr_dncnn 1e-4 --lr_pnet 1e-4 --use_llpm_buf --train_branches

        Post-joint training:
            python cp_train_kpcn.py --single_gpu --batch_size 8 --val_epoch 1 --data_dir /mnt/ssd3/iycho/KPCN --model_name KPCN_path --desc "KPCN ablation study" --num_epoch 10 --lr_dncnn 1e-6 --lr_pnet 1e-6 --use_llpm_buf --start_epoch <best pre-training epoch>
    """

    BS_VAL = 4 # validation set batch size

    parser = BasicArgumentParser()
    parser.add_argument('--desc', type=str, required=True, 
                        help='short description of the current experiment.')
    parser.add_argument('--lr_dncnn', type=float, default=1e-4, 
                        help='learning rate of PathNet.')
    parser.add_argument('--lr_pnet', type=float, nargs='+', default=[0.0001], 
                        help='learning rate of PathNet.')
    parser.add_argument('--lr_ckpt', action='store_true',
                        help='')
    parser.add_argument('--best_err', type=float, required=False)
    parser.add_argument('--pnet_out_size', type=int, nargs='+', default=[3], 
                        help='# of channels of outputs of PathNet.')
    parser.add_argument('--manif_loss', type=str, required=False,
                        help='`FMSE` or `GRS`')
    
    parser.add_argument('--train_branches', action='store_true',
                        help='train the diffuse and specular branches independently.')
    parser.add_argument('--use_llpm_buf', action='store_true',
                        help='use the llpm-specific buffer.')
    parser.add_argument('--manif_learn', action='store_true',
                        help='use the manifold learning loss.')
    parser.add_argument('--w_manif', type=float, nargs='+', default=[0.1], 
                        help='ratio of the manifold learning loss to \
                        the reconstruction loss.')
    
    parser.add_argument('--disentangle', type=str, default='m11r11',
                        help='`m11r11`, `m10r01`, `m10r11`, or `m11r01`')

    parser.add_argument('--single_gpu', action='store_true',
                        help='use only one GPU.')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id')

    parser.add_argument('--kpcn_ref', action='store_true',
                        help='train KPCN-Ref model.')
    parser.add_argument('--kpcn_pre', action='store_true',
                        help='train KPCN-Pre model.')
    parser.add_argument('--not_save', action='store_true',
                        help='do not save checkpoint (debugging purpose).')
    parser.add_argument('--local', action='store_true')

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
