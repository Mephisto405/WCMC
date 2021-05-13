"""Encapsulation of model training and testing for learning light path manifold.
"""

# Python
import sys
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
# NumPy and PyTorch
import torch
import numpy as np
import torch.nn as nn
# Gharbi et al. dependency
#
# Cho et al. dependency
from support.utils import crop_like


class BaseInterface(metaclass=ABCMeta):

    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1):
        self.models = models
        self.optims = optims
        self.loss_funcs = loss_funcs
        self.args = args
        self.visual = visual
        self.use_llpm_buf = use_llpm_buf
        self.manif_learn = manif_learn
        self.w_manif = w_manif

        self.iters = 0
        self.m_losses = {}
        self.best_err = 1e10
        self.fixed_batch = None

    @abstractmethod
    def to_train_mode(self):
        pass

    @abstractmethod
    def preprocess(self, batch=None):
        pass

    @abstractmethod
    def train_batch(self, batch):
        pass

    @abstractmethod
    def _manifold_forward(self, batch):
        return {}

    @abstractmethod
    def _regress_forward(self, batch):
        return {}

    @abstractmethod
    def _backward(self, batch, out, p_buffers):
        return {}
    
    @abstractmethod
    def _logging(self, loss_dict):
        pass

    @abstractmethod
    def _optimization(self):
        pass

    @abstractmethod
    def to_eval_mode(self):
        pass

    @abstractmethod
    def validate_batch(self, batch):
        pass

    @abstractmethod
    def get_epoch_summary(self, mode, norm):
        return 0.0


class KPCNInterface(BaseInterface):

    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1, train_branches=True, disentanglement_option="m11r11"):
        if manif_learn:
            assert 'backbone_diffuse' in models, "argument `models` dictionary should contain `'backbone_diffuse'` key."
            assert 'backbone_specular' in models, "argument `models` dictionary should contain `'backbone_specular'` key."
        assert 'dncnn' in models, "argument `models` dictionary should contain `'dncnn'` key."
        if train_branches:
            assert 'l_diffuse' in loss_funcs
            assert 'l_specular' in loss_funcs
        if manif_learn:
            assert 'l_manif' in loss_funcs
        assert 'l_recon' in loss_funcs
        assert 'l_test' in loss_funcs
        assert disentanglement_option in ['m11r11', 'm10r01', 'm11r01', 'm10r11']
        
        super(KPCNInterface, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif)
        self.train_branches = train_branches
        self.disentanglement_option = disentanglement_option

    def __str__(self):
        return 'KPCNInterface'

    def to_train_mode(self):
        for model_name in self.models:
            self.models[model_name].train()
            assert 'optim_' + model_name in self.optims, '`optim_%s`: an optimization algorithm is not defined.'%(model_name)
    
    def preprocess(self, batch=None):
        assert 'target_total' in batch
        assert 'target_diffuse' in batch
        assert 'target_specular' in batch
        assert 'kpcn_diffuse_in' in batch
        assert 'kpcn_specular_in' in batch
        assert 'kpcn_diffuse_buffer' in batch
        assert 'kpcn_specular_buffer' in batch
        assert 'kpcn_albedo' in batch
        if self.use_llpm_buf:
            assert 'paths' in batch

        self.iters += 1
    
    def train_batch(self, batch, grad_hook_mode=False):
        out_manif = None

        if self.use_llpm_buf:
            self.models['backbone_diffuse'].zero_grad()
            self.models['backbone_specular'].zero_grad()
            p_buffers = self._manifold_forward(batch)

            if self.iters % 1000 == 1:
                pimg = np.mean(np.transpose(p_buffers['diffuse'].detach().cpu().numpy()[0,:,:3,...], (2, 3, 0, 1)), 2)
                pimg = np.clip(pimg, 0.0, 1.0)
                plt.imsave('../LLPM_results/pbuf_%s_diffuse.png'%(self.args.model_name), pimg)

                pimg = np.mean(np.transpose(p_buffers['specular'].detach().cpu().numpy()[0,:,:3,...], (2, 3, 0, 1)), 2)
                pimg = np.clip(pimg, 0.0, 1.0)
                plt.imsave('../LLPM_results/pbuf_%s_specular.png'%(self.args.model_name), pimg)
            
            """ Feature disentanglement """
            _, _, c, _, _ = p_buffers['diffuse'].shape
            assert c >= 2
            if self.disentanglement_option == 'm11r11':
                out_manif = p_buffers
            elif self.disentanglement_option == 'm10r01':
                out_manif = {
                        'diffuse': p_buffers['diffuse'][:,:,c//2:,...],
                        'specular': p_buffers['specular'][:,:,c//2:,...]
                }
                p_buffers = {
                        'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                        'specular': p_buffers['specular'][:,:,:c//2,...]
                }
            elif self.disentanglement_option == 'm11r01':
                out_manif = p_buffers
                p_buffers = {
                        'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                        'specular': p_buffers['specular'][:,:,:c//2,...]
                }
            elif self.disentanglement_option == 'm10r11':
                out_manif = {
                        'diffuse': p_buffers['diffuse'][:,:,c//2:,...],
                        'specular': p_buffers['specular'][:,:,c//2:,...]
                }

            p_var_diffuse = p_buffers['diffuse'].var(1).mean(1, keepdims=True).detach()
            p_var_diffuse /= p_buffers['diffuse'].shape[1] # spp
            p_var_specular = p_buffers['specular'].var(1).mean(1, keepdims=True).detach()
            p_var_specular /= p_buffers['specular'].shape[1]

            # make a new batch
            batch = {
                'target_total': batch['target_total'],
                'target_diffuse': batch['target_diffuse'],
                'target_specular': batch['target_specular'],
                'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'], p_buffers['diffuse'].mean(1), p_var_diffuse], 1),
                'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'], p_buffers['specular'].mean(1), p_var_specular], 1),
                'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                'kpcn_albedo': batch['kpcn_albedo'],
            }
        
        self.models['dncnn'].zero_grad()
        out = self._regress_forward(batch)

        loss_dict = self._backward(batch, out, out_manif)

        if grad_hook_mode: # do not update this model
            return

        self._logging(loss_dict)

        self._optimization()

    def _manifold_forward(self, batch):
        p_buffer_diffuse = self.models['backbone_diffuse'](batch)
        p_buffer_specular = self.models['backbone_specular'](batch)
        p_buffers = {
            'diffuse': p_buffer_diffuse,
            'specular': p_buffer_specular
        }
        return p_buffers

    def _regress_forward(self, batch):
        return self.models['dncnn'](batch)

    def _backward(self, batch, out, p_buffers):
        assert 'radiance' in out
        assert 'diffuse' in out
        assert 'specular' in out

        total, diffuse, specular = out['radiance'], out['diffuse'], out['specular']
        loss_dict = {}
        tgt_total = crop_like(batch['target_total'], total)
        
        if self.train_branches: # training diffuse and specular branches
            tgt_diffuse = crop_like(batch['target_diffuse'], diffuse)
            L_diffuse = self.loss_funcs['l_diffuse'](diffuse, tgt_diffuse)

            tgt_specular = crop_like(batch['target_specular'], specular)
            L_specular = self.loss_funcs['l_specular'](specular, tgt_specular)

            loss_dict['l_diffuse'] = L_diffuse.detach()
            loss_dict['l_specular'] = L_specular.detach()

            if self.manif_learn:
                p_buffer_diffuse = crop_like(p_buffers['diffuse'], diffuse)
                L_manif_diffuse = self.loss_funcs['l_manif'](p_buffer_diffuse, tgt_diffuse)
                L_diffuse += L_manif_diffuse * self.w_manif

                p_buffer_specular = crop_like(p_buffers['specular'], specular)
                L_manif_specular = self.loss_funcs['l_manif'](p_buffer_specular, tgt_specular)
                L_specular += L_manif_specular * self.w_manif

                loss_dict['l_manif_diffuse'] = L_manif_diffuse.detach()
                loss_dict['l_manif_specular'] = L_manif_specular.detach()
            
            L_diffuse.backward()
            L_specular.backward()

            with torch.no_grad():
                L_total = self.loss_funcs['l_recon'](total, tgt_total)
                loss_dict['l_total'] = L_total.detach()
        else: # post-training the entire system
            L_total = self.loss_funcs['l_recon'](total, tgt_total)
            loss_dict['l_total'] = L_total.detach()
            L_total.backward()
        
        with torch.no_grad():
            loss_dict['rmse'] = self.loss_funcs['l_test'](total, tgt_total).detach()
        
        return loss_dict

    def _logging(self, loss_dict):
        """ error handling """
        for key in loss_dict:
            if not torch.isfinite(loss_dict[key]).all():
                raise RuntimeError("%s: Non-finite loss at train time."%(key))
        
        # (NOTE: modified for each model)
        for model_name in self.models:
            nn.utils.clip_grad_value_(self.models[model_name].parameters(), clip_value=1.0)

        """ logging """
        for key in loss_dict:
            if 'm_' + key not in self.m_losses:
                self.m_losses['m_' + key] = torch.tensor(0.0, device=loss_dict[key].device)
            self.m_losses['m_' + key] += loss_dict[key]
    
    def _optimization(self):
        for model_name in self.models:
            self.optims['optim_' + model_name].step()

    def to_eval_mode(self):
        for model_name in self.models:
            self.models[model_name].eval()
        self.m_losses['m_val'] = torch.tensor(0.0)

    def validate_batch(self, batch):
        p_buffers = None

        if self.use_llpm_buf:
            p_buffers = self._manifold_forward(batch)
            
            """ Feature disentanglement """
            _, _, c, _, _ = p_buffers['diffuse'].shape
            assert c >= 2
            if self.disentanglement_option == 'm10r01' or self.disentanglement_option == 'm11r01':
                p_buffers = {
                        'diffuse': p_buffers['diffuse'][:,:,:c//2,...],
                        'specular': p_buffers['specular'][:,:,:c//2,...]
                }

            p_var_diffuse = p_buffers['diffuse'].var(1).mean(1, keepdims=True).detach()
            p_var_diffuse /= p_buffers['diffuse'].shape[1] # spp
            p_var_specular = p_buffers['specular'].var(1).mean(1, keepdims=True).detach()
            p_var_specular /= p_buffers['specular'].shape[1]

            # make a new batch
            batch = {
                'target_total': batch['target_total'],
                'target_diffuse': batch['target_diffuse'],
                'target_specular': batch['target_specular'],
                'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'], p_buffers['diffuse'].mean(1), p_var_diffuse], 1),
                'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'], p_buffers['specular'].mean(1), p_var_specular], 1),
                'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                'kpcn_albedo': batch['kpcn_albedo'],
            }
        
        out = self._regress_forward(batch)

        tgt_total = crop_like(batch['target_total'], out['radiance'])
        L_total = self.loss_funcs['l_test'](out['radiance'], tgt_total)
        if self.m_losses['m_val'] == 0.0 and self.m_losses['m_val'].device != L_total.device:
            self.m_losses['m_val'] = torch.tensor(0.0, device=L_total.device)
        self.m_losses['m_val'] += L_total.detach()

        return out['radiance'], p_buffers

    def get_epoch_summary(self, mode, norm):
        if mode == 'train':
            print('[][][]', end=' ')
            for key in self.m_losses:
                if key == 'm_val':
                    continue
                tr_l_tmp = self.m_losses[key] / (norm * 2)
                tr_l_tmp *= 1000
                print('%s: %.3fE-3'%(key, tr_l_tmp), end='\t')
                self.m_losses[key] = torch.tensor(0.0, device=self.m_losses[key].device)
            print('')
            return -1.0
        else:
            return self.m_losses['m_val'].item() / (norm * 2)


class SBMCInterface(BaseInterface):

    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1, use_sbmc_buf=True, disentangle="m11r11"):
        if manif_learn:
            assert 'backbone' in models, "argument `models` dictionary should contain `'backbone'` key."
        assert 'dncnn' in models, "argument `models` dictionary should contain `'dncnn'` key."
        if manif_learn:
            assert 'l_manif' in loss_funcs
        assert 'l_recon' in loss_funcs
        assert 'l_test' in loss_funcs
        assert disentangle in ['m11r11', 'm10r01', 'm11r01', 'm10r11']

        super(SBMCInterface, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif)
        self.disentangle = disentangle
        self.use_sbmc_buf = use_sbmc_buf
    
    def __str__(self):
        return 'SBMCInterface'

    def to_train_mode(self):
        for model_name in self.models:
            self.models[model_name].train()
            assert 'optim_' + model_name in self.optims, '`optim_%s`: an optimization algorithm is not defined.'%(model_name)
    
    def preprocess(self, batch=None):
        assert 'target_image' in batch
        assert 'radiance' in batch
        assert 'features' in batch
        if self.use_llpm_buf:
            assert 'paths' in batch
        self.iters += 1
    
    def train_batch(self, batch, grad_hook_mode=False):
        out_manif = None

        if self.use_llpm_buf:
            self.models['backbone'].zero_grad()
            p_buffer = self._manifold_forward(batch)

            if self.iters % 1000 == 1:
                pimg = np.mean(np.transpose(p_buffer.detach().cpu().numpy()[0,:,:3,...], (2, 3, 0, 1)), 2)
                pimg = np.clip(pimg, 0.0, 1.0)
                plt.imsave('../LLPM_results/pbuf_%s.png'%(self.args.model_name), pimg)
            
            """ Feature disentanglement """
            _, s, c, _, _ = p_buffer.shape
            assert c >= 2
            if self.disentangle == 'm11r11':
                out_manif = p_buffer
            elif self.disentangle == 'm10r01':
                out_manif = p_buffer[:,:,c//2:,...]
                p_buffer = p_buffer[:,:,:c//2,...]
            elif self.disentangle == 'm11r01':
                out_manif = p_buffer
                p_buffer = p_buffer[:,:,:c//2,...]
            elif self.disentangle == 'm10r11':
                out_manif = p_buffer[:,:,c//2:,...]

            p_var = p_buffer.var(1).mean(1, keepdims=True)
            p_var /= s # spp
            p_var = torch.stack([p_var,]*s, axis=1).detach()

            # make a new batch
            batch = {
                'target_image': batch['target_image'],
                'radiance': batch['radiance'],
                'features':  torch.cat([batch['features'], p_buffer, p_var], 2),
            }

        self.models['dncnn'].zero_grad()
        out = self._regress_forward(batch)

        loss_dict = self._backward(batch, out, out_manif)

        if grad_hook_mode: # do not update this model
            return

        self._logging(loss_dict)

        self._optimization()
        
    def _manifold_forward(self, batch):
        return self.models['backbone'](batch)

    def _regress_forward(self, batch):
        return self.models['dncnn'](batch)
    
    def _backward(self, batch, out, p_buffer):
        loss_dict = {}
        tgt_total = crop_like(batch['target_image'], out)

        L_total = self.loss_funcs['l_recon'](out, tgt_total)

        if self.manif_learn:
            p_buffer = crop_like(p_buffer, out)
            L_manif = self.loss_funcs['l_manif'](p_buffer, tgt_total)

            loss_dict['l_manif'] = L_manif.detach()
            loss_dict['l_recon'] = L_total.detach()

            L_total += L_manif * self.w_manif
        
        loss_dict['l_total'] = L_total.detach()

        L_total.backward()

        with torch.no_grad():
            loss_dict['rmse'] = self.loss_funcs['l_test'](out, tgt_total).detach()

        return loss_dict

    def _logging(self, loss_dict):
        """ error handling """
        for key in loss_dict:
            if not torch.isfinite(loss_dict[key]).all():
                raise RuntimeError("%s: Non-finite loss at train time."%(key))
        
        # (NOTE: modified for each model)
        clip = 1000
        for model_name in self.models:
            actual = nn.utils.clip_grad_norm_(self.models[model_name].parameters(), max_norm=clip)
            if actual > clip:
                print("Clipped %s gradients %f -> %f"%(model_name, clip, actual))

        """ logging """
        for key in loss_dict:
            if 'm_' + key not in self.m_losses:
                self.m_losses['m_' + key] = torch.tensor(0.0, device=loss_dict[key].device)
            self.m_losses['m_' + key] += loss_dict[key]
    
    def _optimization(self):
        for model_name in self.models:
            self.optims['optim_' + model_name].step()

    def to_eval_mode(self):
        for model_name in self.models:
            self.models[model_name].eval()
        self.m_losses['m_val'] = torch.tensor(0.0)
    
    def validate_batch(self, batch):
        p_buffer = None

        if self.use_llpm_buf:
            p_buffer = self._manifold_forward(batch)
            
            """ Feature disentanglement """
            _, s, c, _, _ = p_buffer.shape
            assert c >= 2
            if self.disentangle == 'm10r01':
                p_buffer = p_buffer[:,:,:c//2,...]
            elif self.disentangle == 'm11r01':
                p_buffer = p_buffer[:,:,:c//2,...]

            p_var = p_buffer.var(1).mean(1, keepdims=True)
            p_var /= s # spp
            p_var = torch.stack([p_var,]*s, axis=1).detach()

            # make a new batch
            batch = {
                'target_image': batch['target_image'],
                'radiance': batch['radiance'],
                'features':  torch.cat([batch['features'], p_buffer, p_var], 2),
            }
        
        out = self._regress_forward(batch)

        tgt_total = crop_like(batch['target_image'], out)
        L_total = self.loss_funcs['l_test'](out, tgt_total)
        if self.m_losses['m_val'] == 0.0 and self.m_losses['m_val'].device != L_total.device:
            self.m_losses['m_val'] = torch.tensor(0.0, device=L_total.device)
        self.m_losses['m_val'] += L_total.detach()

        return out, p_buffer
    
    def get_epoch_summary(self, mode, norm):
        if mode == 'train':
            print('[][][]', end=' ')
            for key in self.m_losses:
                if key == 'm_val':
                    continue
                tr_l_tmp = self.m_losses[key] / (norm * 2)
                tr_l_tmp *= 1000
                print('%s: %.3fE-3'%(key, tr_l_tmp), end='\t')
                self.m_losses[key] = torch.tensor(0.0, device=self.m_losses[key].device)
            print('')
            return -1.0
        else:
            return self.m_losses['m_val'].item() / (norm * 2)


class KPCNRefInterface(KPCNInterface):

    def __init__(self, models, optims, loss_funcs, args, visual=False, use_llpm_buf=False, manif_learn=False, w_manif=0.1, train_branches=True):
        assert not use_llpm_buf
        assert not manif_learn
        if train_branches:
            assert 'l_diffuse' in loss_funcs
            assert 'l_specular' in loss_funcs
        assert 'l_recon' in loss_funcs
        assert 'l_test' in loss_funcs

        super(KPCNRefInterface, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif, train_branches)
        self.train_branches = train_branches
    
    def train_batch(self, batch):
        out_manif = None

        batch = {
            'target_total': batch['target_total'],
            'target_diffuse': batch['target_diffuse'],
            'target_specular': batch['target_specular'],
            'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'], batch['target_diffuse']], 1),
            'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'], batch['target_specular']], 1),
            'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
            'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
            'kpcn_albedo': batch['kpcn_albedo'],
        }

        self.models['dncnn'].zero_grad()
        out = self._regress_forward(batch)

        loss_dict = self._backward(batch, out, out_manif)

        self._logging(loss_dict)

        self._optimization()
    
    def validate_batch(self, batch):
        p_buffers = None
        
        batch = {
            'target_total': batch['target_total'],
            'target_diffuse': batch['target_diffuse'],
            'target_specular': batch['target_specular'],
            'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'], batch['target_diffuse']], 1),
            'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'], batch['target_specular']], 1),
            'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
            'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
            'kpcn_albedo': batch['kpcn_albedo'],
        }

        out = self._regress_forward(batch)

        tgt_total = crop_like(batch['target_total'], out['radiance'])
        L_total = self.loss_funcs['l_test'](out['radiance'], tgt_total)
        if self.m_losses['m_val'] == 0.0 and self.m_losses['m_val'].device != L_total.device:
            self.m_losses['m_val'] = torch.tensor(0.0, device=L_total.device)
        self.m_losses['m_val'] += L_total.detach()

        return out['radiance'], p_buffers


class KPCNPreInterface(KPCNInterface):

    def __init__(self, models, optims, loss_funcs, args, visual=False, manif_learn=False, w_manif=0.1, train_branches=True):
        # if manif_learn is True, pre-train the manifold feature extractor.
        # else, train KPCN using the freezed & pre-trained feature extractor.
        if train_branches:
            assert 'l_diffuse' in loss_funcs
            assert 'l_specular' in loss_funcs
        assert 'l_recon' in loss_funcs
        assert 'l_test' in loss_funcs
        
        use_llpm_buf = True
        super(KPCNPreInterface, self).__init__(models, optims, loss_funcs, args, visual, use_llpm_buf, manif_learn, w_manif, train_branches)
    
    def to_train_mode(self):
        for model_name in self.models:
            if self.manif_learn:
                if 'dncnn' in model_name: # KPCN
                    self.models[model_name].eval()
                if 'backbone' in model_name: # manifold feature extractor
                    self.models[model_name].train()
            else:
                if 'dncnn' in model_name: # KPCN
                    self.models[model_name].train()
                if 'backbone' in model_name: # manifold feature extractor
                    self.models[model_name].eval()
            assert 'optim_' + model_name in self.optims, '`optim_%s`: an optimization algorithm is not defined.'%(model_name)
    
    def train_batch(self, batch):
        out_manif = None

        if self.manif_learn:
            self.models['backbone_diffuse'].zero_grad()
            self.models['backbone_specular'].zero_grad()

            p_buffers = self._manifold_forward(batch)

            if self.iters % 1000 == 1:
                pimg = np.mean(np.transpose(p_buffers['diffuse'].detach().cpu().numpy()[0,:,:3,...], (2, 3, 0, 1)), 2)
                pimg = np.clip(pimg, 0.0, 1.0)
                plt.imsave('../LLPM_results/pbuf_%s_diffuse.png'%(self.args.model_name), pimg)

                pimg = np.mean(np.transpose(p_buffers['specular'].detach().cpu().numpy()[0,:,:3,...], (2, 3, 0, 1)), 2)
                pimg = np.clip(pimg, 0.0, 1.0)
                plt.imsave('../LLPM_results/pbuf_%s_specular.png'%(self.args.model_name), pimg)
            
            out_manif = p_buffers

            loss_dict = self._backward(batch, None, out_manif)

            self._logging(loss_dict)

            self._optimization()
        else:
            self.models['backbone_diffuse'].zero_grad()
            self.models['backbone_specular'].zero_grad()
            self.models['dncnn'].zero_grad()

            p_buffers = self._manifold_forward(batch)
            out_manif = p_buffers

            p_var_diffuse = p_buffers['diffuse'].var(1).mean(1, keepdims=True).detach()
            p_var_diffuse /= p_buffers['diffuse'].shape[1] # spp
            p_var_specular = p_buffers['specular'].var(1).mean(1, keepdims=True).detach()
            p_var_specular /= p_buffers['specular'].shape[1]

            # make a new batch
            batch = {
                'target_total': batch['target_total'],
                'target_diffuse': batch['target_diffuse'],
                'target_specular': batch['target_specular'],
                'kpcn_diffuse_in': torch.cat([batch['kpcn_diffuse_in'], p_buffers['diffuse'].mean(1), p_var_diffuse], 1),
                'kpcn_specular_in': torch.cat([batch['kpcn_specular_in'], p_buffers['specular'].mean(1), p_var_specular], 1),
                'kpcn_diffuse_buffer': batch['kpcn_diffuse_buffer'],
                'kpcn_specular_buffer': batch['kpcn_specular_buffer'],
                'kpcn_albedo': batch['kpcn_albedo'],
            }

            out = self._regress_forward(batch)

            loss_dict = self._backward(batch, out, None)

            self._logging(loss_dict)

            self._optimization()
    
    def _backward(self, batch, out, p_buffers):
        assert not out or 'radiance' in out
        assert not out or 'diffuse' in out
        assert not out or 'specular' in out

        if out:
            total, diffuse, specular = out['radiance'], out['diffuse'], out['specular']
            tgt_total = crop_like(batch['target_total'], total)
        loss_dict = {}

        if self.manif_learn:
            tgt_diffuse = batch['target_diffuse'] #crop_like(batch['target_diffuse'], diffuse)
            tgt_specular = batch['target_specular'] #crop_like(batch['target_specular'], specular)

            p_buffer_diffuse = p_buffers['diffuse'] #crop_like(p_buffers['diffuse'], diffuse)
            L_manif_diffuse = self.loss_funcs['l_manif'](p_buffer_diffuse, tgt_diffuse) * self.w_manif

            p_buffer_specular = p_buffers['specular'] #crop_like(p_buffers['specular'], specular)
            L_manif_specular = self.loss_funcs['l_manif'](p_buffer_specular, tgt_specular) * self.w_manif

            loss_dict['l_manif_diffuse'] = L_manif_diffuse.detach() / self.w_manif
            loss_dict['l_manif_specular'] = L_manif_specular.detach() / self.w_manif

            L_manif_diffuse.backward()
            L_manif_specular.backward()
        elif self.train_branches:
            tgt_diffuse = crop_like(batch['target_diffuse'], diffuse)
            L_diffuse = self.loss_funcs['l_diffuse'](diffuse, tgt_diffuse)

            tgt_specular = crop_like(batch['target_specular'], specular)
            L_specular = self.loss_funcs['l_specular'](specular, tgt_specular)

            loss_dict['l_diffuse'] = L_diffuse.detach()
            loss_dict['l_specular'] = L_specular.detach()

            L_diffuse.backward()
            L_specular.backward()

            with torch.no_grad():
                L_total = self.loss_funcs['l_recon'](total, tgt_total)
                loss_dict['l_total'] = L_total.detach()
        else:
            L_total = self.loss_funcs['l_recon'](total, tgt_total)
            loss_dict['l_total'] = L_total.detach()
            L_total.backward()

        return loss_dict

    def _logging(self, loss_dict):
        """ error handling """
        for key in loss_dict:
            if not torch.isfinite(loss_dict[key]).all():
                raise RuntimeError("%s: Non-finite loss at train time."%(key))
        
        # (NOTE: modified for each model)
        for model_name in self.models:
            if self.manif_learn:
                if 'backbone' in model_name: # manifold feature extractor
                    nn.utils.clip_grad_value_(self.models[model_name].parameters(), clip_value=1.0)
            else:
                if 'dncnn' in model_name: # KPCN
                    nn.utils.clip_grad_value_(self.models[model_name].parameters(), clip_value=1.0)

        """ logging """
        for key in loss_dict:
            if 'm_' + key not in self.m_losses:
                self.m_losses['m_' + key] = torch.tensor(0.0, device=loss_dict[key].device)
            self.m_losses['m_' + key] += loss_dict[key]
    
    def _optimization(self):
        for model_name in self.models:
            if self.manif_learn:
                if 'backbone' in model_name:
                    self.optims['optim_' + model_name].step()
            else:
                if 'dncnn' in model_name:
                    self.optims['optim_' + model_name].step()


class LBMCInterface(SBMCInterface):

    def __init__(self, models, optims, loss_funcs, args, use_llpm_buf=False, manif_learn=False, w_manif=0.1, disentangle='m11r11'):
        if manif_learn:
            assert 'backbone' in models, "argument `models` dictionary should contain `'backbone'` key."
        assert 'dncnn' in models, "argument `models` dictionary should contain `'dncnn'` key."
        if manif_learn:
            assert 'l_manif' in loss_funcs
        assert 'l_recon' in loss_funcs
        assert 'l_test' in loss_funcs
        assert disentangle in ['m11r11', 'm10r01', 'm11r01', 'm10r11']

        super(LBMCInterface, self).__init__(models, optims, loss_funcs, args, False, use_llpm_buf, manif_learn, w_manif, False, disentangle)
    
    def __str__(self):
        return 'LBMCInterface'
    
    def train_batch(self, batch, grad_hook_mode=False):
        out_manif = None

        if self.use_llpm_buf:
            self.models['backbone'].zero_grad()
            p_buffer = self._manifold_forward(batch)

            if self.iters % 1000 == 1:
                pimg = np.mean(np.transpose(p_buffer.detach().cpu().numpy()[0,:,:3,...], (2, 3, 0, 1)), 2)
                pimg = np.clip(pimg, 0.0, 1.0)
                plt.imsave('../LLPM_results/pbuf_%s.png'%(self.args.model_name), pimg)
            
            """ Feature disentanglement """
            _, s, c, _, _ = p_buffer.shape
            assert c >= 2
            if self.disentangle == 'm11r11':
                out_manif = p_buffer
            elif self.disentangle == 'm10r01':
                out_manif = p_buffer[:,:,c//2:,...]
                p_buffer = p_buffer[:,:,:c//2,...]
            elif self.disentangle == 'm11r01':
                out_manif = p_buffer
                p_buffer = p_buffer[:,:,:c//2,...]
            elif self.disentangle == 'm10r11':
                out_manif = p_buffer[:,:,c//2:,...]

            p_var = p_buffer.var(1).mean(1, keepdims=True)
            p_var /= s # spp
            p_var = torch.stack([p_var,]*s, axis=1).detach()

            # make a new batch
            batch = {
                'target_image': batch['target_image'],
                'radiance': batch['radiance'],
                'features':  torch.cat([batch['features'], p_buffer, p_var], 2),
            }
        
        self.models['dncnn'].zero_grad()
        out = self._regress_forward(batch)

        loss_dict = self._backward(batch, out, out_manif)

        if grad_hook_mode: # do not update this model
            return

        self._logging(loss_dict)

        self._optimization()
    
    def _logging(self, loss_dict):
        """ error handling """
        for key in loss_dict:
            if not torch.isfinite(loss_dict[key]).all():
                raise RuntimeError("%s: Non-finite loss at train time."%(key))
        
        # (NOTE: modified for each model)
        GRADIENT_CLAMP_N = 0.25 * 1000
        #GRADIENT_CLAMP   = 0.001 * 1000
        for model_name in self.models:
            actual = nn.utils.clip_grad_norm_(self.models[model_name].parameters(), GRADIENT_CLAMP_N)
            #nn.utils.clip_grad_value_(self.models[model_name].parameters(), GRADIENT_CLAMP)
            if actual > GRADIENT_CLAMP_N:
                print("Clipped %s gradients %f -> %f"%(model_name, GRADIENT_CLAMP_N, actual))
                #self.models[model_name].zero_grad()

        """ logging """
        for key in loss_dict:
            if 'm_' + key not in self.m_losses:
                self.m_losses['m_' + key] = torch.tensor(0.0, device=loss_dict[key].device)
            self.m_losses['m_' + key] += loss_dict[key]
