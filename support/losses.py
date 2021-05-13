import torch
from kornia import rgb_to_hls
import math


__all__ = ["GlobalRelativeSimilarityLoss", "FeatureMSE", "RelativeMSE", "SMAPE", "TonemappedMSE", "TonemappedRelativeMSE"]


class FeatureMSE(torch.nn.Module):
    """Feature Mean-Squared Error. Path disentangling loss
    """
    def __init__(self, color='rgb', non_local=True):
        super(FeatureMSE, self).__init__()
        self.color = color
        self.non_local = non_local
        print('FeatureMSE locality: %s'%('Non-local' if non_local else 'Local'))
    
    def intra_pixel_dist(self, p_buffer, ref):
        b, s, c, h, w = p_buffer.shape
        idx = torch.randperm(s)

        ref_1 = ref.permute(0, 3, 4, 1, 2).reshape(b * h * w, s, 3)
        ref_2 = ref_1[:, idx, :]
        mse_rad_inter = 0.5 * torch.sum(torch.pow(ref_1 - ref_2, 2), dim=2)

        p_1 = p_buffer.permute(0, 1, 3, 4, 2).reshape(b, s * h * w, c)
        p_2 = p_1[:, idx, :]
        mse_p_inter = 0.5 * torch.sum(torch.pow(p_1 - p_2, 2), dim=2)

        loss = 0.5 * torch.mean(torch.pow(mse_p_inter - mse_rad_inter, 2))
        return loss
    
    def intra_patch_dist(self, p_buffer, ref):
        b, s, c, h, w = p_buffer.shape
        idx = torch.randperm(s * h * w)

        ref_1 = ref.permute(0, 1, 3, 4, 2).reshape(b, s * h * w, 3)
        ref_2 = ref_1[:, idx, :]
        mse_rad_inter = 0.5 * torch.sum(torch.pow(ref_1 - ref_2, 2), dim=2)

        p_1 = p_buffer.permute(0, 1, 3, 4, 2).reshape(b, s * h * w, c)
        p_2 = p_1[:, idx, :]
        mse_p_inter = 0.5 * torch.sum(torch.pow(p_1 - p_2, 2), dim=2)

        loss = 0.5 * torch.mean(torch.pow(mse_p_inter - mse_rad_inter, 2))
        return loss
    
    def intra_batch_dist(self, p_buffer, ref):
        b, s, c, h, w = p_buffer.shape
        idx = torch.randperm(b * s * h * w)

        ref_1 = ref.permute(0, 1, 3, 4, 2).reshape(-1, 3)
        ref_2 = ref_1[idx, :]
        mse_rad_inter = 0.5 * torch.sum(torch.pow(ref_1 - ref_2, 2), dim=1)

        p_1 = p_buffer.permute(0, 1, 3, 4, 2).reshape(-1, c)
        p_2 = p_1[idx, :]
        mse_p_inter = 0.5 * torch.sum(torch.pow(p_1 - p_2, 2), dim=1)

        loss = 0.5 * torch.mean(torch.pow(mse_p_inter - mse_rad_inter, 2))
        return loss
    
    def _tonemap_gamma(self, img):
        img = torch.clamp(img, min=0)
        return (img / (1 + img)) ** 0.454545

    def _preprocess(self, img):
        """Convert to the Cartesian HLS coordinates
        Args:
            img(torch.Tensor): three channel image. (*, C=3, H, W)
        """
        img = rgb_to_hls(self._tonemap_gamma(img))
        
        theta = img[...,0,:,:].clone()
        r = img[...,2,:,:].clone()
        img[...,0,:,:] = r * torch.cos(theta)
        img[...,1,:,:] *= 2
        img[...,2,:,:] = r * torch.sin(theta)

        return img

    def forward(self, p_buffer, ref):
        """Evaluate the metric.
        Args:
            p_buffer(torch.Tensor): embedded path. (B, S, C=3, H, W)
            ref(torch.Tensor): reference radiance. (B, C=3, H, W)
        """
        # convert to the same HLS space
        if self.color == 'hls':
            p_buffer = self._preprocess(p_buffer)
            ref = self._preprocess(ref)
        else:
            #p_buffer = self._tonemap_gamma(p_buffer)
            ref = self._tonemap_gamma(ref)
        
        _, s, _, _, _ = p_buffer.shape
        ref = torch.stack((ref,) * s, dim=1)

        if not torch.isfinite(p_buffer).all():
            raise RuntimeError("Infinite loss at train time.")
        if not torch.isfinite(ref).all():
            raise RuntimeError("Infinite loss at train time.")

        # intra-patch
        loss_p = self.intra_patch_dist(p_buffer, ref)

        # intra-batch
        if self.non_local:
            loss_b = self.intra_batch_dist(p_buffer, ref)
        else:
            loss_b = loss_p

        return loss_p + loss_b


class GlobalRelativeSimilarityLoss(torch.nn.Module):
    """Global Relative Similarity Loss.

    Focused on uniformly reducing the norm of feature errors. 
    But doesn't give superior improvements.
    
    Contrastive2: z = x^2 + y^2
    Contrastive1: z = |x| + |y|
    GRS         : z = ln(1 + exp(x) + exp(-x) + exp(y) + exp(-y))

    Inspired by 
        RelocNet [Balntas et al. ECCV 2018]
        Multi-Similarity Loss [Wang et al. CVPR 2019]
    """

    def __init__(self, alpha=2, color='rgb'):
        super(GlobalRelativeSimilarityLoss, self).__init__()
        self.color = color
        self.alpha = alpha
    
    def intra_patch_dist(self, p_buffer, ref):
        b, s, c, h, w = p_buffer.shape
        idx = torch.randperm(s * h * w)

        ref_1 = ref.permute(0, 1, 3, 4, 2).reshape(b, s * h * w, 3)
        ref_2 = ref_1[:, idx, :]
        mse_rad_inter = 0.5 * torch.sum(torch.pow(ref_1 - ref_2, 2), dim=2)

        p_1 = p_buffer.permute(0, 1, 3, 4, 2).reshape(b, s * h * w, c)
        p_2 = p_1[:, idx, :]
        mse_p_inter = 0.5 * torch.sum(torch.pow(p_1 - p_2, 2), dim=2)

        disp = mse_p_inter - mse_rad_inter # displacement
        disp = disp.reshape(b * s * h * w)
        return disp
    
    def intra_batch_dist(self, p_buffer, ref):
        b, s, c, h, w = p_buffer.shape
        idx = torch.randperm(b * s * h * w)

        ref_1 = ref.permute(0, 1, 3, 4, 2).reshape(-1, 3)
        ref_2 = ref_1[idx, :]
        mse_rad_inter = 0.5 * torch.sum(torch.pow(ref_1 - ref_2, 2), dim=1)

        p_1 = p_buffer.permute(0, 1, 3, 4, 2).reshape(-1, c)
        p_2 = p_1[idx, :]
        mse_p_inter = 0.5 * torch.sum(torch.pow(p_1 - p_2, 2), dim=1)

        disp = mse_p_inter - mse_rad_inter # displacement
        return disp
    
    def _tmap1(self, img):
        # img channel 수 무관
        img = torch.clamp(img, min=0)
        return (img / (1 + img)) ** 0.454545

    def _tmap2(self, img):
        # img channel = 3
        img = torch.clamp(img, min=0)
        lum = 0.2126 * img[...,0:1,:,:] + 0.7152 * img[...,1:2,:,:] + \
              0.0722 * img[...,2:3,:,:]
        return (img / (1 + lum)) ** 0.454545
    
    def _tmap3(self, img):
        img = torch.clamp(img, min=0)
        lum = img[...,0:1,:,:] + img[...,1:2,:,:] + img[...,2:3,:,:]
        lum /= 3.0
        return (img / (1 + lum)) ** 0.454545
    
    def forward(self, p_buffer, ref):
        """Evaluate the metric.
        Args:
            p_buffer(torch.Tensor): embedded path. (B, S, C, H, W)
            ref(torch.Tensor): reference radiance. (B, C=3, H, W)
        """

        if not torch.isfinite(p_buffer).all():
            raise RuntimeError("Infinite loss at train time.")
        if not torch.isfinite(ref).all():
            raise RuntimeError("Infinite loss at train time.")

        #p_buffer = self._tmap1(p_buffer)
        ref = self._tmap1(ref)

        b, s, c, h, w = p_buffer.shape
        ref = torch.stack((ref,) * s, dim=1)

        disp_p = self.intra_patch_dist(p_buffer, ref)
        disp_b = self.intra_batch_dist(p_buffer, ref)

        zero = torch.zeros((1), dtype=p_buffer.dtype, device=p_buffer.device)
        exp = self.alpha * torch.cat([disp_p, disp_b, -disp_p, -disp_b, zero], dim=0)
        output = torch.logsumexp(exp, dim=0) - math.log(1 + 4 * b * s * h * w)
        output /= math.sqrt(self.alpha)

        return output

"""
The below code is written by Michaël Gharbi.
"""
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

def _tonemap(im):
    """Helper Reinhards tonemapper.
    Args:
        im(torch.Tensor): image to tonemap.
    Returns:
        (torch.Tensor) tonemaped image.
    """
    im = torch.clamp(im, min=0)
    return im / (1+im)


class RelativeMSE(torch.nn.Module):
    """Relative Mean-Squared Error.
    :math:`0.5 * \\frac{(x - y)^2}{y^2 + \epsilon}`
    Args:
        eps(float): small number to avoid division by 0.
    """
    def __init__(self, eps=1e-2):
        super(RelativeMSE, self).__init__()
        self.eps = eps

    def forward(self, im, ref):
        """Evaluate the metric.
        Args:
            im(torch.Tensor): image.
            ref(torch.Tensor): reference.
        """
        mse = torch.pow(im-ref, 2)
        loss = mse/(torch.pow(ref, 2) + self.eps)
        loss = 0.5*torch.mean(loss)
        return loss


class SMAPE(torch.nn.Module):
    """Symmetric Mean Absolute error.
    :math:`\\frac{|x - y|} {|x| + |y| + \epsilon}`
    Args:
        eps(float): small number to avoid division by 0.
    """

    def __init__(self, eps=1e-2):
        super(SMAPE, self).__init__()
        self.eps = eps

    def forward(self, im, ref):
        # NOTE: the denominator is used to scale the loss, but does not
        # contribute gradients, hence the '.detach()' call.
        loss = (torch.abs(im-ref) / (
            self.eps + torch.abs(im.detach()) + torch.abs(ref.detach()))).mean()

        return loss


class TonemappedMSE(torch.nn.Module):
    """Mean-squared error on tonemaped images.
    Args:
        eps(float): small number to avoid division by 0.
    """

    def __init__(self, eps=1e-2):
        super(TonemappedMSE, self).__init__()
        self.eps = eps  # avoid division by zero

    def forward(self, im, ref):
        im = _tonemap(im)
        ref = _tonemap(ref)
        loss = torch.pow(im-ref, 2)
        loss = 0.5*torch.mean(loss)
        return loss


class TonemappedRelativeMSE(torch.nn.Module):
    """Relative mean-squared error on tonemaped images.
    Args:
        eps(float): small number to avoid division by 0.
    """
    def __init__(self, eps=1e-2):
        super(TonemappedRelativeMSE, self).__init__()
        self.eps = eps  # avoid division by zero

    def forward(self, im, ref):
        im = _tonemap(im)
        ref = _tonemap(ref)
        mse = torch.pow(im-ref, 2)
        loss = mse/(torch.pow(ref, 2) + self.eps)
        loss = 0.5*torch.mean(loss)
        return loss


def _tonemap(im):
    """Helper Reinhards tonemapper.
    Args:
        im(torch.Tensor): image to tonemap.
    Returns:
        (torch.Tensor) tonemaped image.
    """
    im = torch.clamp(im, min=0)
    return im / (1+im)
