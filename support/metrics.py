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
"""Helpers to evaluate on the rendering results."""

import numpy as np
from skimage.metrics import structural_similarity as ssim


def _tonemap(im):
    col = np.clip(np.copy(im), 0.0, a_max=None)
    col /= (1.0 + col)
    return col


def MSE(im, ref, reduce=True):
    """Mean-squared error between images.
    Args:
        im(np.array): image to test.
        ref(np.array): reference for the comparison.
    Returns:
        (float) error value.
    """
    return np.square(im-ref).mean() if reduce else np.square(im-ref)


def RelMSE(im, ref, eps=1e-4, reduce=True):
    """Relative Mean-squared error between images.
    Args:
        im(np.array): image to test.
        ref(np.array): reference for the comparison.
        eps(float): small number to prevent division by 0.
    Returns:
        (float) error value.
    """
    diff = (np.square(im-ref) / (np.square(ref) + eps))
    diff = np.ravel(diff)
    diff = diff[~np.isnan(diff)]
    return diff.mean() if reduce else diff


def TRelMSE(im, ref, eps=1e-4, reduce=True):
    im = _tonemap(im)
    ref = _tonemap(ref)
    return RelMSE(im, ref, eps, reduce)


def L1(im, ref, reduce=True):
    """Absolute error between images.
    Args:
        im(np.array): image to test.
        ref(np.array): reference for the comparison.
    Returns:
        (float) error value.
    """
    return np.abs(im-ref).mean() if reduce else np.abs(im-ref)


def RelL1(im, ref, eps=1e-4, reduce=True):
    """Relative absolute error between images.
    Args:
        im(np.array): image to test.
        ref(np.array): reference for the comparison.
        eps(float): small number to prevent division by 0.
    Returns:
        (float) error value.
    """
    diff = np.abs(im-ref) / (np.abs(ref) + eps)
    return diff.mean() if reduce else diff


def SSIM(im, ref, reduce=True):
    """Structural Similarity error (1-SSIM, or DSSIM).
    Args:
        im(np.array): image to test.
        ref(np.array): reference for the comparison.
    Returns:
        (float) error value.
    """
    return 1-ssim(im, ref, multichannel=True, full=(not reduce))