# -*- coding: utf-8 -*-

"""
@date: 2021/1/21 上午10:12
@file: functional_albumentation.py
@author: zj
@description: 
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Any, Optional, Sequence

import albumentations as A


@torch.jit.unused
def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)


@torch.jit.unused
def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}


@torch.jit.unused
def _get_image_size(img: Any) -> List[int]:
    """
    Returns image size as [w, h]
    """
    if _is_numpy_image(img):
        return img.shape[1::-1]
    raise TypeError("Unexpected type {}".format(type(img)))


@torch.jit.unused
def resize(img, size, interpolation=cv2.INTER_LINEAR):
    if not _is_numpy_image(img):
        raise TypeError('img should be Numpy Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, Sequence) and len(size) in (1, 2))):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if interpolation not in [cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_NEAREST]:
        raise ValueError("This interpolation mode is unsupported with Numpy input")

    if isinstance(size, int) or len(size) == 1:
        if isinstance(size, Sequence):
            size = size[0]
        h, w = img.shape[:2]
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return A.resize(img, oh, ow, interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return A.resize(img, oh, ow, interpolation)
    else:
        oh, ow = size[:2]
        return A.resize(img, oh, ow, interpolation)


@torch.jit.unused
def crop(img: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    if not _is_numpy_image(img):
        raise TypeError('img should be Numpy Image. Got {}'.format(type(img)))

    return img[top:top + height, left:left + width]
