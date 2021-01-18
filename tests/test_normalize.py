# -*- coding: utf-8 -*-

"""
@date: 2021/1/18 下午3:04
@file: test_normalize.py
@author: zj
@description: 
"""

import numpy as np
import torch
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as F
import ztransforms.cls_transforms as transforms
import ztransforms.cls_functional as F


def test_normalize_different_dtype():
    for dtype1 in [torch.float32, torch.float64]:
        img = torch.rand(3, 10, 10, dtype=dtype1)
        for dtype2 in [torch.int64, torch.float32, torch.float64]:
            mean = torch.tensor([1, 2, 3], dtype=dtype2)
            std = torch.tensor([1, 2, 1], dtype=dtype2)
            # checks that it doesn't crash
            # transforms.functional.normalize(img, mean, std)
            F.normalize(img, mean, std)


def test_normalize_3d_tensor():
    torch.manual_seed(28)
    n_channels = 3
    img_size = 10
    mean = torch.rand(n_channels)
    std = torch.rand(n_channels)
    img = torch.rand(n_channels, img_size, img_size)
    target = F.normalize(img, mean, std).numpy()

    mean_unsqueezed = mean.view(-1, 1, 1)
    std_unsqueezed = std.view(-1, 1, 1)
    result1 = F.normalize(img, mean_unsqueezed, std_unsqueezed)
    result2 = F.normalize(img,
                          mean_unsqueezed.repeat(1, img_size, img_size),
                          std_unsqueezed.repeat(1, img_size, img_size))
    # assert_array_almost_equal(target, result1.numpy())
    # assert_array_almost_equal(target, result2.numpy())
    assert np.allclose(target, result1.numpy())
    assert np.allclose(target, result2.numpy())
