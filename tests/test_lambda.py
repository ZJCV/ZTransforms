# -*- coding: utf-8 -*-

"""
@date: 2021/1/18 下午2:26
@file: test_lambda.py
@author: zj
@description: 
"""

import torch
# import torchvision.transforms as transforms
import ztransforms.cls_transforms as transforms


def test_lambda():
    trans = transforms.Lambda(lambda x: x.add(10))
    x = torch.randn(10)
    y = trans(x)
    # self.assertTrue(y.equal(torch.add(x, 10)))
    assert y.equal(torch.add(x, 10))

    trans = transforms.Lambda(lambda x: x.add_(10))
    x = torch.randn(10)
    y = trans(x)
    # self.assertTrue(y.equal(x))
    assert y.equal(x)

    # Checking if Lambda can be printed as string
    trans.__repr__()
