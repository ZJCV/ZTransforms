# -*- coding: utf-8 -*-

"""
@date: 2021/1/15 下午3:09
@file: test_crop.py
@author: zj
@description: 
"""

import torch
# import torchvision.transforms as transforms
import ztransforms.cls_transforms as transforms
import random


def test_crop():
    height = random.randint(10, 32) * 2
    width = random.randint(10, 32) * 2
    oheight = random.randint(5, (height - 2) / 2) * 2
    owidth = random.randint(5, (width - 2) / 2) * 2

    img = torch.ones(3, height, width)
    oh1 = (height - oheight) // 2
    ow1 = (width - owidth) // 2
    imgnarrow = img[:, oh1:oh1 + oheight, ow1:ow1 + owidth]
    imgnarrow.fill_(0)
    result = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToNumpyImage(),
        transforms.CenterCrop((oheight, owidth)),
        transforms.ToTensor(),
    ])(img)
    assert result.sum() == 0, "height: {} width: {} oheight: {} owdith: {}".format(height, width, oheight, owidth)
    oheight += 1
    owidth += 1
    result = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToNumpyImage(),
        transforms.CenterCrop((oheight, owidth)),
        transforms.ToTensor(),
    ])(img)
    sum1 = result.sum()
    assert sum1 > 1, "height: {} width: {} oheight: {} owdith: {}".format(height, width, oheight, owidth)
    oheight += 1
    owidth += 1
    result = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToNumpyImage(),
        transforms.CenterCrop((oheight, owidth)),
        transforms.ToTensor(),
    ])(img)
    sum2 = result.sum()
    assert sum2 > 0, "height: {} width: {} oheight: {} owdith: {}".format(height, width, oheight, owidth)
    assert sum2 > sum1, "height: {} width: {} oheight: {} owdith: {}".format(height, width, oheight, owidth)