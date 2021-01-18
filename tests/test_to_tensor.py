# -*- coding: utf-8 -*-

"""
@date: 2021/1/18 下午2:53
@file: test_to_tensor.py
@author: zj
@description: 
"""

import pytest
import torch
import numpy as np
# import torchvision.transforms as transforms
import ztransforms.cls_transforms as transforms


def test_to_tensor():
    test_channels = [1, 3, 4]
    height, width = 4, 4
    trans = transforms.ToTensor()

    # with self.assertRaises(TypeError):
    with pytest.raises(TypeError):
        trans(np.random.rand(1, height, width).tolist())

    # with self.assertRaises(ValueError):
    with pytest.raises(ValueError):
        trans(np.random.rand(height))
        trans(np.random.rand(1, 1, height, width))

    for channels in test_channels:
        input_data = torch.ByteTensor(channels, height, width).random_(0, 255).float().div_(255)
        # img = transforms.ToPILImage()(input_data)
        img = transforms.ToNumpyImage()(input_data)
        output = trans(img)
        # self.assertTrue(np.allclose(input_data.numpy(), output.numpy()))
        assert np.allclose(input_data.numpy(), output.numpy())

        ndarray = np.random.randint(low=0, high=255, size=(height, width, channels)).astype(np.uint8)
        output = trans(ndarray)
        expected_output = ndarray.transpose((2, 0, 1)) / 255.0
        # self.assertTrue(np.allclose(output.numpy(), expected_output))
        assert np.allclose(output.numpy(), expected_output)

        ndarray = np.random.rand(height, width, channels).astype(np.float32)
        output = trans(ndarray)
        expected_output = ndarray.transpose((2, 0, 1))
        # self.assertTrue(np.allclose(output.numpy(), expected_output))
        assert np.allclose(output.numpy(), expected_output)

    # separate test for mode '1' PIL images
    input_data = torch.ByteTensor(1, height, width).bernoulli_()
    # img = transforms.ToPILImage()(input_data.mul(255)).convert('1')
    img = transforms.ToNumpyImage()(input_data.mul(255)).astype(np.bool)
    output = trans(img)
    # self.assertTrue(np.allclose(input_data.numpy(), output.numpy()))
    assert np.allclose(input_data.numpy(), output.numpy())
