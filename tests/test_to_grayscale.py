# -*- coding: utf-8 -*-

"""
@date: 2021/1/23 下午2:25
@file: test_to_grayscale.py
@author: zj
@description: 
"""

import torch
import numpy as np
from PIL import Image
import random
# import torchvision.transforms as transforms
# import ztransforms.cls_transforms as transforms
import ztransforms.transforms as transforms


def test_to_grayscale():
    """Unit tests for grayscale transform"""

    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    x_pil = Image.fromarray(x_np, mode='RGB')
    x_pil_2 = x_pil.convert('L')
    gray_np = np.array(x_pil_2)

    # Test Set: Grayscale an image with desired number of output channels
    # Case 1: RGB -> 1 channel grayscale
    trans1 = transforms.Grayscale(num_output_channels=1)
    gray_pil_1 = trans1(x_pil)
    gray_np_1 = np.array(gray_pil_1)
    # self.assertEqual(gray_pil_1.mode, 'L', 'mode should be L')
    # self.assertEqual(gray_np_1.shape, tuple(x_shape[0:2]), 'should be 1 channel')
    assert gray_pil_1 == , 'mode should be L')
    self.assertEqual(gray_np_1.shape, tuple(x_shape[0:2]), 'should be 1 channel')
    np.testing.assert_equal(gray_np, gray_np_1)

    # Case 2: RGB -> 3 channel grayscale
    trans2 = transforms.Grayscale(num_output_channels=3)
    gray_pil_2 = trans2(x_pil)
    gray_np_2 = np.array(gray_pil_2)
    self.assertEqual(gray_pil_2.mode, 'RGB', 'mode should be RGB')
    self.assertEqual(gray_np_2.shape, tuple(x_shape), 'should be 3 channel')
    np.testing.assert_equal(gray_np_2[:, :, 0], gray_np_2[:, :, 1])
    np.testing.assert_equal(gray_np_2[:, :, 1], gray_np_2[:, :, 2])
    np.testing.assert_equal(gray_np, gray_np_2[:, :, 0])

    # Case 3: 1 channel grayscale -> 1 channel grayscale
    trans3 = transforms.Grayscale(num_output_channels=1)
    gray_pil_3 = trans3(x_pil_2)
    gray_np_3 = np.array(gray_pil_3)
    self.assertEqual(gray_pil_3.mode, 'L', 'mode should be L')
    self.assertEqual(gray_np_3.shape, tuple(x_shape[0:2]), 'should be 1 channel')
    np.testing.assert_equal(gray_np, gray_np_3)

    # Case 4: 1 channel grayscale -> 3 channel grayscale
    trans4 = transforms.Grayscale(num_output_channels=3)
    gray_pil_4 = trans4(x_pil_2)
    gray_np_4 = np.array(gray_pil_4)
    self.assertEqual(gray_pil_4.mode, 'RGB', 'mode should be RGB')
    self.assertEqual(gray_np_4.shape, tuple(x_shape), 'should be 3 channel')
    np.testing.assert_equal(gray_np_4[:, :, 0], gray_np_4[:, :, 1])
    np.testing.assert_equal(gray_np_4[:, :, 1], gray_np_4[:, :, 2])
    np.testing.assert_equal(gray_np, gray_np_4[:, :, 0])

    # Checking if Grayscale can be printed as string
    trans4.__repr__()
