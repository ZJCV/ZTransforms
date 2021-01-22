# -*- coding: utf-8 -*-

"""
@date: 2021/1/22 下午8:16
@file: test_color_jitter.py
@author: zj
@description: 
"""

import cv2
import numpy as np
from PIL import Image

# import torchvision.transforms as transforms
import ztransforms.transforms as transforms


def test_color_jitter():
    color_jitter = transforms.ColorJitter(2, 2, 2, 0.1)

    x_shape = [2, 2, 3]
    x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
    x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
    # x_pil = Image.fromarray(x_np, mode='RGB')
    # x_pil_2 = x_pil.convert('L')
    x_pil = x_np
    x_pil_2 = cv2.cvtColor(x_pil, cv2.COLOR_RGB2GRAY)

    for i in range(10):
        y_pil = color_jitter(x_pil)
        # self.assertEqual(y_pil.mode, x_pil.mode)
        # assert y_pil.mode == x_pil.mode
        assert y_pil.shape == x_pil.shape

        y_pil_2 = color_jitter(x_pil_2)
        # self.assertEqual(y_pil_2.mode, x_pil_2.mode)
        # assert y_pil_2.mode == x_pil_2.mode
        assert y_pil_2.shape == x_pil_2.shape

    # Checking if ColorJitter can be printed as string
    color_jitter.__repr__()
