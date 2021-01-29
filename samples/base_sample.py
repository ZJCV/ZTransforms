# -*- coding: utf-8 -*-

"""
@date: 2021/1/29 下午2:35
@file: base_sample.py
@author: zj
@description: 
"""

import imageio
import imgaug as ia
from ztransforms.transforms import CenterCrop, Compose, ToPILImage, ToNumpyImage, ToTensor


class BaseSample:

    def __init__(self, func):
        self.func = func

        self.func_dict = {
            'transform_pil': self.transform_pil,
            'transform_tensor': self.transform_tensor,
            'transform_numpy': self.transform_numpy
        }

    def transform_pil(self, img, size):
        transform = Compose([
            ToPILImage(),
            self.func(size),
            ToNumpyImage(),
        ])
        return transform(img)

    def transform_tensor(self, img, size):
        transform = Compose([
            ToTensor(),
            self.func(size),
            ToNumpyImage(),
        ])
        return transform(img)

    def transform_numpy(self, img, size):
        transform = self.func(size)
        return transform(img)

    def run(self):
        pass
