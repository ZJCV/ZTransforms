# -*- coding: utf-8 -*-

"""
@date: 2021/1/29 下午2:35
@file: base_sample.py
@author: zj
@description: 
"""

from ztransforms.transforms import Compose, ToPILImage, ToNumpyImage, ToTensor


class BaseSample:

    def __init__(self, func):
        self.func = func

        self.func_dict = {
            'transform_pil': self.transform_pil,
            'transform_tensor': self.transform_tensor,
            'transform_numpy': self.transform_numpy
        }

    def transform_pil(self, img, **kwargs):
        transform = Compose([
            ToPILImage(),
            self.func(**kwargs),
            ToNumpyImage(),
        ])
        return transform(img)

    def transform_tensor(self, img, **kwargs):
        transform = Compose([
            ToTensor(),
            self.func(**kwargs),
            ToNumpyImage(),
        ])
        return transform(img)

    def transform_numpy(self, img, **kwargs):
        transform = self.func(**kwargs)
        return transform(img)

    def run(self):
        pass
