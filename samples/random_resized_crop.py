# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午9:46
@file: random_resized_crop.py
@author: zj
@description: 
"""

import torch
import imageio
import imgaug as ia

from ztransforms.transforms import RandomResizedCrop
from samples.base_sample import BaseSample


class RandomResizedCropSample(BaseSample):

    def __init__(self):
        super().__init__(RandomResizedCrop)

    def run(self):
        for name, func in self.func_dict.items():
            print(name)
            torch.manual_seed(0)
            res_list = list()
            for i in range(1, 10):
                size = 20 * i + 50
                img = imageio.imread('./assets/building.jpg')
                res_list.append(func(img, size=size))
            ia.show_grid(res_list)


if __name__ == '__main__':
    RandomResizedCropSample().run()
