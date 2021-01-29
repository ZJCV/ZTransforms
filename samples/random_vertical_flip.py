# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午9:44
@file: random_horizontal_flip.py
@author: zj
@description: 
"""

import torch
import imageio
import imgaug as ia

from ztransforms.transforms import RandomVerticalFlip
from samples.base_sample import BaseSample


class RandomVerticalFlipSample(BaseSample):

    def __init__(self):
        super().__init__(RandomVerticalFlip)

    def run(self):
        for name, func in self.func_dict.items():
            print(name)
            torch.manual_seed(0)
            res_list = list()
            for i in range(1, 10):
                img = imageio.imread('./assets/building.jpg')
                res_list.append(func(img, p=0.5))
            ia.show_grid(res_list)


if __name__ == '__main__':
    RandomVerticalFlipSample().run()
