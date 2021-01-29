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

from ztransforms.transforms import RandomHorizontalFlip
from samples.base_sample import BaseSample


class RandomHorizontalFlipSample(BaseSample):

    def __init__(self):
        super().__init__(RandomHorizontalFlip)

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
    model = RandomHorizontalFlipSample()
    model.run()
