# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午9:42
@file: random_crop.py
@author: zj
@description: 
"""

import torch
import imageio
import imgaug as ia

from ztransforms.transforms import RandomCrop
from samples.base_sample import BaseSample


class RandomCropSample(BaseSample):

    def __init__(self):
        super().__init__(RandomCrop)

    def run(self):
        for name, func in self.func_dict.items():
            print(name)
            torch.manual_seed(0)
            res_list = list()
            for i in range(1, 10):
                size = i * 40
                img = imageio.imread('./assets/building.jpg')
                res_list.append(
                    func(img, size=size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"))
            ia.show_grid(res_list)


if __name__ == '__main__':
    RandomCropSample().run()
