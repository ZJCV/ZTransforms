# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午9:30
@file: center_crop.py
@author: zj
@description: 
"""

import imageio
import imgaug as ia
from samples.base_sample import BaseSample
from ztransforms.transforms import CenterCrop


class CenterCropSample(BaseSample):

    def __init__(self):
        super().__init__(CenterCrop)

    def run(self):
        for name, func in self.func_dict.items():
            print(name)
            res_list = list()
            for i in range(1, 10):
                size = i * 40
                img = imageio.imread('./assets/building.jpg')
                res_list.append(func(img, size=size))
            ia.show_grid(res_list)


if __name__ == '__main__':
    CenterCropSample().run()
