# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午9:39
@file: pad.py
@author: zj
@description: 
"""

import imageio
import imgaug as ia
from samples.base_sample import BaseSample
from ztransforms.transforms import Pad


class PadSample(BaseSample):

    def __init__(self):
        super().__init__(Pad)

    def run(self):
        for name, func in self.func_dict.items():
            print(name)
            res_list = list()
            for i in range(1, 10):
                size = i * 10
                img = imageio.imread('./assets/building.jpg')
                res_list.append(func(img, padding=size, fill=1, padding_mode="constant"))
            ia.show_grid(res_list)


if __name__ == '__main__':
    PadSample().run()
