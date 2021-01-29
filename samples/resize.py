# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午9:38
@file: resize.py
@author: zj
@description: 
"""

import imageio
import imgaug as ia

from samples.base_sample import BaseSample
from ztransforms.transforms import Resize


class ResizeSample(BaseSample):

    def __init__(self):
        super().__init__(Resize)

    def run(self):
        for name, func in self.func_dict.items():
            print(name)
            res_list = list()
            for i in range(1, 10):
                size = i * 80
                img = imageio.imread('./assets/building.jpg')
                res_list.append(func(img, size=size))
            ia.show_grid(res_list)


if __name__ == '__main__':
    ResizeSample().run()
