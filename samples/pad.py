# -*- coding: utf-8 -*-

"""
@date: 2021/1/19 上午9:39
@file: pad.py
@author: zj
@description: 
"""

import imageio
import imgaug as ia
from ztransforms.transforms import Pad, Compose, ToPILImage, ToNumpyImage, ToTensor


def transform_pil(img, **kwargs):
    """
    :param img: ndarray
    """
    transform = Compose([
        ToPILImage(),
        Pad(**kwargs),
        ToNumpyImage(),
    ])
    return transform(img)


def transform_tensor(img, **kwargs):
    transform = Compose([
        ToTensor(),
        Pad(**kwargs),
        ToNumpyImage(),
    ])
    return transform(img)


def transform_numpy(img, **kwargs):
    transform = Pad(**kwargs)
    return transform(img)


def main():
    func_dict = {
        'transform_pil': transform_pil,
        'transform_tensor': transform_tensor,
        'transform_numpy': transform_numpy
    }
    for name, func in func_dict.items():
        print(name)
        res_list = list()
        for i in range(1, 10):
            size = i * 10
            img = imageio.imread('./assets/building.jpg')
            res_list.append(func(img, padding=size, fill=1, padding_mode="constant"))
        ia.show_grid(res_list)


if __name__ == '__main__':
    main()
