# -*- coding: utf-8 -*-

"""
@date: 2021/1/13 下午8:32
@file: rotate.py
@author: zj
@description: 
"""

import imgaug as ia
import imgaug.augmenters as iaa

iaa.Rotate((-45, 45))

iaa.Affine(rotate=35)