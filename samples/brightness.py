# -*- coding: utf-8 -*-

"""
@date: 2021/1/13 下午8:10
@file: brightness.py
@author: zj
@description: 
"""

import imgaug.augmenters as iaa
aug = iaa.imgcorruptlike.Brightness(severity=2)