# -*- coding: utf-8 -*-

"""
@date: 2021/2/5 上午10:03
@file: test_resize.py
@author: zj
@description: 
"""

import numpy as np
from PIL import Image

import ztransforms.transforms as transforms
from ztransforms.transforms import InterpolationMode
from tests.test_case import TestCase

NEAREST, BILINEAR, BICUBIC = InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC


class TestResize(TestCase):

    def test_resize_pil(self):
        input_sizes = [
            # height, width
            # square image
            (28, 28),
            (27, 27),
            # rectangular image: h < w
            (28, 34),
            (29, 35),
            # rectangular image: h > w
            (34, 28),
            (35, 29),
        ]
        test_output_sizes_1 = [
            # single integer
            22, 27, 28, 36,
            # single integer in tuple/list
            [22, ], (27,),
        ]
        test_output_sizes_2 = [
            # two integers
            [22, 22], [22, 28], [22, 36],
            [27, 22], [36, 22], [28, 28],
            [28, 37], [37, 27], [37, 37]
        ]

        for height, width in input_sizes:
            img = Image.new("RGB", size=(width, height), color=127)

            for osize in test_output_sizes_1:

                t = transforms.Resize(osize)
                result = t(img)

                msg = "{}, {} - {}".format(height, width, osize)
                osize = osize[0] if isinstance(osize, (list, tuple)) else osize
                # If size is an int, smaller edge of the image will be matched to this number.
                # i.e, if height > width, then image will be rescaled to (size * height / width, size).
                if height < width:
                    expected_size = (int(osize * width / height), osize)  # (w, h)
                    self.assertEqual(result.size, expected_size, msg=msg)
                elif width < height:
                    expected_size = (osize, int(osize * height / width))  # (w, h)
                    self.assertEqual(result.size, expected_size, msg=msg)
                else:
                    expected_size = (osize, osize)  # (w, h)
                    self.assertEqual(result.size, expected_size, msg=msg)

        for height, width in input_sizes:
            img = Image.new("RGB", size=(width, height), color=127)

            for osize in test_output_sizes_2:
                oheight, owidth = osize

                t = transforms.Resize(osize)
                result = t(img)

                self.assertEqual((owidth, oheight), result.size)

    def test_resize_numpy(self):
        input_sizes = [
            # height, width
            # square image
            (28, 28),
            (27, 27),
            # rectangular image: h < w
            (28, 34),
            (29, 35),
            # rectangular image: h > w
            (34, 28),
            (35, 29),
        ]
        test_output_sizes_1 = [
            # single integer
            22, 27, 28, 36,
            # single integer in tuple/list
            [22, ], (27,),
        ]
        test_output_sizes_2 = [
            # two integers
            [22, 22], [22, 28], [22, 36],
            [27, 22], [36, 22], [28, 28],
            [28, 37], [37, 27], [37, 37]
        ]

        for height, width in input_sizes:
            img = np.ones((height, width, 3)) * 127

            for osize in test_output_sizes_1:

                t = transforms.Resize(osize)
                result = t(img)

                msg = "{}, {} - {}".format(height, width, osize)
                osize = osize[0] if isinstance(osize, (list, tuple)) else osize
                # If size is an int, smaller edge of the image will be matched to this number.
                # i.e, if height > width, then image will be rescaled to (size * height / width, size).
                if height < width:
                    expected_size = (osize, int(osize * height / width))  # (h, w)
                    self.assertEqual(result.shape[:2], expected_size, msg=msg)
                elif width < height:
                    expected_size = (int(osize * width / height), osize)  # (h ,w)
                    self.assertEqual(result.shape[:2], expected_size, msg=msg)
                else:
                    expected_size = (osize, osize)  # (h, w)
                    self.assertEqual(result.shape[:2], expected_size, msg=msg)

        for height, width in input_sizes:
            img = np.ones((height, width, 3)) * 127

            for osize in test_output_sizes_2:
                oheight, owidth = osize

                t = transforms.Resize(osize)
                result = t(img)

                self.assertEqual((oheight, owidth), result.shape[:2])
