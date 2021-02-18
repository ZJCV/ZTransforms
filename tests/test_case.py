# -*- coding: utf-8 -*-

"""
@date: 2021/2/5 上午10:17
@file: test_case.py
@author: zj
@description: 
"""

import torch
import pytest


class TestCase:

    def assertLess(self, a, b, msg=None):
        assert a < b, msg

    def assertLessEqual(self, a, b, msg=None):
        assert a <= b, msg

    def assertGreater(self, a, b, msg=None):
        assert a > b, msg

    def assertEqual(self, a, b, msg=None):
        assert a == b, msg
