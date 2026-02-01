# -*- coding: utf-8 -*-
"""
@Time        : 2026/1/21 22:55
@Author      : Seaton
@Email       : https://github.com/seatonyang
@Project     : LSI_Algorithm
@File        : phase_extract.py
@Software    : PyCharm
@Description : 相位提取算法
"""
import numpy as np


class PhaseExtract:
    def __init__(self, imgs):
        self.imgs = imgs
        self.img_size = imgs[0].shape

