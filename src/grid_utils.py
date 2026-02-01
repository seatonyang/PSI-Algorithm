# -*- coding: utf-8 -*-
"""
@Time        : 2026/2/1 16:04
@Author      : Seaton
@Email       : https://github.com/seatonyang
@Project     : lsi-algorithm
@File        : grid_utils.py
@Software    : PyCharm
@Description : 
"""

import numpy as np


class GridGenerator:
    """
    统一网格生成工具类：
    - 生成笛卡尔/极坐标网格
    - 生成单位圆掩码
    - 所有模块复用同一套网格规则，避免极径处理不一致
    """

    def __init__(self, size: int):
        """
        :param size: 网格尺寸（正方形，size×size）
        """
        self.size = size
        self.x = None  # 笛卡尔x坐标（-1~1）
        self.y = None  # 笛卡尔y坐标（-1~1）
        self.rho = None  # 极径（0~√2，原始值，未修改）
        self.theta = None  # 极角（-π~π）
        self.circle_mask = None  # 单位圆掩码（rho<=1 → True）

        # 初始化时自动生成所有网格
        self._generate_grid()

    def _generate_grid(self):
        """生成笛卡尔/极坐标网格 + 单位圆掩码"""
        # 1. 生成笛卡尔坐标（-1~1，覆盖整个正方形）
        axis = np.linspace(-1, 1, self.size)
        self.x, self.y = np.meshgrid(axis, axis)

        # 2. 计算极坐标（原始值，不做任何修改）
        self.rho = np.sqrt(self.x ** 2 + self.y ** 2)  # 极径：0~√2
        self.theta = np.arctan2(self.y, self.x)  # 极角：-π~π

        # 3. 生成单位圆掩码（仅rho<=1为有效区域）
        self.circle_mask = (self.rho <= 1.0)

    def get_valid_phase(self, phase: np.ndarray) -> np.ndarray:
        """
        标准化相位：仅保留单位圆内的值，圆外置为NaN
        :param phase: 原始相位（size×size）
        :return: 标准化相位（圆内保留值，圆外NaN）
        """
        return np.where(self.circle_mask, phase, np.nan)


