# -*- coding: utf-8 -*-
"""
@Time        : 2026/1/21 22:55
@Author      : Seaton
@Email       : https://github.com/seatonyang
@Project     : LSI_Algorithm
@File        : phase_extraction.py
@Software    : PyCharm
@Description : 相位提取算法
"""


import numpy as np

class PhaseExtractor:
    """相位提取类：从相移干涉图提取包裹相位"""
    def __init__(self, interferograms: np.ndarray, phase_shifts: list, circle_mask: np.ndarray):
        """
        初始化参数
        :param interferograms: 相移干涉图（M×H×W）
        :param phase_shifts: 相移量列表
        :param circle_mask: 内切圆掩码
        """
        self.interferograms = interferograms
        self.phase_shifts = phase_shifts
        self.circle_mask = circle_mask
        self.wrapped_phase = None

    def extract(self) -> np.ndarray:
        """4步相移法提取包裹相位（核心方法）"""
        if len(self.interferograms) != 4:
            raise ValueError("仅支持4步相移法（需输入4张干涉图）")

        I1, I2, I3, I4 = self.interferograms
        numerator = I4 - I2
        denominator = I1 - I3

        # 避免分母为0
        denominator = np.where(denominator == 0, 1e-8, denominator)
        self.wrapped_phase = np.arctan2(numerator, denominator)

        # 解包裹前：圆外相位置0
        self.wrapped_phase = np.where(self.circle_mask, self.wrapped_phase, 0)
        return self.wrapped_phase


