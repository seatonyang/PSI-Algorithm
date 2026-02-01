# -*- coding: utf-8 -*-
"""
@Time        : 2026/1/21 22:56
@Author      : Seaton
@Email       : https://github.com/seatonyang
@Project     : LSI_Algorithm
@File        : phase_unwrapping.py
@Software    : PyCharm
@Description : 
"""

import numpy as np
from skimage.restoration import unwrap_phase

class PhaseUnwrapper:
    """相位解包裹类：处理包裹相位解包裹，圆外后置NaN"""
    def __init__(self, wrapped_phase: np.ndarray, circle_mask: np.ndarray):
        """
        初始化参数
        :param wrapped_phase: 包裹相位（解包裹前已圆外0）
        :param circle_mask: 内切圆掩码
        """
        self.wrapped_phase = wrapped_phase
        self.circle_mask = circle_mask
        self.unwrapped_phase = None

    def unwrap(self) -> np.ndarray:
        """解包裹（核心方法）"""
        # 质量导向解包裹
        self.unwrapped_phase = unwrap_phase(self.wrapped_phase)
        # 解包裹后：圆外相位置NaN
        self.unwrapped_phase = np.where(self.circle_mask, self.unwrapped_phase, np.nan)
        return self.unwrapped_phase


