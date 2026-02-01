# -*- coding: utf-8 -*-
"""
@Time        : 2026/2/1 15:04
@Author      : Seaton
@Email       : https://github.com/seatonyang
@Project     : lsi-algorithm
@File        : zernike_fitting.py
@Software    : PyCharm
@Description : 
"""

import numpy as np
import math
from scipy.linalg import lstsq


class ZernikeFitter:
    """Zernike拟合类：最小二乘拟合相位，仅用圆内非NaN像素"""

    def __init__(self,
                 unwrapped_phase: np.ndarray,
                 rho: np.ndarray,
                 theta: np.ndarray,
                 zernike_params: list,
                 circle_mask: np.ndarray):
        """
        初始化参数
        :param unwrapped_phase: 解包裹相位（圆外NaN）
        :param rho: 极径矩阵
        :param theta: 极角矩阵
        :param zernike_params: Zernike项(n,m)列表
        :param circle_mask: 内切圆掩码
        """
        self.unwrapped_phase = unwrapped_phase
        self.rho = rho
        self.theta = theta
        self.zernike_params = zernike_params
        self.circle_mask = circle_mask

        # 拟合结果
        self.fitted_coeffs = None
        self.fitted_phase = None

    def _zernike_radial(self, n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """内部方法：Zernike径向多项式"""
        abs_m = abs(m)
        if abs_m > n or (n - abs_m) % 2 != 0:
            return np.zeros_like(rho)

        s_max = (n - abs_m) // 2
        R = np.zeros_like(rho, dtype=np.float64)
        for s in range(s_max + 1):
            coeff = ((-1) ** s) * math.factorial(n - s) / (
                    math.factorial(s) *
                    math.factorial((n + abs_m) // 2 - s) *
                    math.factorial((n - abs_m) // 2 - s)
            )
            R += coeff * (rho ** (n - 2 * s))
        return R

    def _zernike_polynomial(self, n: int, m: int) -> np.ndarray:
        """内部方法：完整Zernike多项式"""
        abs_m = abs(m)
        R = self._zernike_radial(n, abs_m, self.rho)

        if m > 0:
            angular = np.cos(abs_m * self.theta)
        elif m < 0:
            angular = np.sin(abs_m * self.theta)
        else:
            angular = np.ones_like(self.theta)

        if m == 0:
            norm = np.sqrt(2 * n + 1)
        else:
            norm = np.sqrt(2 * (2 * n + 1))
        return norm * R * angular

    def fit(self) -> tuple:
        """最小二乘拟合（核心方法）"""
        # 筛选有效像素：圆内+非NaN
        valid_mask = self.circle_mask & (~np.isnan(self.unwrapped_phase))
        if np.sum(valid_mask) == 0:
            raise ValueError("无有效像素用于拟合！")

        # 构建Zernike基底矩阵
        N = len(self.zernike_params)
        K = np.sum(valid_mask)
        A = np.zeros((K, N), dtype=np.float64)
        for j, (n, m) in enumerate(self.zernike_params):
            Z_full = self._zernike_polynomial(n, m)
            A[:, j] = Z_full[valid_mask]

        # 最小二乘拟合
        y = self.unwrapped_phase[valid_mask]
        self.fitted_coeffs, _, _, _ = lstsq(A, y, cond=None)

        # 生成拟合相位（圆外NaN）
        self.fitted_phase = np.full_like(self.unwrapped_phase, np.nan, dtype=np.float64)
        fitted_phase_valid = np.zeros(K, dtype=np.float64)
        for j, coeff in enumerate(self.fitted_coeffs):
            fitted_phase_valid += coeff * A[:, j]
        self.fitted_phase[valid_mask] = fitted_phase_valid

        return self.fitted_coeffs, self.fitted_phase

