# -*- coding: utf-8 -*-
"""
@Time        : 2026/2/1 14:58
@Author      : Seaton
@Email       : https://github.com/seatonyang
@Project     : lsi-algorithm
@File        : interferogram_simulation.py
@Software    : PyCharm
@Description : 
"""


import numpy as np
import math

class InterferogramSimulator:
    """干涉图仿真类：生成带圆掩码的相移干涉图"""
    def __init__(self,
                 img_size: tuple = (512, 512),
                 zernike_params: list = None,
                 true_coeffs: np.ndarray = None,
                 phase_shifts: list = None,
                 I0: float = 1.0,
                 gamma: float = 0.8,
                 noise_std: float = 0.03):
        """
        初始化仿真参数
        :param img_size: 干涉图尺寸
        :param zernike_params: Zernike项(n,m)列表
        :param true_coeffs: 真实Zernike系数
        :param phase_shifts: 相移量列表
        :param I0: 光强直流分量
        :param gamma: 调制深度
        :param noise_std: 噪声标准差
        """
        self.img_size = img_size
        self.zernike_params = zernike_params
        self.true_coeffs = true_coeffs
        self.phase_shifts = phase_shifts
        self.I0 = I0
        self.gamma = gamma
        self.noise_std = noise_std

        # 内部变量（生成后赋值）
        self.interferograms = None
        self.true_phase = None
        self.rho = None
        self.theta = None
        self.circle_mask = None

    def _zernike_radial(self, n: int, m: int, rho: np.ndarray) -> np.ndarray:
        """内部方法：计算Zernike径向多项式"""
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

    def _zernike_polynomial(self, n: int, m: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """内部方法：计算完整Zernike多项式"""
        abs_m = abs(m)
        R = self._zernike_radial(n, abs_m, rho)

        if m > 0:
            angular = np.cos(abs_m * theta)
        elif m < 0:
            angular = np.sin(abs_m * theta)
        else:
            angular = np.ones_like(theta)

        if m == 0:
            norm = np.sqrt(2 * n + 1)
        else:
            norm = np.sqrt(2 * (2 * n + 1))
        return norm * R * angular

    def generate(self) -> tuple:
        """生成相移干涉图（核心方法）"""
        H, W = self.img_size
        min_size = min(H, W)
        x = np.linspace(-1, 1, min_size, dtype=np.float64)
        y = np.linspace(-1, 1, min_size, dtype=np.float64)
        xx, yy = np.meshgrid(x, y)

        # 极坐标+圆掩码
        self.rho = np.sqrt(xx ** 2 + yy ** 2)
        self.theta = np.arctan2(yy, xx)
        self.circle_mask = (self.rho <= 1.0)

        # 生成真实相位（圆外NaN）
        self.true_phase = np.zeros_like(self.rho, dtype=np.float64)
        for coeff, (n, m) in zip(self.true_coeffs, self.zernike_params):
            self.true_phase += coeff * self._zernike_polynomial(n, m, self.rho, self.theta)
        self.true_phase = np.where(self.circle_mask, self.true_phase, np.nan)

        # 生成相移干涉图（圆外NaN）
        M = len(self.phase_shifts)
        self.interferograms = np.full((M, min_size, min_size), np.nan, dtype=np.float64)
        for i, delta in enumerate(self.phase_shifts):
            intensity = self.I0 * (1 + self.gamma * np.cos(self.true_phase + delta))
            noise = np.random.normal(0, self.noise_std, intensity.shape)
            intensity_noisy = np.clip(intensity + noise, 0, 2 * self.I0)
            self.interferograms[i][self.circle_mask] = intensity_noisy[self.circle_mask]

        return self.interferograms, self.true_phase, self.rho, self.theta, self.circle_mask

