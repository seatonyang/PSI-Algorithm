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
from scipy.linalg import lstsq
from fringe_zernike_generator import FringeZernike  # 导入用户的Zernike生成类
import matplotlib.pyplot as plt


class ZernikeFitter:
    """Zernike拟合类：最小二乘拟合相位（适配用户Fringe索引Zernike）"""

    def __init__(self,
                 unwrapped_phase: np.ndarray,
                 rho: np.ndarray,
                 theta: np.ndarray,
                 max_order: int,  # Fringe最大索引
                 circle_mask: np.ndarray):
        """
        初始化参数（适配Fringe索引）
        :param unwrapped_phase: 解包裹相位（圆外NaN）
        :param rho: 极径矩阵
        :param theta: 极角矩阵
        :param max_order: Fringe索引最大阶数
        :param circle_mask: 内切圆掩码
        """
        self.unwrapped_phase = unwrapped_phase
        self.rho = rho
        self.theta = theta
        self.max_order = max_order
        self.circle_mask = circle_mask

        # 拟合结果
        self.fitted_coeffs = None
        self.fitted_phase = None
        self.zernike_generator = None  # 用户的FringeZernike实例

    def fit(self) -> tuple:
        """最小二乘拟合（核心方法：使用用户的Zernike生成基底）"""
        # 1. 筛选有效像素：圆内+非NaN
        valid_mask = self.circle_mask & (~np.isnan(self.unwrapped_phase))
        if np.sum(valid_mask) == 0:
            raise ValueError("无有效像素用于拟合！")
        K = np.sum(valid_mask)  # 有效像素数
        print(f"✅ 拟合有效像素数 = {K}")

        # 2. 初始化用户的FringeZernike生成器（分辨率与相位矩阵一致）
        resolution = self.unwrapped_phase.shape[0]
        self.zernike_generator = FringeZernike(
            max_order=self.max_order,
            resolution=resolution
        )
        # 替换用户生成器的网格为当前极坐标（关键修复）
        self.zernike_generator.rr = self.rho  # 原始极径
        self.zernike_generator.tt = self.theta  # 原始极角
        self.zernike_generator.x = self.rho * np.cos(self.theta)
        self.zernike_generator.y = self.rho * np.sin(self.theta)

        # 3. 构建Zernike基底矩阵A（K×max_order）
        A = np.zeros((K, self.max_order), dtype=np.float64)
        for idx in range(1, self.max_order + 1):
            # 用用户的generate方法生成对应Fringe索引的Zernike多项式
            zernike_poly = self.zernike_generator.generate(idx)
            # plt.figure()
            # plt.imshow(zernike_poly, cmap="jet")
            # plt.colorbar()
            # plt.title(idx)
            # plt.show()
            # 仅提取圆形区域内的有效像素
            A[:, idx - 1] = zernike_poly[valid_mask]

        # 4. 最小二乘拟合
        y = self.unwrapped_phase[valid_mask]
        self.fitted_coeffs, _, _, _ = lstsq(A, y, cond=None)

        # 5. 生成拟合相位（仅圆形区域有效，圆外NaN）
        self.fitted_phase = np.full_like(self.unwrapped_phase, np.nan, dtype=np.float64)
        fitted_phase_valid = A @ self.fitted_coeffs  # 矩阵乘法加速
        self.fitted_phase[valid_mask] = fitted_phase_valid

        return self.fitted_coeffs, self.fitted_phase


# ------------------------------
# 自验证main函数（独立运行验证Zernike拟合）
# ------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("ZernikeFitter 自验证开始")
    print("=" * 80)

    # 1. 生成测试数据（模拟解包裹相位）
    np.random.seed(42)
    size = 512
    max_order = 64

    # 生成网格和掩码
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)
    circle_mask = (rho <= 1.0)

    # 生成真实Zernike相位（已知系数）
    true_coeffs = np.zeros(max_order)
    true_coeffs[1] = 0.6  # Z2: Tilt x
    true_coeffs[2] = 0.4  # Z3: Tilt y
    true_coeffs[3] = 1.2  # Z4: Focus
    true_coeffs[4] = 0.9  # Z5: Astigmatism x
    true_coeffs[6] = 0.7  # Z7: Coma x
    true_coeffs[8] = 0.5  # Z9: Spherical aberration

    # 用用户的FringeZernike生成真实相位
    zernike_gen = FringeZernike(max_order=max_order, resolution=size)
    zernike_gen.rr = rho  # 原始极径
    zernike_gen.tt = theta  # 原始极角
    zernike_gen.x = rho * np.cos(theta)
    zernike_gen.y = rho * np.sin(theta)
    true_phase = np.zeros((size, size))
    for idx in range(1, max_order + 1):
        z_poly = zernike_gen.generate(idx)
        true_phase += true_coeffs[idx - 1] * z_poly
    true_phase = np.where(circle_mask, true_phase, np.nan)  # 圆外NaN
    true_phase += np.random.normal(0, 0.01, true_phase.shape) * circle_mask  # 添加少量噪声

    print(f"✅ 测试数据生成成功：")
    print(f"   真实相位形状：{true_phase.shape}")
    print(f"   真实系数：{true_coeffs}")

    # 2. 初始化拟合器
    try:
        fitter = ZernikeFitter(
            unwrapped_phase=true_phase,
            rho=rho,
            theta=theta,
            max_order=max_order,
            circle_mask=circle_mask
        )
        print("✅ 拟合器初始化成功")
    except Exception as e:
        print(f"❌ 拟合器初始化失败：{e}")
        exit(1)

    # 3. 执行拟合
    try:
        fitted_coeffs, fitted_phase = fitter.fit()
        print(f"✅ 拟合成功，拟合系数：{fitted_coeffs}")

        # 计算拟合误差
        coeff_rmse = np.sqrt(np.mean((true_coeffs - fitted_coeffs) ** 2))
        phase_rmse = np.sqrt(np.nanmean((true_phase - fitted_phase) ** 2))
        print(f"✅ 系数拟合RMSE：{coeff_rmse:.6f}（应接近0）")
        print(f"✅ 相位拟合RMSE：{phase_rmse:.6f}（应接近0）")
    except Exception as e:
        print(f"❌ 拟合失败：{e}")
        exit(1)

    # 4. 极简可视化验证
    try:
        import matplotlib.pyplot as plt

        plt.rcParams['font.sans-serif'] = ['Arial']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        # 真实相位
        ax1.imshow(true_phase, cmap='jet')
        ax1.set_title('True Phase', fontsize=12)
        ax1.axis('off')
        # 拟合相位
        ax2.imshow(fitted_phase, cmap='jet')
        ax2.set_title('Fitted Phase', fontsize=12)
        ax2.axis('off')
        # 系数对比
        coeff_idx = np.arange(1, max_order + 1)
        ax3.bar(coeff_idx - 0.2, true_coeffs, 0.4, label='True', alpha=0.8)
        ax3.bar(coeff_idx + 0.2, fitted_coeffs, 0.4, label='Fitted', alpha=0.8)
        ax3.set_xlabel('Fringe Index')
        ax3.set_ylabel('Coefficient')
        ax3.set_title('Coefficient Comparison')
        ax3.legend()
        # 拟合误差
        error = np.abs(true_phase - fitted_phase)
        ax4.imshow(error, cmap='jet')
        ax4.set_title(f'Fitting Error (RMSE={phase_rmse:.4f})')
        ax4.axis('off')

        plt.tight_layout()
        plt.show()
        print("✅ 可视化验证成功")
    except Exception as e:
        print(f"❌ 可视化验证失败：{e}")

    print("=" * 80)
    print("ZernikeFitter 自验证完成")
    print("=" * 80)