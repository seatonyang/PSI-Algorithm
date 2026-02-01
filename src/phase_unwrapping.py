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


# ------------------------------
# 自验证main函数（独立运行验证相位解包裹）
# ------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("PhaseUnwrapper 自验证开始")
    print("=" * 80)

    # 1. 生成测试数据（模拟包裹相位）
    np.random.seed(42)
    size = 128
    # 生成圆形掩码
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    circle_mask = (rho <= 1.0)

    # 生成模拟包裹相位（带2π跳变）
    wrapped_phase = np.arctan2(np.sin(3 * xx + 2 * yy), np.cos(3 * xx + 2 * yy))
    wrapped_phase = np.where(circle_mask, wrapped_phase, 0)  # 圆外0
    print(f"✅ 测试数据生成成功：")
    print(f"   包裹相位形状：{wrapped_phase.shape}")
    print(f"   包裹相位范围：{wrapped_phase.min():.4f} ~ {wrapped_phase.max():.4f}（应为-π~π）")

    # 2. 初始化解包裹器
    try:
        unwrapper = PhaseUnwrapper(wrapped_phase, circle_mask)
        print("✅ 解包裹器初始化成功")
    except Exception as e:
        print(f"❌ 解包裹器初始化失败：{e}")
        exit(1)

    # 3. 解包裹
    try:
        unwrapped_phase = unwrapper.unwrap()
        print(f"✅ 相位解包裹成功，形状：{unwrapped_phase.shape}")
        print(f"✅ 解包裹相位范围：{np.nanmin(unwrapped_phase):.4f} ~ {np.nanmax(unwrapped_phase):.4f}（应大于2π）")
        print(f"✅ 圆外相位NaN数：{np.sum(np.isnan(unwrapped_phase))}（应等于总像素-有效像素）")
    except Exception as e:
        print(f"❌ 解包裹失败：{e}")
        exit(1)

    # 4. 极简可视化验证
    try:
        import matplotlib.pyplot as plt

        plt.rcParams['font.sans-serif'] = ['Arial']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(wrapped_phase, cmap='jet', vmin=-np.pi, vmax=np.pi)
        ax1.set_title('Wrapped Phase', fontsize=12)
        ax1.axis('off')

        ax2.imshow(unwrapped_phase, cmap='jet')
        ax2.set_title('Unwrapped Phase', fontsize=12)
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
        print("✅ 可视化验证成功")
    except Exception as e:
        print(f"❌ 可视化验证失败：{e}")

    print("=" * 80)
    print("PhaseUnwrapper 自验证完成")
    print("=" * 80)