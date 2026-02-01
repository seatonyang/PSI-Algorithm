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


# ------------------------------
# 自验证main函数（独立运行验证相位提取）
# ------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("PhaseExtractor 自验证开始")
    print("=" * 80)

    # 1. 生成测试数据（模拟4步相移干涉图）
    np.random.seed(42)  # 固定随机种子
    size = 512
    # 生成圆形掩码
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    circle_mask = (rho <= 1.0)

    # 生成模拟相位（倾斜+离焦）
    true_phase = 0.5 * xx + 1.0 * (xx ** 2 + yy ** 2)
    true_phase = np.where(circle_mask, true_phase, np.nan)

    # 生成4步相移干涉图
    phase_shifts = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    interferograms = []
    for delta in phase_shifts:
        intensity = 1.0 * (1 + 0.8 * np.cos(true_phase + delta))
        intensity = np.where(circle_mask, intensity, np.nan)
        interferograms.append(intensity)
    interferograms = np.array(interferograms)

    print(f"✅ 测试数据生成成功：")
    print(f"   干涉图形状：{interferograms.shape}")
    print(f"   圆形掩码有效像素数：{np.sum(circle_mask)}")

    # 2. 初始化提取器
    try:
        extractor = PhaseExtractor(interferograms, phase_shifts, circle_mask)
        print("✅ 提取器初始化成功")
    except Exception as e:
        print(f"❌ 提取器初始化失败：{e}")
        exit(1)

    # 3. 提取包裹相位
    try:
        wrapped_phase = extractor.extract()
        print(f"✅ 包裹相位提取成功，形状：{wrapped_phase.shape}")
        print(f"✅ 包裹相位范围：{wrapped_phase.min():.4f} ~ {wrapped_phase.max():.4f}（理论应为-π~π）")
        print(f"✅ 圆外相位值：{np.unique(wrapped_phase[~circle_mask])[0]}（应等于0）")
    except Exception as e:
        print(f"❌ 相位提取失败：{e}")
        exit(1)

    # 4. 极简可视化验证
    try:
        import matplotlib.pyplot as plt

        plt.rcParams['font.sans-serif'] = ['Arial']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(interferograms[0], cmap='jet', vmin=0, vmax=2)
        ax1.set_title('Test Interferogram 1', fontsize=12)
        ax1.axis('off')

        ax2.imshow(wrapped_phase, cmap='jet', vmin=-np.pi, vmax=np.pi)
        ax2.set_title('Extracted Wrapped Phase', fontsize=12)
        ax2.axis('off')
        plt.tight_layout()
        plt.show()
        print("✅ 可视化验证成功")
    except Exception as e:
        print(f"❌ 可视化验证失败：{e}")

    print("=" * 80)
    print("PhaseExtractor 自验证完成")
    print("=" * 80)