# -*- coding: utf-8 -*-
"""
@Time        : 2026/2/1 15:05
@Author      : Seaton
@Email       : https://github.com/seatonyang
@Project     : lsi-algorithm
@File        : main.py
@Software    : PyCharm
@Description :
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_utils import GridGenerator
from interferogram_simulation import InterferogramSimulator
from phase_extraction import PhaseExtractor
from phase_unwrapping import PhaseUnwrapper
from zernike_fitting import ZernikeFitter


def main():
    # -------------------------- 1. 生成统一网格 --------------------------
    grid_size = 512
    grid = GridGenerator(size=grid_size)
    print(f"✅ 统一网格生成成功（size={grid_size}）")

    # -------------------------- 2. 配置参数 --------------------------
    max_order = 64
    # Fringe索引对应的真实系数
    true_coeffs = np.zeros(max_order)
    true_coeffs[1] = 0.6  # Tilt x
    true_coeffs[2] = 0.4  # Tilt y
    true_coeffs[3] = 1.2  # Focus
    true_coeffs[4] = 0.9  # Astigmatism x
    true_coeffs[6] = 0.7  # Coma x
    true_coeffs[8] = 0.5  # Spherical aberration
    # 4步相移量
    phase_shifts = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    # -------------------------- 3. 干涉图仿真 --------------------------
    simulator = InterferogramSimulator(
        grid=grid,
        max_order=max_order,
        true_coeffs=true_coeffs,
        phase_shifts=phase_shifts,
        noise_std=0.05
    )
    interferograms, true_phase = simulator.generate()

    # -------------------------- 4. 相位提取 --------------------------
    extractor = PhaseExtractor(interferograms, phase_shifts, grid.circle_mask)
    wrapped_phase = extractor.extract()

    # -------------------------- 5. 相位解包裹 --------------------------
    unwrapper = PhaseUnwrapper(wrapped_phase, grid.circle_mask)
    unwrapped_phase = unwrapper.unwrap()

    # -------------------------- 6. Zernike拟合 --------------------------
    fitter = ZernikeFitter(
        grid=grid,
        unwrapped_phase=unwrapped_phase,
        max_order=max_order
    )
    fitted_coeffs, fitted_phase = fitter.fit()

    # -------------------------- 7. 可视化 --------------------------
    plt.rcParams['font.sans-serif'] = ['Arial']

    # 7.1 圆形掩码验证
    fig0, ax0 = plt.subplots(1, 1, figsize=(6, 6))
    ax0.imshow(grid.circle_mask, cmap='gray')
    ax0.set_title('Circle Mask (Valid Region = White)', fontsize=12)
    ax0.axis('off')
    plt.tight_layout()
    plt.show()

    # 7.2 干涉图
    fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4))
    fig1.suptitle('Phase-Shifted Interferograms (Circle Masked)', fontsize=14)
    for i in range(4):
        im = axes1[i].imshow(interferograms[i], cmap='gray', vmin=0, vmax=2)
        axes1[i].set_title(f'Shift = {phase_shifts[i] / np.pi:.1f}π', fontsize=10)
        axes1[i].axis('off')
        plt.colorbar(im, ax=axes1[i], shrink=0.8)
    plt.tight_layout()
    plt.show()

    # 7.3 相位解算
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('Phase Extraction & Unwrapping', fontsize=14)
    # 包裹相位
    im1 = axes2[0].imshow(wrapped_phase, cmap='jet', vmin=-np.pi, vmax=np.pi)
    axes2[0].set_title('Wrapped Phase', fontsize=10)
    axes2[0].axis('off')
    plt.colorbar(im1, ax=axes2[0], shrink=0.8)
    # 解包裹相位
    im2 = axes2[1].imshow(unwrapped_phase, cmap='jet')
    axes2[1].set_title('Unwrapped Phase', fontsize=10)
    axes2[1].axis('off')
    plt.colorbar(im2, ax=axes2[1], shrink=0.8)
    plt.tight_layout()
    plt.show()

    # 7.4 拟合结果
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
    fig3.suptitle('Zernike Fitting Results (Fringe Index)', fontsize=14)
    # 真实相位
    im3 = axes3[0, 0].imshow(true_phase, cmap='jet')
    axes3[0, 0].set_title('True Phase', fontsize=10)
    axes3[0, 0].axis('off')
    plt.colorbar(im3, ax=axes3[0, 0], shrink=0.8)
    # 拟合相位
    im4 = axes3[0, 1].imshow(fitted_phase, cmap='jet')
    axes3[0, 1].set_title('Fitted Phase', fontsize=10)
    axes3[0, 1].axis('off')
    plt.colorbar(im4, ax=axes3[0, 1], shrink=0.8)
    # 系数对比
    coeff_idx = np.arange(1, max_order + 1)
    axes3[1, 0].bar(coeff_idx - 0.2, true_coeffs, 0.4, label='True', alpha=0.8, color='blue')
    axes3[1, 0].bar(coeff_idx + 0.2, fitted_coeffs, 0.4, label='Fitted', alpha=0.8, color='orange')
    axes3[1, 0].set_xlabel('Fringe Index', fontsize=10)
    axes3[1, 0].set_ylabel('Coefficient', fontsize=10)
    axes3[1, 0].set_title('Coefficient Comparison', fontsize=10)
    axes3[1, 0].legend()
    axes3[1, 0].grid(alpha=0.3)
    # 拟合误差
    error = np.abs(unwrapped_phase - fitted_phase)
    im5 = axes3[1, 1].imshow(error, cmap='jet')
    axes3[1, 1].set_title(f'Fitting Error (RMSE={np.sqrt(np.nanmean(error ** 2)):.4f})', fontsize=10)
    axes3[1, 1].axis('off')
    plt.colorbar(im5, ax=axes3[1, 1], shrink=0.8)
    plt.tight_layout()
    plt.show()

    # -------------------------- 8. 输出量化结果 --------------------------
    print("=" * 80)
    print("Zernike拟合结果汇总（Fringe索引）")
    print("=" * 80)
    print(f"拟合项数：1~{max_order}")
    print(f"系数拟合RMSE：{np.sqrt(np.mean((true_coeffs - fitted_coeffs) ** 2)):.6f}")
    print(f"相位拟合RMSE：{np.sqrt(np.nanmean((true_phase - fitted_phase) ** 2)):.6f}")
    print("\nFringe索引 | 真实系数 | 拟合系数 | 绝对误差")
    print("-" * 50)
    for idx in range(1, max_order + 1):
        t = true_coeffs[idx - 1]
        f = fitted_coeffs[idx - 1]
        print(f"{idx:10d} | {t:10.6f} | {f:10.6f} | {abs(t - f):10.6f}")
    print("=" * 80)


# ------------------------------
# 自验证main函数
# ------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("整体流程自验证开始")
    print("=" * 80)
    try:
        main()
        print("✅ 整体流程执行成功")
    except Exception as e:
        print(f"❌ 整体流程执行失败：{e}")
    print("=" * 80)
    print("整体流程自验证完成")
    print("=" * 80)