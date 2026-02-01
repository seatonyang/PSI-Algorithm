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
from interferogram_simulation import InterferogramSimulator
from phase_extraction import PhaseExtractor
from phase_unwrapping import PhaseUnwrapper
from zernike_fitting import ZernikeFitter


def main():
    # -------------------------- 1. 配置参数（适配Fringe索引） --------------------------
    img_size = (512, 512)
    max_order = 64  # Fringe索引最大阶数（1~16）
    # Fringe索引对应的真实系数（索引1~16）
    true_coeffs = np.zeros(max_order)
    true_coeffs[1] = 0.6  # Z2: Tilt x
    true_coeffs[2] = 0.4  # Z3: Tilt y
    true_coeffs[3] = 1.2  # Z4: Focus
    true_coeffs[4] = 0.9  # Z5: Astigmatism x
    true_coeffs[6] = 0.7  # Z7: Coma x
    true_coeffs[8] = 0.5  # Z9: Spherical aberration
    # 4步相移量
    phase_shifts = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    # -------------------------- 2. 实例化各模块 --------------------------
    # 干涉图仿真（适配Fringe索引）
    simulator = InterferogramSimulator(
        img_size=img_size,
        max_order=max_order,
        true_coeffs=true_coeffs,
        phase_shifts=phase_shifts,
        noise_std=0.03
    )
    interferograms, true_phase, rho, theta, circle_mask = simulator.generate()

    # 相位提取（无修改）
    extractor = PhaseExtractor(interferograms, phase_shifts, circle_mask)
    wrapped_phase = extractor.extract()

    # 相位解包裹（无修改）
    unwrapper = PhaseUnwrapper(wrapped_phase, circle_mask)
    unwrapped_phase = unwrapper.unwrap()

    # Zernike拟合（适配Fringe索引）
    fitter = ZernikeFitter(
        unwrapped_phase=unwrapped_phase,
        rho=rho,
        theta=theta,
        max_order=max_order,
        circle_mask=circle_mask
    )
    fitted_coeffs, fitted_phase = fitter.fit()

    # -------------------------- 3. 可视化（新增圆形掩码验证） --------------------------
    plt.rcParams['font.sans-serif'] = ['Arial']

    # ========== Figure 0：圆形掩码验证（新增） ==========
    fig0, ax0 = plt.subplots(1, 1, figsize=(6, 6))
    ax0.imshow(circle_mask, cmap='gray')
    ax0.set_title('Circle Mask (Valid Region = White)', fontsize=12)
    ax0.axis('off')
    plt.tight_layout()
    plt.show()

    # ========== Figure 1：干涉图 ==========
    fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4))
    fig1.suptitle('Phase-Shifted Interferograms (Circle Masked)', fontsize=14)
    for i in range(4):
        im = axes1[i].imshow(interferograms[i], cmap='gray', vmin=0, vmax=2)
        axes1[i].set_title(f'Shift = {phase_shifts[i] / np.pi:.1f}π', fontsize=10)
        axes1[i].axis('off')
        plt.colorbar(im, ax=axes1[i], shrink=0.8)
    plt.tight_layout()
    plt.show()

    # ========== Figure 2：相位解算 ==========
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('Phase Extraction & Unwrapping', fontsize=14)
    # 包裹相位
    im1 = axes2[0].imshow(wrapped_phase, cmap='jet', vmin=-np.pi, vmax=np.pi)
    axes2[0].set_title('Wrapped Phase (Non-circle=0)', fontsize=10)
    axes2[0].axis('off')
    plt.colorbar(im1, ax=axes2[0], shrink=0.8)
    # 解包裹相位
    im2 = axes2[1].imshow(unwrapped_phase, cmap='jet')
    axes2[1].set_title('Unwrapped Phase (Non-circle=NaN)', fontsize=10)
    axes2[1].axis('off')
    plt.colorbar(im2, ax=axes2[1], shrink=0.8)
    plt.tight_layout()
    plt.show()

    # ========== Figure 3：Zernike拟合结果 ==========
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
    fig3.suptitle('Zernike Fitting Results (Fringe Index)', fontsize=14)
    # 真实相位
    im3 = axes3[0, 0].imshow(true_phase, cmap='jet')
    axes3[0, 0].set_title('True Phase (Fringe Zernike Combo)', fontsize=10)
    axes3[0, 0].axis('off')
    plt.colorbar(im3, ax=axes3[0, 0], shrink=0.8)
    # 拟合相位
    im4 = axes3[0, 1].imshow(fitted_phase, cmap='jet')
    axes3[0, 1].set_title('Fitted Phase (Fringe Zernike)', fontsize=10)
    axes3[0, 1].axis('off')
    plt.colorbar(im4, ax=axes3[0, 1], shrink=0.8)
    # 系数对比（Fringe索引1~16）
    coeff_idx = np.arange(1, max_order + 1)
    axes3[1, 0].bar(coeff_idx - 0.2, true_coeffs, 0.4, label='True Coeffs', alpha=0.8, color='blue')
    axes3[1, 0].bar(coeff_idx + 0.2, fitted_coeffs, 0.4, label='Fitted Coeffs', alpha=0.8, color='orange')
    axes3[1, 0].set_xlabel('Fringe Zernike Index', fontsize=10)
    axes3[1, 0].set_ylabel('Coefficient Value', fontsize=10)
    axes3[1, 0].set_title('Fringe Zernike Coefficients Comparison', fontsize=10)
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

    # -------------------------- 4. 输出量化结果 --------------------------
    print("=" * 80)
    print("Zernike拟合结果汇总（Fringe索引）")
    print("=" * 80)
    print(f"拟合项数（Fringe索引）：1~{max_order}")
    print(f"系数拟合RMSE：{np.sqrt(np.mean((true_coeffs - fitted_coeffs) ** 2)):.6f}")
    print(f"相位拟合RMSE：{np.sqrt(np.nanmean((true_phase - fitted_phase) ** 2)):.6f}")
    print("\nFringe索引 | 真实系数 | 拟合系数 | 绝对误差")
    print("-" * 50)
    for idx in range(1, max_order + 1):
        t = true_coeffs[idx - 1]
        f = fitted_coeffs[idx - 1]
        print(f"{idx:10d} | {t:10.6f} | {f:10.6f} | {abs(t - f):10.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()