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
from src.interferogram_simulation import InterferogramSimulator
from src.phase_extraction import PhaseExtractor
from src.phase_unwrapping import PhaseUnwrapper
from src.zernike_fitting import ZernikeFitter


def main():
    # -------------------------- 1. 配置参数 --------------------------
    img_size = (512, 512)
    # Zernike项参数（前16项）
    zernike_params = [
        (0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2),
        (3, 1), (3, -1), (4, 0), (3, 3), (3, -3), (4, 2),
        (4, -2), (5, 1), (5, -1), (6, 0)
    ]
    # 真实Zernike系数
    true_coeffs = np.zeros(len(zernike_params))
    true_coeffs[1] = 0.6  # X倾斜
    true_coeffs[2] = 0.4  # Y倾斜
    true_coeffs[3] = 1.2  # 离焦
    true_coeffs[4] = 0.9  # X像散
    true_coeffs[6] = 0.7  # X彗差
    true_coeffs[8] = 0.5  # 球差
    # 4步相移量
    phase_shifts = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    # -------------------------- 2. 实例化各模块 --------------------------
    # 干涉图仿真
    simulator = InterferogramSimulator(
        img_size=img_size,
        zernike_params=zernike_params,
        true_coeffs=true_coeffs,
        phase_shifts=phase_shifts,
        noise_std=0.03
    )
    interferograms, true_phase, rho, theta, circle_mask = simulator.generate()

    # 相位提取
    extractor = PhaseExtractor(interferograms, phase_shifts, circle_mask)
    wrapped_phase = extractor.extract()

    # 相位解包裹
    unwrapper = PhaseUnwrapper(wrapped_phase, circle_mask)
    unwrapped_phase = unwrapper.unwrap()

    # Zernike拟合
    fitter = ZernikeFitter(unwrapped_phase, rho, theta, zernike_params, circle_mask)
    fitted_coeffs, fitted_phase = fitter.fit()

    # -------------------------- 3. 可视化（拆分3个Figure，jet配色） --------------------------
    plt.rcParams['font.sans-serif'] = ['Arial']

    # Figure 1：干涉图
    fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4))
    fig1.suptitle('Phase-Shifted Interferograms (Circle Masked)', fontsize=14)
    for i in range(4):
        im = axes1[i].imshow(interferograms[i], cmap='gray', vmin=0, vmax=2)
        axes1[i].set_title(f'Shift = {phase_shifts[i] / np.pi:.1f}π', fontsize=10)
        axes1[i].axis('off')
        plt.colorbar(im, ax=axes1[i], shrink=0.8)
    plt.tight_layout()
    plt.show()

    # Figure 2：相位解算
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

    # Figure 3：Zernike拟合结果
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
    fig3.suptitle('Zernike Fitting Results', fontsize=14)
    # 真实相位
    im3 = axes3[0, 0].imshow(true_phase, cmap='jet')
    axes3[0, 0].set_title('True Phase (Zernike Combo)', fontsize=10)
    axes3[0, 0].axis('off')
    plt.colorbar(im3, ax=axes3[0, 0], shrink=0.8)
    # 拟合相位
    im4 = axes3[0, 1].imshow(fitted_phase, cmap='jet')
    axes3[0, 1].set_title('Fitted Phase (Zernike)', fontsize=10)
    axes3[0, 1].axis('off')
    plt.colorbar(im4, ax=axes3[0, 1], shrink=0.8)
    # 系数对比
    coeff_idx = np.arange(len(true_coeffs))
    axes3[1, 0].bar(coeff_idx - 0.2, true_coeffs, 0.4, label='True Coeffs', alpha=0.8, color='blue')
    axes3[1, 0].bar(coeff_idx + 0.2, fitted_coeffs, 0.4, label='Fitted Coeffs', alpha=0.8, color='orange')
    axes3[1, 0].set_xlabel('Zernike Term Index', fontsize=10)
    axes3[1, 0].set_ylabel('Coefficient Value', fontsize=10)
    axes3[1, 0].set_title('Zernike Coefficients Comparison', fontsize=10)
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
    print("=" * 60)
    print("Zernike拟合结果汇总")
    print("=" * 60)
    print(f"拟合项数：{len(zernike_params)}")
    print(f"系数拟合RMSE：{np.sqrt(np.mean((true_coeffs - fitted_coeffs) ** 2)):.6f}")
    print(f"相位拟合RMSE：{np.sqrt(np.nanmean((true_phase - fitted_phase) ** 2)):.6f}")
    print("\n真实系数 vs 拟合系数：")
    print("-" * 60)
    print(f"{'项号':<4} {'(n,m)':<8} {'真实系数':<12} {'拟合系数':<12} {'绝对误差':<10}")
    print("-" * 60)
    for i, ((n, m), t, f) in enumerate(zip(zernike_params, true_coeffs, fitted_coeffs)):
        print(f"{i + 1:<4} ({n},{m}){'':<6} {t:<12.6f} {f:<12.6f} {abs(t - f):<10.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()