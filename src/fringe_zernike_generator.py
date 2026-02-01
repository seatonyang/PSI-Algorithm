# -*- coding: utf-8 -*-
"""
@Time        : 2026/2/1 11:33
@Author      : Seaton
@Email       : https://github.com/seatonyang
@Project     : LSI_Algorithm
@File        : fringe_zernike_generator.py
@Software    : PyCharm
@Description :
                文件作用：基于Fringe索引的Zernike多项式自动生成、数学表达式打印与论文风格可视化脚本
                核心功能：
                    1.  自动生成任意阶数（自定义max_order）的Fringe Zernike多项式，无需手动编写各阶表达式
                    2.  支持打印单个/所有阶数的Zernike多项式数学表达式（径向部分+角向部分+完整形式）
                    3.  绘制论文规范的阶梯图（按s=m+k分组、右对齐，默认jet色彩映射，标记m=0项）
                    4.  支持单个多项式单独绘制，可自定义网格分辨率、色彩映射等参数
                核心特性：
                    - 严格遵循Fringe索引规则，适配光学检测、光刻等工程领域需求（区别于Noll/Standard排序）
                    - 多项式定义完全匹配论文《Straightforward path to Zernike polynomials》
                    - 完善的输入验证与错误处理，支持高分辨率网格生成，适配学术与工程仿真场景
                依赖库：numpy, matplotlib
                适用场景：光学系统像差分析、微光刻仿真、成像质量评估、学术研究中的Zernike多项式快速生成与验证
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import math
from matplotlib.patches import Patch


# ------------------------------
# 核心工具函数：自动生成Fringe索引映射和径向多项式
# ------------------------------
def generate_fringe_mapping(N):
    """
    自动生成Fringe索引与(m, k, n, 类型, 名称)的映射关系
    严格遵循论文排序规则：
    1. 按s = m+k 升序分组（行）
    2. 每行内按m从s降序到0（m最大→m=0）
    3. m>0时生成cos(mθ)（x向）和sin(mθ)（y向）两个项
    4. m=0时生成1个项（无角度依赖）
    Parameters:
        N: 最大Fringe索引（需要生成的阶数）
    Returns:
        mapping: 列表，index从0（未使用）到N，每个元素包含多项式参数
    """
    mapping = [{}]  # index 0未使用
    current_index = 1
    s = 0  # s = m + k（分组标识）

    while current_index <= N:
        # 每个s组内，m从s递减到0
        for m in range(s, -1, -1):
            k = s - m  # k = s - m（保证s = m+k）
            n = m + 2 * k  # Zernike径向阶数（n ≥ m，n和m同奇偶）

            # 自动生成多项式名称（遵循论文Table 1命名规则）
            if m == 0:
                if n == 0:
                    name = "Piston"
                elif n == 2:
                    name = "Focus"
                else:  # n ≥4 且为偶数（球差）
                    name = "Spherical aberration"
                # m=0：仅1个多项式（无角度项）
                mapping.append({
                    "index": current_index,
                    "m": m, "k": k, "n": n, "s": s,
                    "poly_type": "zero",  # 无角度依赖
                    "name": name
                })
                current_index += 1
                if current_index > N:
                    break
            else:
                # m>0：生成cos和sin两个多项式（x/y向）
                if m == 1:
                    name_cos = "Tilt x" if n == 1 else "Coma x"
                    name_sin = "Tilt y" if n == 1 else "Coma y"
                elif m == 2:
                    name_cos = "Astigmatism x"
                    name_sin = "Astigmatism y"
                elif m >= 3:
                    name_cos = f"{m}-fold x"
                    name_sin = f"{m}-fold y"
                else:
                    name_cos = f"m={m} x"
                    name_sin = f"m={m} y"

                # 添加cos(mθ)项（x向）
                mapping.append({
                    "index": current_index,
                    "m": m, "k": k, "n": n, "s": s,
                    "poly_type": "cos",
                    "name": name_cos
                })
                current_index += 1
                if current_index > N:
                    break

                # 添加sin(mθ)项（y向）
                mapping.append({
                    "index": current_index,
                    "m": m, "k": k, "n": n, "s": s,
                    "poly_type": "sin",
                    "name": name_sin
                })
                current_index += 1
                if current_index > N:
                    break
        s += 1  # 下一组s
    return mapping


def radial_polynomial(r, n, m):
    """
    计算Zernike径向多项式Rₙᵐ(r)（基于论文Eq.(1)求和公式）
    Parameters:
        r: 径向坐标（标量或2D数组，r ∈ [0,1]）
        n: 径向阶数（n ≥ m，n和m同奇偶）
        m: 角向阶数（m ≥ 0）
    Returns:
        R: 径向多项式值（与r同形状）
    """
    if n < m or (n - m) % 2 != 0:
        return np.zeros_like(r, dtype=np.float64)

    k = (n - m) // 2  # k = (n-m)/2（整数）
    R = np.zeros_like(r, dtype=np.float64)

    # 论文Eq.(1)的求和计算
    for s in range(0, k + 1):
        numerator = (-1) ** s * math.factorial(n - s)
        denominator = (math.factorial(s) *
                       math.factorial((n + m) // 2 - s) *
                       math.factorial((n - m) // 2 - s))
        term = numerator / denominator * r ** (n - 2 * s)
        R += term
    return R


def get_radial_expression(n, m):
    """
    生成径向多项式Rₙᵐ(r)的数学表达式字符串（LaTeX风格）
    Parameters:
        n: 径向阶数
        m: 角向阶数
    Returns:
        expr: 径向多项式表达式字符串
    """
    if n < m or (n - m) % 2 != 0:
        return "0"

    k = (n - m) // 2
    terms = []
    for s in range(0, k + 1):
        # 计算系数
        numerator = f"(-1)^{s} × {math.factorial(n - s)}"
        denom_part1 = math.factorial(s)
        denom_part2 = math.factorial((n + m) // 2 - s)
        denom_part3 = math.factorial((n - m) // 2 - s)
        denominator = f"{denom_part1} × {denom_part2} × {denom_part3}"

        # 幂次项
        power = n - 2 * s
        r_term = f"r^{power}" if power != 1 else "r"

        # 组合项
        if denom_part1 * denom_part2 * denom_part3 == 1:
            term_str = f"(-1)^{s} × {math.factorial(n - s)} × {r_term}"
        else:
            term_str = f"[ {numerator} / ({denominator}) ] × {r_term}"
        terms.append(term_str)

    # 组合所有项
    if len(terms) == 1:
        radial_expr = terms[0]
    else:
        radial_expr = " + ".join(terms)

    return f"R_{n}^{m}(r) = {radial_expr}"


# ------------------------------
# Zernike多项式生成与绘图类（优化版）
# ------------------------------
class FringeZernike:
    """
    基于Fringe索引的Zernike多项式自动生成与阶梯图绘制类
    特性：
    1. 支持自定义阶数（1~任意正整数，如64阶）
    2. 自动生成多项式（无需手动编写）
    3. 严格遵循论文阶梯图排布（按s=m+k分组、右对齐）
    4. 默认jet色彩映射
    5. 支持打印各阶多项式的数学表达式
    """

    def __init__(self, max_order, resolution=128):
        """
        初始化生成器
        Parameters:
            max_order: 最大Fringe索引（需要生成的阶数，如64）
            resolution: 网格分辨率（默认128x128，越高越清晰）
        """
        # 输入验证
        if not isinstance(max_order, int) or max_order < 1:
            raise ValueError(f"阶数必须是正整数，当前输入：{max_order}")

        self.max_order = max_order
        self.resolution = resolution

        # 生成极坐标/笛卡尔坐标网格
        self._create_grid()

        # 自动生成多项式定义（核心优化：无需手动写每个多项式）
        self.zernike_defs = self._auto_generate_zernike()

        # 按s=m+k分组（用于阶梯图布局）
        self.s_groups = self._group_by_s()

        # 预计算全局最大振幅（统一颜色范围，保证对比一致性）
        self.max_amplitude = self._get_global_max_amp()

        # 最大列数（用于右对齐布局：最大2s+1）
        self.max_columns = max(2 * s + 1 for s in self.s_groups.keys())

    def _create_grid(self):
        """生成极坐标（r, θ）和笛卡尔坐标（x, y）网格"""
        r = np.linspace(0, 1, self.resolution)
        theta = np.linspace(0, 2 * np.pi, self.resolution)
        self.rr, self.tt = np.meshgrid(r, theta)
        self.x = self.rr * np.cos(self.tt)
        self.y = self.rr * np.sin(self.tt)

    def _auto_generate_zernike(self):
        """自动生成所有多项式的定义（基于Fringe索引映射）"""
        fringe_mapping = generate_fringe_mapping(self.max_order)
        zernike_defs = [{}]  # index 0未使用

        for idx in range(1, self.max_order + 1):
            if idx >= len(fringe_mapping):
                break
            params = fringe_mapping[idx]

            # 动态创建多项式函数
            def create_zernike_func(m, n, poly_type):
                def func(rr, tt):
                    R = radial_polynomial(rr, n, m)  # 径向部分
                    # 角向部分（论文中的cos mθ/sin mθ）
                    if poly_type == "zero":
                        angular = np.ones_like(tt)
                    elif poly_type == "cos":
                        angular = np.cos(m * tt)
                    elif poly_type == "sin":
                        angular = np.sin(m * tt)
                    else:
                        angular = np.zeros_like(tt)
                    return R * angular  # Zernike多项式 = 径向 × 角向

                return func

            # 封装多项式信息
            zernike_func = create_zernike_func(
                params["m"], params["n"], params["poly_type"]
            )
            zernike_defs.append({
                "index": idx,
                "name": params["name"],
                "m": params["m"],  # 角向阶数
                "n": params["n"],  # 径向阶数
                "s": params["s"],  # s = m+k（分组标识）
                "poly_type": params["poly_type"],
                "func": zernike_func
            })
        return zernike_defs

    def _group_by_s(self):
        """按s=m+k分组，返回{s: [索引列表]}（用于阶梯图行布局）"""
        s_groups = {}
        for idx in range(1, self.max_order + 1):
            s = self.zernike_defs[idx]["s"]
            if s not in s_groups:
                s_groups[s] = []
            s_groups[s].append(idx)
        return dict(sorted(s_groups.items()))  # 按s升序排序

    def _get_global_max_amp(self):
        """计算所有多项式的最大绝对值（统一颜色范围）"""
        max_amp = 0.0
        for idx in range(1, self.max_order + 1):
            z = self.generate(idx)
            current_max = np.max(np.abs(z))
            if current_max > max_amp:
                max_amp = current_max
        return max_amp

    def generate(self, index):
        """
        根据Fringe索引生成Zernike多项式值
        Parameters:
            index: Fringe索引（1~self.max_order）
        Returns:
            z: 2D数组（resolution×resolution），多项式振幅分布
        """
        if not (1 <= index <= self.max_order):
            raise ValueError(f"索引必须在1~{self.max_order}之间，当前输入：{index}")
        return self.zernike_defs[index]["func"](self.rr, self.tt)

    def print_zernike_expression(self, index=None):
        """
        打印Zernike多项式的数学表达式
        Parameters:
            index: 可选，指定要打印的索引（1~self.max_order）；若为None，打印所有阶数
        """
        print("\n" + "=" * 80)
        print("Zernike多项式数学表达式（Fringe索引）")
        print("=" * 80)

        # 确定要打印的索引范围
        if index is not None:
            if not (1 <= index <= self.max_order):
                raise ValueError(f"索引必须在1~{self.max_order}之间，当前输入：{index}")
            indices = [index]
        else:
            indices = range(1, self.max_order + 1)

        for idx in indices:
            z_info = self.zernike_defs[idx]
            m = z_info["m"]
            n = z_info["n"]
            poly_type = z_info["poly_type"]

            # 生成径向部分表达式
            radial_expr = get_radial_expression(n, m)

            # 生成角向部分表达式
            if poly_type == "zero":
                angular_expr = "1"
            elif poly_type == "cos":
                angular_expr = f"cos({m}θ)" if m != 1 else "cos(θ)"
            elif poly_type == "sin":
                angular_expr = f"sin({m}θ)" if m != 1 else "sin(θ)"
            else:
                angular_expr = "0"

            # 生成完整表达式
            full_expr = f"Z_{idx}(r,θ) = {radial_expr.split('=')[1].strip()} × {angular_expr}"

            # 打印格式化信息
            print(f"\n【Fringe索引 {idx:3d}】")
            print(f"  名称: {z_info['name']:25s}")
            print(f"  参数: m={m:2d} (角向阶数), n={n:2d} (径向阶数), s={z_info['s']:2d} (m+k)")
            print(f"  径向部分: {radial_expr}")
            print(f"  角向部分: Θ(θ) = {angular_expr}")
            print(f"  完整表达式: {full_expr}")

        print("\n" + "=" * 80)

    def plot_single(self, index, figsize=(6, 5), cmap="jet"):
        """
        绘制单个Zernike多项式（默认jet色彩）
        Parameters:
            index: Fringe索引（1~self.max_order）
            figsize: 图像尺寸
            cmap: 色彩映射（默认jet）
        """
        z = self.generate(index)
        z_info = self.zernike_defs[index]

        fig, ax = plt.subplots(figsize=figsize)
        norm = Normalize(vmin=-self.max_amplitude, vmax=self.max_amplitude)

        # 绘制圆形区域的多项式分布
        contour = ax.contourf(
            self.x, self.y, z,
            levels=50, cmap=cmap, norm=norm,
            extend="both"
        )

        # 图形美化
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(
            f"Fringe Zernike #{index}\n"
            f"Name: {z_info['name']} | m={z_info['m']}, n={z_info['n']}, s={z_info['s']}",
            fontsize=12, pad=10
        )
        ax.axis("off")

        # 添加颜色条
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label("Amplitude", fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_all_stepwise(self, figsize=None, cmap="jet", title_fontsize=22):
        """
        按论文阶梯图排布绘制所有多项式（核心功能）
        布局规则：
        - 行：按s=m+k升序（s=0,1,2,...）
        - 列：每行按m降序（从s→0），右对齐（最后一列均为m=0项）
        """
        # 自动调整图大小（根据阶数动态适配）
        if figsize is None:
            rows = len(self.s_groups)
            cols = self.max_columns
            figsize = (cols * 2.2, rows * 2.2)  # 阶数高时自动扩大

        fig = plt.figure(figsize=figsize)
        norm = Normalize(vmin=-self.max_amplitude, vmax=self.max_amplitude)

        # 关键修正：GridSpec参数改为nrows/ncols（原错误：rows/cols）
        gs = gridspec.GridSpec(
            nrows=len(self.s_groups), ncols=self.max_columns,  # 修正参数名
            figure=fig, hspace=0.3, wspace=0.3
        )

        # 遍历每个s组（行）
        for row_idx, (s, indices) in enumerate(self.s_groups.items()):
            row_cols = 2 * s + 1  # 当前行的列数（2s+1）
            start_col = self.max_columns - row_cols  # 右对齐起始列

            # 遍历当前行的每个多项式（列）
            for col_offset, idx in enumerate(indices):
                col_idx = start_col + col_offset
                z = self.generate(idx)
                z_info = self.zernike_defs[idx]

                # 创建子图
                ax = fig.add_subplot(gs[row_idx, col_idx])

                # 绘制多项式
                ax.contourf(
                    self.x, self.y, z,
                    levels=30, cmap=cmap, norm=norm,
                    extend="both"
                )

                # 子图属性设置
                ax.set_xlim(-1.02, 1.02)
                ax.set_ylim(-1.02, 1.02)
                ax.set_aspect("equal")
                ax.set_title(
                    f"#{idx}\n{z_info['name']}",
                    fontsize=7 if self.max_order > 36 else 8,  # 阶数高时缩小字体
                    pad=3
                )
                ax.axis("off")

        # 全局标题和颜色条
        fig.suptitle(
            f"Fringe Zernike Polynomials (Order 1-{self.max_order})\n"
            f"Stepwise Layout (Grouped by s=m+k, Right-Aligned)",
            fontsize=title_fontsize, y=0.98
        )

        # 全局颜色条（右侧）
        cbar_ax = fig.add_axes([0.93, 0.08, 0.015, 0.82])
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax, orientation="vertical"
        )
        cbar.set_label("Normalized Amplitude", fontsize=14, labelpad=10)
        cbar.ax.tick_params(labelsize=12)

        # 保存高分辨率图片
        # filename = f"fringe_zernike_order_{self.max_order}_stepwise_jet.png"
        # plt.savefig(filename, dpi=300, bbox_inches="tight")
        # print(f"阶梯图已保存为：{filename}")
        plt.show()


# ------------------------------
# 测试代码
# ------------------------------
if __name__ == "__main__":
    # 1. 手动输入需要生成的阶数（如64）
    max_order = int(100)


    # 2. 创建生成器（分辨率可调整为256提升清晰度，耗时略增加）
    zernike_gen = FringeZernike(max_order=max_order, resolution=128)

    # 3. 打印多项式表达式（新增功能）
    print("\n📝 打印前10阶Zernike多项式表达式...")
    zernike_gen.print_zernike_expression(index=None)  # None打印所有，指定index打印单个（如index=4）

    # 4. 可选：绘制单个多项式（示例：索引1=Piston，索引4=Focus）
    print(f"\n📊 绘制单个多项式（索引1：{zernike_gen.zernike_defs[1]['name']}）...")
    zernike_gen.plot_single(index=1, cmap="jet")

    # 5. 绘制所有多项式的阶梯图（论文风格，右对齐，jet色彩）
    print(f"\n📊 绘制1-{max_order}阶阶梯图（请耐心等待，阶数越高耗时越长）...")
    zernike_gen.plot_all_stepwise(cmap="jet")

    # 6. 打印多项式的信息（验证Fringe索引正确性）
    print("\n📋 前10个多项式信息（Fringe索引顺序）：")
    for idx in range(1, max_order + 1):
        z = zernike_gen.zernike_defs[idx]
        print(f"索引{idx:2d} | 名称：{z['name']:20s} | m={z['m']:2d} | n={z['n']:2d} | s={z['s']:2d}")