"""
文件名称：fringe_zernike_auto_generate_visualization.py
文件作用：基于Fringe索引的Zernike多项式+横线剪切Zernike多项式自动生成、数学表达式打印与论文风格可视化工具
核心功能：
    1.  基础功能：生成任意阶数Fringe Zernike多项式，支持表达式打印、单多项式/阶梯图可视化
    2.  新增功能：生成横线剪切（x方向）Fringe Zernike多项式（适用于横线剪切干涉技术），包含：
        - 剪切多项式解析计算（∂Z/∂x，极坐标偏导转换）
        - 剪切多项式表达式打印（含偏导化简）
        - 剪切多项式可视化（单多项式/阶梯图，与基础版风格一致）
核心特性：
    - 严格遵循Fringe索引规则，适配光学检测、光刻等工程领域需求（区别于Noll/Standard排序）
    - 剪切多项式采用解析偏导（非数值偏导），精度更高，符合干涉技术仿真要求
    - 完善的输入验证与错误处理，支持高分辨率网格生成，适配学术与工程仿真场景
依赖库：numpy, matplotlib
适用场景：光学系统像差分析、微光刻仿真、横线剪切干涉技术、成像质量评估
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import math
from matplotlib.patches import Patch


# ------------------------------
# 核心工具函数：基础Fringe Zernike相关
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
    生成径向多项式Rₙᵐ(r)的数学表达式字符串（系数化简为具体数值）
    Parameters:
        n: 径向阶数
        m: 角向阶数
    Returns:
        expr: 径向多项式表达式字符串
    """
    if n < m or (n - m) % 2 != 0:
        return f"R_{n}^{m}(r) = 0"  # 统一格式：保留等号

    k = (n - m) // 2
    terms = []
    for s in range(0, k + 1):
        # 计算系数的具体数值（化简阶乘）
        sign = (-1) ** s
        fact_n_s = math.factorial(n - s)
        fact_s = math.factorial(s)
        fact_nm2_s = math.factorial((n + m) // 2 - s)
        fact_nm2_s2 = math.factorial((n - m) // 2 - s)

        # 计算系数值
        coefficient = sign * fact_n_s / (fact_s * fact_nm2_s * fact_nm2_s2)
        # 简化系数显示（整数显示为整数，小数保留3位）
        if coefficient.is_integer():
            coeff_str = f"{int(coefficient)}"
        else:
            coeff_str = f"{coefficient:.3f}"

        # 幂次项
        power = n - 2 * s
        if power == 0:
            r_term = "1"
        elif power == 1:
            r_term = "r"
        else:
            r_term = f"r^{power}"

        # 组合项（处理系数为1/-1的特殊情况）
        if coeff_str == "1" and power != 0:
            term_str = r_term
        elif coeff_str == "-1" and power != 0:
            term_str = f"-{r_term}"
        else:
            term_str = f"{coeff_str}×{r_term}"

        terms.append(term_str)

    # 组合所有项（处理符号，避免出现"+ -"）
    radial_expr = " + ".join(terms).replace(" + -", " - ")
    return f"R_{n}^{m}(r) = {radial_expr}"


# ------------------------------
# 剪切Zernike工具函数（横线剪切：∂Z/∂x）
# ------------------------------
def radial_derivative(r, n, m):
    """
    计算径向多项式Rₙᵐ(r)的一阶偏导∂R/∂r（解析解）
    Parameters:
        r: 径向坐标（标量或2D数组，r ∈ [0,1]）
        n: 径向阶数
        m: 角向阶数
    Returns:
        dR_dr: ∂R/∂r的值（与r同形状）
    """
    if n < m or (n - m) % 2 != 0:
        return np.zeros_like(r, dtype=np.float64)

    k = (n - m) // 2
    dR_dr = np.zeros_like(r, dtype=np.float64)

    for s in range(0, k + 1):
        numerator = (-1) ** s * math.factorial(n - s)
        denominator = (math.factorial(s) *
                       math.factorial((n + m) // 2 - s) *
                       math.factorial((n - m) // 2 - s))
        power = n - 2 * s
        if power == 0:
            term = 0  # r⁰的导数为0
        else:
            term = numerator / denominator * power * r ** (power - 1)
        dR_dr += term
    return dR_dr


def get_radial_derivative_expression(n, m):
    """
    生成径向多项式偏导∂Rₙᵐ/∂r的数学表达式（系数化简）
    【修复点1】统一返回格式：即使为0也保留"∂Rₙᵐ/∂r = "前缀，避免split索引错误
    """
    if n < m or (n - m) % 2 != 0:
        return f"∂R_{n}^{m}/∂r = 0"  # 统一格式：保留等号

    k = (n - m) // 2
    terms = []
    for s in range(0, k + 1):
        # 计算系数的具体数值
        sign = (-1) ** s
        fact_n_s = math.factorial(n - s)
        fact_s = math.factorial(s)
        fact_nm2_s = math.factorial((n + m) // 2 - s)
        fact_nm2_s2 = math.factorial((n - m) // 2 - s)

        coefficient = sign * fact_n_s / (fact_s * fact_nm2_s * fact_nm2_s2)
        power = n - 2 * s

        if power == 0:
            continue  # 导数为0，跳过

        # 导数后的系数和幂次
        deriv_coeff = coefficient * power
        deriv_power = power - 1

        # 简化系数显示
        if deriv_coeff.is_integer():
            coeff_str = f"{int(deriv_coeff)}"
        else:
            coeff_str = f"{deriv_coeff:.3f}"

        # 幂次项
        if deriv_power == 0:
            r_term = "1"
        elif deriv_power == 1:
            r_term = "r"
        else:
            r_term = f"r^{deriv_power}"

        # 组合项
        if coeff_str == "1" and deriv_power != 0:
            term_str = r_term
        elif coeff_str == "-1" and deriv_power != 0:
            term_str = f"-{r_term}"
        else:
            term_str = f"{coeff_str}×{r_term}"

        terms.append(term_str)

    if not terms:
        return f"∂R_{n}^{m}/∂r = 0"  # 无有效项时返回0（带前缀）

    # 组合所有项
    deriv_expr = " + ".join(terms).replace(" + -", " - ")
    return f"∂R_{n}^{m}/∂r = {deriv_expr}"


def shear_zernike_expression(n, m, poly_type):
    """
    生成横线剪切Zernike多项式（∂Z/∂x）的数学表达式
    极坐标转换：∂Z/∂x = cosθ·∂R/∂r - (sinθ/r)·∂Z/∂θ
    【修复点2】增加异常处理，确保split安全
    """
    # 基础径向多项式表达式（确保有等号）
    r_expr_full = get_radial_expression(n, m)
    r_expr = r_expr_full.split("=")[1].strip() if "=" in r_expr_full else "0"

    # 径向偏导表达式（确保有等号）
    dr_expr_full = get_radial_derivative_expression(n, m)
    dr_expr = dr_expr_full.split("=")[1].strip() if "=" in dr_expr_full else "0"

    # 角向偏导部分
    if poly_type == "zero":  # m=0，无角度依赖
        dZ_dθ = "0"
        shear_expr = f"cosθ × ({dr_expr})"
    elif poly_type == "cos":  # Z = R·cos(mθ)，∂Z/∂θ = -m·R·sin(mθ)
        dZ_dθ = f"-{m} × ({r_expr}) × sin({m}θ)"
        shear_expr = f"cosθ × ({dr_expr}) - (sinθ/r) × ({dZ_dθ})"
    else:  # poly_type == "sin"，Z = R·sin(mθ)，∂Z/∂θ = m·R·cos(mθ)
        dZ_dθ = f"{m} × ({r_expr}) × cos({m}θ)"
        shear_expr = f"cosθ × ({dr_expr}) - (sinθ/r) × ({dZ_dθ})"

    # 简化表达式
    shear_expr = shear_expr.replace("× 0", "0").replace(" - -", " + ")
    return f"∂Z_{n}^{m}/∂x = {shear_expr}"


# ------------------------------
# 基础Fringe Zernike类
# ------------------------------
class FringeZernike:
    """
    基于Fringe索引的Zernike多项式自动生成与阶梯图绘制类
    特性：
    1. 支持自定义阶数（1~任意正整数，如64阶）
    2. 自动生成多项式（无需手动编写）
    3. 严格遵循论文阶梯图排布（按s=m+k分组、右对齐）
    4. 默认jet色彩映射
    5. 支持打印各阶多项式的数学表达式（系数已化简）
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

        # 预计算全局最大振幅（统一颜色范围保证对比一致性）
        self.max_amplitude = self._get_global_max_amp()

        # 最大列数（用于右对齐布局：最大2s+1）
        self.max_columns = max(2 * s + 1 for s in self.s_groups.keys())

    def _create_grid(self):
        """生成极坐标（r, θ）和笛卡尔坐标（x, y）网格"""
        # 生成笛卡尔网格
        x = np.linspace(-1, 1, self.resolution)
        y = np.linspace(-1, 1, self.resolution)
        self.x, self.y = np.meshgrid(x, y)

        # 转换为极坐标
        self.r = np.sqrt(self.x ** 2 + self.y ** 2)
        self.theta = np.arctan2(self.y, self.x)

        # 超出单位圆的部分置0（Zernike仅定义在单位圆内）
        self.mask = self.r <= 1
        self.r[~self.mask] = 0

    def _auto_generate_zernike(self):
        """自动生成所有多项式的定义（基于Fringe索引映射）"""
        fringe_mapping = generate_fringe_mapping(self.max_order)
        zernike_defs = [{}]  # index 0未使用

        for idx in range(1, self.max_order + 1):
            if idx >= len(fringe_mapping):
                break
            params = fringe_mapping[idx]
            m = params["m"]
            n = params["n"]
            poly_type = params["poly_type"]

            # 定义基础Zernike多项式计算函数
            def create_zernike_func(m_val, n_val, poly_type_val):
                def func():
                    R = radial_polynomial(self.r, n_val, m_val)
                    if poly_type_val == "zero":
                        Z = R  # m=0，无角度依赖
                    elif poly_type_val == "cos":
                        Z = R * np.cos(m_val * self.theta)
                    else:  # sin
                        Z = R * np.sin(m_val * self.theta)
                    # 单位圆外置0
                    Z[~self.mask] = 0
                    return Z

                return func

            # 封装多项式信息
            zernike_func = create_zernike_func(m, n, poly_type)
            zernike_defs.append({
                "index": idx,
                "name": params["name"],
                "m": m,  # 角向阶数
                "n": n,  # 径向阶数
                "s": params["s"],  # s = m+k（分组标识）
                "poly_type": poly_type,
                "func": zernike_func  # 基础Zernike计算函数
            })
        return zernike_defs

    def _group_by_s(self):
        """按s=m+k分组，返回{s: [索引列表]}（用于阶梯图行布局）"""
        s_groups = {}
        for idx in range(1, self.max_order + 1):
            if idx >= len(self.zernike_defs):
                continue
            s = self.zernike_defs[idx]["s"]
            if s not in s_groups:
                s_groups[s] = []
            s_groups[s].append(idx)
        return dict(sorted(s_groups.items()))  # 按s升序排序

    def _get_global_max_amp(self):
        """计算所有多项式的最大绝对值（统一颜色范围）"""
        max_amp = 0.0
        for idx in range(1, self.max_order + 1):
            if idx >= len(self.zernike_defs):
                continue
            Z = self.generate(idx)
            current_max = np.max(np.abs(Z))
            if current_max > max_amp:
                max_amp = current_max
        return max_amp

    def generate(self, index):
        """
        根据Fringe索引生成基础Zernike多项式值
        Parameters:
            index: Fringe索引（1~self.max_order）
        Returns:
            Z: 2D数组（resolution×resolution），多项式振幅分布
        """
        if not (1 <= index <= self.max_order) or index >= len(self.zernike_defs):
            raise ValueError(f"索引必须在1~{self.max_order}之间，当前输入：{index}")
        return self.zernike_defs[index]["func"]()

    def print_zernike_expression(self, index=None):
        """
        打印基础Zernike多项式的数学表达式（系数已化简）
        Parameters:
            index: 可选，指定要打印的索引；若为None，打印所有阶数
        """
        print("\n" + "=" * 80)
        print("基础Fringe Zernike多项式数学表达式（系数已化简）")
        print("=" * 80)

        # 确定要打印的索引范围
        if index is not None:
            if not (1 <= index <= self.max_order) or index >= len(self.zernike_defs):
                raise ValueError(f"索引必须在1~{self.max_order}之间，当前输入：{index}")
            indices = [index]
        else:
            indices = range(1, min(self.max_order + 1, len(self.zernike_defs)))

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
            else:
                angular_expr = f"sin({m}θ)" if m != 1 else "sin(θ)"

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

    def plot_single(self, index, figsize=(6, 5), cmap="jet", title_suffix=""):
        """
        绘制单个基础Zernike多项式
        Parameters:
            index: Fringe索引
            figsize: 图像尺寸
            cmap: 色彩映射
            title_suffix: 标题后缀（用于区分剪切版）
        """
        Z = self.generate(index)
        z_info = self.zernike_defs[index]

        fig, ax = plt.subplots(figsize=figsize)
        norm = Normalize(vmin=-self.max_amplitude, vmax=self.max_amplitude)

        # 绘制圆形区域的多项式分布
        contour = ax.contourf(
            self.x, self.y, Z,
            levels=50, cmap=cmap, norm=norm,
            extend="both"
        )

        # 图形美化
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(
            f"Fringe Zernike #{index}{title_suffix}\n"
            f"Name: {z_info['name']} | m={z_info['m']}, n={z_info['n']}, s={z_info['s']}",
            fontsize=12, pad=10
        )
        ax.axis("off")

        # 添加颜色条
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label("Amplitude", fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_all_stepwise(self, figsize=None, cmap="jet", title_suffix=""):
        """
        绘制基础Zernike多项式的论文风格阶梯图
        布局规则：按s=m+k升序分行，每行右对齐
        """
        # 自动调整图大小（根据阶数动态适配）
        if figsize is None:
            rows = len(self.s_groups)
            cols = self.max_columns
            figsize = (cols * 2.2, rows * 2.2)

        fig = plt.figure(figsize=figsize)
        norm = Normalize(vmin=-self.max_amplitude, vmax=self.max_amplitude)

        # 创建网格布局
        gs = gridspec.GridSpec(
            len(self.s_groups), self.max_columns,
            figure=fig, hspace=0.3, wspace=0.3
        )

        # 遍历每个s组（行）
        for row_idx, (s, indices) in enumerate(self.s_groups.items()):
            row_cols = 2 * s + 1  # 当前行的列数
            start_col = self.max_columns - row_cols  # 右对齐起始列

            # 遍历当前行的每个多项式（列）
            for col_offset, idx in enumerate(indices):
                if idx >= len(self.zernike_defs):
                    continue
                col_idx = start_col + col_offset
                Z = self.generate(idx)
                z_info = self.zernike_defs[idx]

                # 创建子图
                ax = fig.add_subplot(gs[row_idx, col_idx])

                # 绘制多项式
                ax.contourf(
                    self.x, self.y, Z,
                    levels=30, cmap=cmap, norm=norm,
                    extend="both"
                )

                # 子图属性设置
                ax.set_xlim(-1.02, 1.02)
                ax.set_ylim(-1.02, 1.02)
                ax.set_aspect("equal")
                ax.set_title(
                    f"#{idx}\n{z_info['name'][:6]}",  # 截断名称避免重叠
                    fontsize=7 if self.max_order > 36 else 8,
                    pad=3
                )
                ax.axis("off")

        # 全局标题
        fig.suptitle(
            f"Fringe Zernike Polynomials (Order 1-{self.max_order}){title_suffix}\n"
            f"Stepwise Layout (Grouped by s=m+k, Right-Aligned)",
            fontsize=22, y=0.98
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
        # filename = f"fringe_zernike_order_{self.max_order}_stepwise_jet{title_suffix.replace(' ', '_')}.png"
        # plt.savefig(filename, dpi=300, bbox_inches="tight")
        # print(f"阶梯图已保存为：{filename}")
        plt.show()


# ------------------------------
# 新增：横线剪切Fringe Zernike类（继承基础类）
# ------------------------------
class ShearFringeZernike(FringeZernike):
    """
    横线剪切（x方向）Fringe Zernike多项式类（适用于横线剪切干涉技术）
    新增剪切率（shear_rate）参数：控制剪切量的大小，默认单位剪切率（1.0）
    """
    def __init__(self, max_order, resolution=128, shear_rate=1.0):
        # 调用父类初始化
        super().__init__(max_order, resolution)
        # 显式定义剪切率（归一化剪切量，通常取0~1，如0.1/0.2）
        self.shear_rate = shear_rate  # 剪切率参数（核心新增）
        # 预计算剪切多项式的全局最大振幅（包含剪切率）
        self.shear_max_amplitude = self._get_shear_max_amp()

    def _get_shear_max_amp(self):
        """计算剪切多项式的全局最大绝对值（统一颜色范围）"""
        max_amp = 0.0
        for idx in range(1, self.max_order + 1):
            if idx >= len(self.zernike_defs):
                continue
            shear_Z = self.generate_shear(index=idx)
            current_max = np.max(np.abs(shear_Z))
            if current_max > max_amp:
                max_amp = current_max
        return max_amp

    def generate_shear(self, index):
        """
        生成横线剪切Zernike多项式（包含剪切率）
        Z_shear = shear_rate * ∂Z/∂x
        """
        if not (1 <= index <= self.max_order) or index >= len(self.zernike_defs):
            raise ValueError(f"索引必须在1~{self.max_order}之间，当前输入：{index}")

        z_info = self.zernike_defs[index]
        m = z_info["m"]
        n = z_info["n"]
        poly_type = z_info["poly_type"]

        # 1. 计算径向多项式R和其偏导∂R/∂r
        R = radial_polynomial(self.r, n, m)
        dR_dr = radial_derivative(self.r, n, m)

        # 2. 计算角向偏导∂Z/∂θ
        if poly_type == "zero":  # m=0，无角度依赖
            dZ_dθ = np.zeros_like(self.r)
        elif poly_type == "cos":  # Z = R·cos(mθ) → ∂Z/∂θ = -m·R·sin(mθ)
            dZ_dθ = -m * R * np.sin(m * self.theta)
        else:  # Z = R·sin(mθ) → ∂Z/∂θ = m·R·cos(mθ)
            dZ_dθ = m * R * np.cos(m * self.theta)

        # 3. 计算x方向偏导∂Z/∂x（极坐标转换）
        r_safe = np.where(self.r == 0, 1e-10, self.r)
        dZ_dx = (np.cos(self.theta) * dR_dr) - (np.sin(self.theta) / r_safe) * dZ_dθ

        # 4. 引入剪切率：剪切多项式 = 剪切率 × 偏导数（核心修改）
        shear_Z = self.shear_rate * dZ_dx

        # 单位圆外置0
        shear_Z[~self.mask] = 0
        return shear_Z

    def print_shear_expression(self, index=None):
        """
        打印横线剪切Zernike多项式的数学表达式（包含剪切率）
        """
        print("\n" + "=" * 80)
        print(f"横线剪切Fringe Zernike多项式数学表达式（∂Z/∂x，剪切率={self.shear_rate}）")
        print("=" * 80)

        # 确定要打印的索引范围
        if index is not None:
            if not (1 <= index <= self.max_order) or index >= len(self.zernike_defs):
                raise ValueError(f"索引必须在1~{self.max_order}之间，当前输入：{index}")
            indices = [index]
        else:
            indices = range(1, min(self.max_order + 1, len(self.zernike_defs)))

        for idx in indices:
            z_info = self.zernike_defs[idx]
            m = z_info["m"]
            n = z_info["n"]
            poly_type = z_info["poly_type"]

            # 生成剪切多项式表达式（包含剪切率）
            base_shear_expr = shear_zernike_expression(n, m, poly_type)
            shear_expr = base_shear_expr.replace("∂Z", f"Z_{{shear}} = {self.shear_rate}·∂Z")

            # 打印格式化信息
            print(f"\n【Fringe索引 {idx:3d}】")
            print(f"  名称: {z_info['name']:25s}")
            print(f"  参数: m={m:2d} (角向阶数), n={n:2d} (径向阶数), s={z_info['s']:2d} (m+k)")
            print(f"  剪切率: {self.shear_rate}")
            print(f"  剪切表达式: {shear_expr}")

    def plot_single_shear(self, index, figsize=(6, 5), cmap="jet"):
        """
        绘制单个横线剪切Zernike多项式
        """
        shear_Z = self.generate_shear(index)
        z_info = self.zernike_defs[index]

        fig, ax = plt.subplots(figsize=figsize)
        norm = Normalize(vmin=-self.shear_max_amplitude, vmax=self.shear_max_amplitude)

        # 绘制剪切多项式分布
        contour = ax.contourf(
            self.x, self.y, shear_Z,
            levels=50, cmap=cmap, norm=norm,
            extend="both"
        )

        # 图形美化
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(
            f"Shear Fringe Zernike #{index} (∂Z/∂x)\n"
            f"Name: {z_info['name']} | m={z_info['m']}, n={z_info['n']}, s={z_info['s']}",
            fontsize=12, pad=10
        )
        ax.axis("off")

        # 添加颜色条
        cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label("Shear Amplitude (∂Z/∂x)", fontsize=10)

        plt.tight_layout()
        plt.show()

    def plot_all_stepwise_shear(self, figsize=None, cmap="jet"):
        """
        绘制横线剪切Zernike多项式的论文风格阶梯图
        """
        # 自动调整图大小
        if figsize is None:
            rows = len(self.s_groups)
            cols = self.max_columns
            figsize = (cols * 2.2, rows * 2.2)

        fig = plt.figure(figsize=figsize)
        norm = Normalize(vmin=-self.shear_max_amplitude, vmax=self.shear_max_amplitude)

        # 创建网格布局
        gs = gridspec.GridSpec(
            len(self.s_groups), self.max_columns,
            figure=fig, hspace=0.3, wspace=0.3
        )

        # 遍历每个s组（行）
        for row_idx, (s, indices) in enumerate(self.s_groups.items()):
            row_cols = 2 * s + 1
            start_col = self.max_columns - row_cols

            # 遍历当前行的每个多项式（列）
            for col_offset, idx in enumerate(indices):
                if idx >= len(self.zernike_defs):
                    continue
                col_idx = start_col + col_offset
                shear_Z = self.generate_shear(idx)
                z_info = self.zernike_defs[idx]

                # 创建子图
                ax = fig.add_subplot(gs[row_idx, col_idx])

                # 绘制剪切多项式
                ax.contourf(
                    self.x, self.y, shear_Z,
                    levels=30, cmap=cmap, norm=norm,
                    extend="both"
                )

                # 标记m=0项（红色边框）
                if z_info["m"] == 0:
                    for spine in ax.spines.values():
                        spine.set_color("red")
                        spine.set_linewidth(2)

                # 子图属性设置
                ax.set_xlim(-1.02, 1.02)
                ax.set_ylim(-1.02, 1.02)
                ax.set_aspect("equal")
                ax.set_title(
                    f"#{idx}\n{z_info['name'][:6]}",
                    fontsize=7 if self.max_order > 36 else 8,
                    pad=3
                )
                ax.axis("off")

        # 全局标题
        fig.suptitle(
            f"Shear Fringe Zernike Polynomials (Order 1-{self.max_order}) (∂Z/∂x)\n"
            f"Stepwise Layout (Grouped by s=m+k, Right-Aligned)",
            fontsize=22, y=0.98
        )

        # 全局颜色条（右侧）
        cbar_ax = fig.add_axes([0.93, 0.08, 0.015, 0.82])
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax, orientation="vertical"
        )
        cbar.set_label("Normalized Shear Amplitude (∂Z/∂x)", fontsize=14, labelpad=10)
        cbar.ax.tick_params(labelsize=12)

        # 图例
        legend_elements = [Patch(
            edgecolor="red", facecolor="none", linewidth=2,
            label="m=0 Terms (Piston/Focus/Spherical Aberration)"
        )]
        fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.92, 0.95), fontsize=12)

        # 保存高分辨率图片
        # filename = f"shear_fringe_zernike_order_{self.max_order}_stepwise_jet.png"
        # plt.savefig(filename, dpi=300, bbox_inches="tight")
        # print(f"剪切多项式阶梯图已保存为：{filename}")
        plt.show()


# ------------------------------
# 测试代码（基础+剪切Zernike）
# ------------------------------
if __name__ == "__main__":
    # 1. 输入阶数

    max_order = int(9)

    # 2. 初始化剪切Zernike生成器（自动包含基础功能）
    shear_zernike_gen = ShearFringeZernike(max_order=max_order, resolution=128, shear_rate=0.1)

    # 3. 打印基础Zernike表达式
    print("\n📝 打印基础Fringe Zernike多项式表达式...")
    shear_zernike_gen.print_zernike_expression(index=None)

    # 4. 打印剪切Zernike表达式
    print("\n📝 打印横线剪切Zernike多项式表达式...")
    shear_zernike_gen.print_shear_expression(index=None)

    # 5. 绘制单个基础Zernike（示例：索引4）
    print(f"\n📊 绘制单个基础Zernike多项式（索引4）...")
    shear_zernike_gen.plot_single(index=4, cmap="jet")

    # 6. 绘制单个剪切Zernike（示例：索引4）
    print(f"\n📊 绘制单个横线剪切Zernike多项式（索引4）...")
    shear_zernike_gen.plot_single_shear(index=4, cmap="jet")

    # 7. 绘制基础Zernike阶梯图
    print(f"\n📊 绘制基础Zernike多项式阶梯图...")
    shear_zernike_gen.plot_all_stepwise(cmap="jet")

    # 8. 绘制剪切Zernike阶梯图
    print(f"\n📊 绘制横线剪切Zernike多项式阶梯图...")
    shear_zernike_gen.plot_all_stepwise_shear(cmap="jet")

    # 9. 打印前10个多项式信息
    print("\n📋 前10个多项式信息（Fringe索引顺序）：")
    for idx in range(1, min(11, max_order + 1)):
        if idx >= len(shear_zernike_gen.zernike_defs):
            break
        z = shear_zernike_gen.zernike_defs[idx]
        print(f"索引{idx:2d} | 名称：{z['name']:20s} | m={z['m']:2d} | n={z['n']:2d} | s={z['s']:2d}")