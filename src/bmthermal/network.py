"""集总热网络：单节点 / 核心-表面两节点，以及参数辨识。

模型与它的物理出处
------------------
单节点（DFN 的 lumped 口径）::

    C_th dT/dt = Q(t) - (T - T_amb) / R_conv,      R_conv = 1 / (h A)

两节点（本项目补的核心-表面）::

    C_core dT_c/dt = Q - (T_c - T_s) / R_int
    C_shell dT_s/dt = (T_c - T_s) / R_int - (T_s - T_amb) / R_conv

``R_int = 1 / (4 π k_r L)`` 来自圆柱均匀内热源的稳态解析解
（``ΔT = q''' R² / (4 k)`` 换算成 ``Q / (4 π k L)``），不是拍脑袋分配。

为什么值得多一个节点
--------------------
车端热电偶贴在外壳上，读到的是 ``T_s``；而老化速率由**卷芯温度** ``T_c`` 决定。
两者在高倍率下差几 K，Arrhenius 折算到寿命上是百分之几十的差别（见 aging.py）。
单节点模型在结构上无法表达这个差，只能用两节点或解析解补。

关于"两节点能不能从 DFN 辨识出来"
--------------------------------
DFN 的 lumped 热模型本身只有**一个**温度节点，因此无法从它反演 ``R_int``。
本项目的做法是三条腿互相验证：
  1. 稳态：两节点 vs DFN 的 ``Volume-averaged`` 温度，误差 < 1 K 即认为一致；
  2. 内部梯度：两节点给出的 ``T_c - T_s`` 与圆柱解析解逐点比对（同一物理，独立实现）；
  3. 辨识：在合成阶跃响应上加噪声，检验最小二乘能否把已知参数捞回来
     （闭环真值验证，与 A4 的"已知真值"思路一致）。
这三条都写进了 tests/ 与报告 §4.3。
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .params import CELL, CellSpec


@dataclass(frozen=True)
class TwoNodeCell:
    """两节点热网络参数。"""

    c_th_j_k: float
    frac_core: float
    r_int_k_w: float
    area_m2: float
    h_w_m2k: float
    pcm_extra_capacity_j_k: float = 0.0

    @property
    def c_core(self) -> float:
        return self.c_th_j_k * self.frac_core

    @property
    def c_shell(self) -> float:
        return self.c_th_j_k * (1.0 - self.frac_core) + self.pcm_extra_capacity_j_k

    @property
    def r_conv_k_w(self) -> float:
        return 1.0 / (self.h_w_m2k * self.area_m2)

    def with_h(self, h: float) -> "TwoNodeCell":
        return replace(self, h_w_m2k=float(h))


def build_cell(
    h: float,
    cell: CellSpec = CELL,
    frac_core: float = 0.85,
    k_radial: float | None = None,
    pcm_extra_capacity_j_k: float = 0.0,
) -> TwoNodeCell:
    """按电芯规格搭一个两节点网络。``frac_core=0.85`` 表示卷芯占 85% 热容。"""
    return TwoNodeCell(
        c_th_j_k=cell.thermal_capacity_j_k,
        frac_core=frac_core,
        r_int_k_w=cell.conduction_resistance_core_to_surface(k_radial),
        area_m2=cell.surface_area_m2,
        h_w_m2k=float(h),
        pcm_extra_capacity_j_k=pcm_extra_capacity_j_k,
    )


def _q_interp(t: np.ndarray, q_w: np.ndarray):
    t = np.asarray(t, dtype=float)
    q_w = np.asarray(q_w, dtype=float)
    q_w = np.broadcast_to(q_w, t.shape) if q_w.size == 1 else q_w

    def f(tt: float) -> float:
        return float(np.interp(tt, t, q_w, left=q_w[0], right=q_w[-1]))

    return f


def simulate_two_node(
    t: np.ndarray,
    q_w: np.ndarray,
    t_amb_c: float,
    cell: TwoNodeCell,
    t0_c: float | None = None,
) -> dict:
    """给定产热曲线 Q(t)，解两节点温度。

    返回 ``t``（与输入同网格）、``t_core_c``、``t_surface_c``、
    ``t_avg_c``（按热容加权，等价于 DFN 的 volume-averaged）、``t_max``。
    """
    t = np.asarray(t, dtype=float)
    qf = _q_interp(t, q_w)
    t0 = float(t_amb_c if t0_c is None else t0_c)

    def rhs(_tt, y):
        tc, ts = y
        q = qf(_tt)
        q_int = (tc - ts) / cell.r_int_k_w
        return [
            (q - q_int) / cell.c_core,
            (q_int - (ts - t_amb_c) / cell.r_conv_k_w) / cell.c_shell,
        ]

    sol = solve_ivp(
        rhs,
        (float(t[0]), float(t[-1])),
        y0=[t0, t0],
        t_eval=t,
        method="LSODA",
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"两节点热网络积分失败: {sol.message}")
    tc, ts = sol.y[0], sol.y[1]
    w = cell.c_core / (cell.c_core + cell.c_shell)
    return {
        "t": t,
        "t_core_c": tc,
        "t_surface_c": ts,
        "t_avg_c": w * tc + (1.0 - w) * ts,
        "t_max_c": float(max(tc.max(), ts.max())),
    }


def steady_state_rise(q_w: float, cell: TwoNodeCell) -> dict:
    """稳态温升分解：内阻项 + 外阻项。"""
    rise_int = q_w * cell.r_int_k_w
    rise_ext = q_w * cell.r_conv_k_w
    return {
        "rise_total_k": rise_int + rise_ext,
        "rise_internal_k": rise_int,
        "rise_external_k": rise_ext,
        "internal_share": rise_int / (rise_int + rise_ext) if (rise_int + rise_ext) > 0 else np.nan,
    }


def analytic_core_surface_gap(q_w: float, cell: CellSpec = CELL, k_radial: float | None = None) -> float:
    """圆柱均匀内热源的芯-壳温差解析解 ``Q / (4 π k L)``，K。"""
    return q_w * cell.conduction_resistance_core_to_surface(k_radial)


def identify_two_node(
    t: np.ndarray,
    q_w: np.ndarray,
    t_surface_c: np.ndarray,
    t_amb_c: float,
    h: float,
    cell: CellSpec = CELL,
    noise_k: float = 0.0,
    seed: int = 0,
) -> dict:
    """从表面温度阶跃响应反演 ``[R_int, R_conv, C_core, C_shell]``。

    参数化在对数空间（保证正定），用 ``least_squares`` 的 trf 算法加边界。
    ``R_conv`` 与 ``h`` 强相关（``R_conv = 1/(hA)``），所以默认固定 ``A`` 只拟合
    ``R_conv``，并单独报告"辨识出的 h 与设定 h 的偏差"——这个偏差才是接触热阻的
    工程含义（液冷板"标称 h"与"实际有效 h"的差距）。
    """
    t = np.asarray(t, dtype=float)
    q_w = np.asarray(q_w, dtype=float)
    y = np.asarray(t_surface_c, dtype=float)
    if noise_k > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0.0, noise_k, size=y.shape)

    area = cell.surface_area_m2
    r_int0 = cell.conduction_resistance_core_to_surface()
    c0 = cell.thermal_capacity_j_k
    p0 = np.log([r_int0, 1.0 / (h * area), 0.85 * c0, 0.15 * c0])
    lo = np.log([1e-3, 1e-3, 1e-2, 1e-2])
    hi = np.log([1e3, 1e4, 1e5, 1e5])

    def unpack(p):
        r_int, r_conv, c_core, c_shell = np.exp(p)
        return r_int, r_conv, c_core, c_shell

    def residual(p):
        r_int, r_conv, c_core, c_shell = unpack(p)

        def rhs(_tt, yy):
            tc, ts = yy
            q = float(np.interp(_tt, t, q_w))
            qi = (tc - ts) / r_int
            return [(q - qi) / c_core, (qi - (ts - t_amb_c) / r_conv) / c_shell]

        sol = solve_ivp(
            rhs, (float(t[0]), float(t[-1])), [t_amb_c, t_amb_c], t_eval=t,
            method="LSODA", rtol=1e-7, atol=1e-9,
        )
        if not sol.success:
            return np.full_like(y, 1e3)
        return sol.y[1] - y

    fit = least_squares(residual, p0, bounds=(lo, hi), method="trf", xtol=1e-12, ftol=1e-12)
    r_int, r_conv, c_core, c_shell = unpack(fit.x)
    rmse = float(np.sqrt(np.mean(fit.fun ** 2)))
    return {
        "r_int_k_w": float(r_int),
        "r_conv_k_w": float(r_conv),
        "c_core_j_k": float(c_core),
        "c_shell_j_k": float(c_shell),
        "c_th_j_k": float(c_core + c_shell),
        "h_equivalent": float(1.0 / (r_conv * area)),
        "rmse_k": rmse,
        "n_points": int(t.size),
        "success": bool(fit.success),
        "n_iter": int(fit.nfev),
    }
