"""热-寿命耦合：把温度历程折算成寿命倍数。

模型
----
容量衰减的 Arrhenius 加速形式::

    k(T) = A · exp(-Ea / (R T))            [衰减速率常数, 1/s]
    D    = ∫ k(T(t)) dt                    [无量纲损伤, 累积]

同工况下两次运行的寿命比::

    life_ratio = D_ref / D_case

因为 A 在比值里约掉，**本项目只报相对寿命倍数，不报绝对循环数**。
绝对循环数需要 A_ref 的标定（特定电芯的 25 °C 寿命基准），本项目没有这个数据，
不臆造。这一点在报告 §6.1 明确声明为局限。

为什么要用"损伤积分"而不是"平均温度"
------------------------------------
Arrhenius 是指数函数，``E[exp(-Ea/RT)] ≠ exp(-Ea/R·E[T])``（Jensen 不等式）：
温度在 30↔50 °C 之间摆动，与恒温 40 °C 不等价。本项目两条都算，差值即为
"用平均温度做热管理评估"的系统性偏差（报告 §6.4）。

Ea 取值
-------
石墨/NMC 体系容量衰减的表观活化能文献常见 30–60 kJ/mol（SEI 生长主导时约 50）。
默认 45 kJ/mol，并做 30/45/60 三档敏感性——**这个参数的选取会影响结论量级，
必须一起报**。
"""
from __future__ import annotations

import numpy as np

from .params import R_GAS, T_REF_K

#: 表观活化能默认值与敏感性档位，J/mol
EA_DEFAULT = 45_000.0
EA_GRID = (30_000.0, 45_000.0, 60_000.0)


def _arrhenius_factor(t_c: np.ndarray, ea_j_mol: float = EA_DEFAULT, t_ref_c: float = 25.0) -> np.ndarray:
    """相对 25 °C 的衰减速率倍数 k(T)/k(T_ref)。"""
    t_k = np.asarray(t_c, dtype=float) + 273.15
    t_ref_k = t_ref_c + 273.15
    return np.exp(-ea_j_mol / R_GAS * (1.0 / t_k - 1.0 / t_ref_k))


def damage_integral(
    t_s: np.ndarray,
    t_cell_c: np.ndarray,
    ea_j_mol: float = EA_DEFAULT,
    t_ref_c: float = 25.0,
) -> float:
    """累积损伤 ``D = ∫ exp(-Ea/R (1/T - 1/T_ref)) dt``，单位 s（相对速率 × 秒）。

    返回值为"等效 25 °C 秒数"，物理含义：这段温度历程对寿命的消耗，
    等价于在 25 °C 下搁置多少秒。
    """
    t = np.asarray(t_s, dtype=float)
    f = _arrhenius_factor(t_cell_c, ea_j_mol, t_ref_c)
    return float(np.trapezoid(f, t))


def life_ratio(d_case: float, d_ref: float) -> float:
    """寿命倍数。>1 表示比参考工况更长寿，<1 表示更短。"""
    if d_case <= 0 or not np.isfinite(d_case):
        return np.nan
    return float(d_ref / d_case)


def equivalent_temperature(
    t_s: np.ndarray,
    t_cell_c: np.ndarray,
    ea_j_mol: float = EA_DEFAULT,
    t_ref_c: float = 25.0,
) -> float:
    """与温度历程等效的恒温温度（°C），使 ``k(T_eff)·Δt = D``。"""
    t = np.asarray(t_s, dtype=float)
    dur = float(t[-1] - t[0])
    if dur <= 0:
        return float(np.mean(t_cell_c))
    d = damage_integral(t, t_cell_c, ea_j_mol, t_ref_c)
    k_eff = d / dur
    if k_eff <= 0:
        return float(np.mean(t_cell_c))
    t_ref_k = t_ref_c + 273.15
    inv = 1.0 / t_ref_k - R_GAS / ea_j_mol * np.log(k_eff)
    return float(1.0 / inv - 273.15)


def mean_temperature_bias(
    t_s: np.ndarray,
    t_cell_c: np.ndarray,
    ea_j_mol: float = EA_DEFAULT,
) -> dict:
    """"用平均温度代替历程"会差多少。

    返回 ``t_mean_c``（时间平均温度）、``t_eff_c``（损伤等效温度）、
    ``bias_k``（两者之差）与 ``life_error_pct``（**寿命**偏差，正=用平均温度
    算出的寿命比真实历程乐观）。

    符号方向由 Jensen 不等式决定：``k(T)=exp(-Ea/RT)`` 在工作温区是凸函数，
    ``E[k(T)] ≥ k(E[T])``，所以按平均温度算的**损伤偏小**、**寿命偏乐观**，
    ``life_error_pct`` 恒为正。温度波动越大、Ea 越大，偏差越大。
    """
    t = np.asarray(t_s, dtype=float)
    t_mean = float(np.mean(t_cell_c))
    t_eff = equivalent_temperature(t, t_cell_c, ea_j_mol)
    dur = float(t[-1] - t[0])
    d_true = damage_integral(t, t_cell_c, ea_j_mol)
    d_mean = float(_arrhenius_factor(np.array([t_mean]), ea_j_mol)[0] * dur)
    # 寿命 ∝ 1/损伤：life_mean/life_true = d_true/d_mean
    life_error_pct = (d_true / d_mean - 1.0) * 100.0 if d_mean > 0 else np.nan
    return {
        "t_mean_c": t_mean,
        "t_eff_c": t_eff,
        "bias_k": t_eff - t_mean,
        "damage_underestimate_pct": (1.0 - d_mean / d_true) * 100.0 if d_true > 0 else np.nan,
        "life_error_pct": float(life_error_pct),
    }
