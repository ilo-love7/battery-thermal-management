"""导热垫（TIM）压缩量-热阻映射：接触压力 → 等效 h。

用途
----
v1.1 §16 给了 3C 的接触设计下限（h_eff ≥ 250 W/(m²·K)）但缺
"压缩量/压力 → 热阻"的映射。本模块用**文献典型关联**把这一步接上：

    R''_contact(P) = c · P^{-n}      [m²·K/W, P 单位 Pa]
    R''_stack = R''_contact + t_pad/k_pad + R''_spread
    h_eff = area_fraction / R''_stack

文献口径（工程典型值，不是特定产品数据）：
- 接触热阻-压力：n ≈ 0.7–1.0（TIM 压缩曲线经验指数），
  R_c''(100 kPa) ≈ 15–60 mm²K/W（硅胶导热垫常见区间）；
- 体热阻：t/k，硅胶垫 k ≈ 1–3 W/(m·K)；
- 扩展热阻（圆柱端面/局部侧面 → 冷板，25% 足迹）：
  R''_spread ≈ 250 mm²K/W（量级估计，±50% 敏感性）。

模块输出**区间**而不是单值——参数是文献范围，装成单值就是伪精度。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 接触关联的文献典型区间（100 kPa 处的接触热阻，mm²·K/W）
RC100_RANGE = (15.0, 60.0)
#: 压力指数区间
N_EXP_RANGE = (0.7, 1.0)
#: 扩展热阻（mm²·K/W）中心值与不确定度
RSPREAD_MM2K_W = 250.0
RSPREAD_SIGMA = 0.5  # ±50%


@dataclass(frozen=True)
class PadSpec:
    """导热垫几何与物性。"""

    thickness_mm: float = 1.0
    k_w_mk: float = 1.5
    area_fraction: float = 0.25  # 与 cold_plate 方案一致
    contact_area_m2: float | None = None  # None → 用 CELL 足迹估算

    @property
    def r_bulk_m2k_w(self) -> float:
        return (self.thickness_mm * 1e-3) / self.k_w_mk


def contact_resistance(
    pressure_pa: float, rc100_mm2k_w: float = 30.0, n_exp: float = 0.85
) -> float:
    """R_c''(P) = R_c''(100 kPa) · (P/100 kPa)^(-n)，返回 m²·K/W。"""
    p = max(float(pressure_pa), 1.0)
    p0 = 100e3
    return (rc100_mm2k_w * 1e-6) * (p / p0) ** (-n_exp)


def stack_resistance(
    pad: PadSpec,
    pressure_pa: float,
    rc100_mm2k_w: float = 30.0,
    n_exp: float = 0.85,
    r_spread_mm2k_w: float = RSPREAD_MM2K_W,
) -> float:
    """垫片堆叠总热阻（按接触面积口径），m²·K/W。"""
    return (
        contact_resistance(pressure_pa, rc100_mm2k_w, n_exp)
        + pad.r_bulk_m2k_w
        + r_spread_mm2k_w * 1e-6
    )


def effective_h(
    pad: PadSpec,
    pressure_pa: float,
    rc100_mm2k_w: float = 30.0,
    n_exp: float = 0.85,
    r_spread_mm2k_w: float = RSPREAD_MM2K_W,
) -> float:
    """冷板口径等效 h（含面积折算）：h_eff = area_fraction / R''_stack。"""
    return pad.area_fraction / stack_resistance(
        pad, pressure_pa, rc100_mm2k_w, n_exp, r_spread_mm2k_w
    )


def pressure_for_target_h(
    pad: PadSpec,
    target_h: float,
    rc100_mm2k_w: float = 30.0,
    n_exp: float = 0.85,
    r_spread_mm2k_w: float = RSPREAD_MM2K_W,
) -> float:
    """反解达到 target_h 所需压力（Pa）；物理不可达（体热阻已超限）返回 inf。"""
    r_budget = pad.area_fraction / target_h  # 允许的总热阻
    r_fixed = pad.r_bulk_m2k_w + r_spread_mm2k_w * 1e-6
    r_contact_budget = r_budget - r_fixed
    if r_contact_budget <= 0:
        return float("inf")
    # R_c''(P) = rc100·1e-6 · (P/1e5)^(-n) = r_contact_budget
    # P = 1e5 · (rc100·1e-6 / r_contact_budget)^(1/n)
    p = 1e5 * ((rc100_mm2k_w * 1e-6) / r_contact_budget) ** (1.0 / n_exp)
    return float(p)


def force_for_pressure(
    pressure_pa: float, contact_area_m2: float
) -> float:
    """压力 → 压装力（N）。"""
    return float(pressure_pa * contact_area_m2)
