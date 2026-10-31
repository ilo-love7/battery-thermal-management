"""浸没式冷却的两相边界：沸腾曲线、CHF 与等效 h 假设的适用域。

为什么需要它
------------
v1.0–v1.1 把浸没式统一折算为"等效 h = 1000–2000 W/(m²·K)（单相）"。
这个假设的两个隐含前提：
  1. 流体不沸腾（壁温 < 饱和温度，纯单相对流）；
  2. 局部热流密度远低于临界热流密度（CHF，偏离核态沸腾）。
本模块用文献关联式把这两个前提画成可检查的边界（报告 §20）。

介质口径（**文献典型值**，代表 FC-72/Novec 类工程介电流体；
不是某一具体牌号的实测值——牌号间性质差异 ±20%，结论只用于边界量级）：
  T_sat = 56 °C，ρ_l = 1680，ρ_v = 13 kg/m³，h_fg = 88 kJ/kg，
  σ = 8.1e-3 N/m，μ_l = 4.5e-4 Pa·s，c_pl = 1100 J/(kg·K)，k_l = 0.054 W/(m·K)

关联式
------
- 核态沸腾（Rohsenow）::

      q'' = μ_l h_fg [g(ρ_l−ρ_v)/σ]^{1/2} · [c_pl ΔT_e /(C_sf h_fg Pr_l^n)]^3

  C_sf = 0.007（介电液体/铜表面，工程典型），n = 1.0（非水）。
- CHF（Kutateladze/Zuber 大平板池沸腾）::

      q''_CHF = 0.149 h_fg ρ_v [σ g (ρ_l−ρ_v) / ρ_v²]^{1/4}

- 单相对流：q'' = h_sp · (T_wall − T_fluid)，h_sp 由循环方式决定。

模块只做**边界判断**，不做 CFD；输出是"当前热流密度在哪个区、
等效 h 假设能不能用"。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G_ACC = 9.81


@dataclass(frozen=True)
class DielectricFluid:
    """介电流体饱和性质（文献典型值）。"""

    name: str = "FC-72 类介电流体（典型值）"
    t_sat_c: float = 56.0
    rho_l: float = 1680.0      # kg/m³
    rho_v: float = 13.0        # kg/m³
    h_fg: float = 88_000.0     # J/kg
    sigma: float = 8.1e-3      # N/m
    mu_l: float = 4.5e-4       # Pa·s
    c_pl: float = 1100.0       # J/(kg·K)
    k_l: float = 0.054         # W/(m·K)
    c_sf: float = 0.007        # Rohsenow 表面-流体组合系数
    n_pr: float = 1.0          # 非水取 1.0

    @property
    def pr_l(self) -> float:
        return self.mu_l * self.c_pl / self.k_l


FLUID = DielectricFluid()


def chf_zuber(f: DielectricFluid = FLUID) -> float:
    """Kutateladze/Zuber 大平板池沸腾 CHF，W/m²。"""
    term = (f.sigma * G_ACC * (f.rho_l - f.rho_v) / f.rho_v**2) ** 0.25
    return 0.149 * f.h_fg * f.rho_v * term


def nucleate_boiling_curve(
    dt_e_k: np.ndarray, f: DielectricFluid = FLUID
) -> np.ndarray:
    """Rohsenow 核态沸腾：给壁面过热度 ΔT_e = T_wall − T_sat，返回热流密度 W/m²。"""
    dt_e = np.asarray(dt_e_k, dtype=float)
    prefac = f.mu_l * f.h_fg * np.sqrt(G_ACC * (f.rho_l - f.rho_v) / f.sigma)
    rhs = f.c_pl * dt_e / (f.c_sf * f.h_fg * f.pr_l**f.n_pr)
    return prefac * rhs**3


def effective_h_boiling(q_wm2: float, f: DielectricFluid = FLUID) -> dict:
    """给定热流密度，反解核态沸腾的壁面过热度与有效 h。

    返回：所在区（onset 前 / 核态 / 超 CHF）、ΔT_e、h_eff = q''/ΔT_e。
    """
    q_chf = chf_zuber(f)
    if q_wm2 <= 0:
        return {"regime": "no_load", "dt_e_k": 0.0, "h_eff": np.nan, "q_chf": q_chf}
    if q_wm2 >= q_chf:
        return {
            "regime": "above_chf_film",
            "dt_e_k": np.nan,
            "h_eff": np.nan,
            "q_chf": q_chf,
        }
    # 反解 Rohsenow
    prefac = f.mu_l * f.h_fg * np.sqrt(G_ACC * (f.rho_l - f.rho_v) / f.sigma)
    rhs = (q_wm2 / prefac) ** (1.0 / 3.0)
    dt_e = rhs * (f.c_sf * f.h_fg * f.pr_l**f.n_pr) / f.c_pl
    return {
        "regime": "nucleate_boiling",
        "dt_e_k": float(dt_e),
        "h_eff": float(q_wm2 / dt_e),
        "q_chf": float(q_chf),
    }


def single_phase_wall_temp(q_wm2: float, h_sp: float, t_fluid_c: float) -> float:
    """单相：T_wall = T_fluid + q''/h_sp。"""
    return t_fluid_c + q_wm2 / h_sp


def immersion_validity(
    q_w: float,
    area_m2: float,
    h_sp: float = 1500.0,
    t_fluid_c: float = 25.0,
    hotspot_factor: float = 1.0,
    f: DielectricFluid = FLUID,
) -> dict:
    """等效 h 假设的检查单：单相前提 + CHF 裕度 + 热点放大。

    ``hotspot_factor``：局部热流密度 / 平均热流密度（缺陷/极耳/边缘的集中）。
    """
    q_avg = q_w / area_m2
    q_loc = q_avg * hotspot_factor
    t_wall = single_phase_wall_temp(q_avg, h_sp, t_fluid_c)
    boiling_onset = t_wall >= f.t_sat_c
    bo = effective_h_boiling(q_loc, f)
    q_chf = chf_zuber(f)
    chf_margin = q_chf / q_loc if q_loc > 0 else np.inf
    if boiling_onset or bo["regime"] == "above_chf_film":
        regime = "two_phase_required" if boiling_onset else "above_chf_film"
    else:
        regime = "single_phase_ok"
    return {
        "q_w": q_w,
        "q_avg_wm2": q_avg,
        "q_local_wm2": q_loc,
        "hotspot_factor": hotspot_factor,
        "t_wall_single_phase_c": t_wall,
        "t_sat_c": f.t_sat_c,
        "boiling_onset": bool(boiling_onset),
        "regime": regime,
        "q_chf_wm2": q_chf,
        "chf_margin": float(chf_margin),
        "boiling": bo,
    }


def hotspot_factor_to_chf(q_w: float, area_m2: float, f: DielectricFluid = FLUID) -> float:
    """平均热流下，局部放大多少倍会撞 CHF。"""
    q_avg = q_w / area_m2
    if q_avg <= 0:
        return np.inf
    return float(chf_zuber(f) / q_avg)
