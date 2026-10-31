"""模组级热网络：电芯阵列 + 冷却液沿程温升。

为什么模组必须单独建模（而不能把单芯结果乘以 12）
------------------------------------------------
单芯分析回答"这颗电芯会到多少度"，模组要回答的是**一致性**：最热与最冷的电芯差多少。
温差有三个来源，单芯模型一个都装不下：

1. **冷却液沿流动方向升温**。冷板入口 25 °C、出口可能 30 °C，下游电芯天然更热。
   这一项的量级由流量决定（``ΔT_coolant = Q_total / (ṁ c_p)``），与 h 无关——
   这是"加大 h 也解决不了"的那部分温差。
2. **有效冷却面积**。圆柱电芯贴底部冷板，真正参与换热的只有端面 + 局部侧面，
   而不是全表面。同样的 h，作用在 25% 还是 100% 面积上，差 4 倍。
3. **相邻电芯互加热**。热失控传播研究关心这个（B3），常规工况下它把中间电芯抬高 1–2 K。

模型
----
每个电芯一个集总节点::

    C_i dT_i/dt = Q_i(t) - (T_i - T_cool,j) · h · A_eff - Σ_k g_ik (T_i - T_k)

冷却液按"沿程离散柱"处理，且**准稳态**（冷却液热容小、停留时间秒级，远快于电芯
的百秒级热时间常数）::

    T_cool,1 = T_in
    T_cool,j+1 = T_cool,j + Σ_{i∈column j} (T_i - T_cool,j) · h · A_eff / (ṁ c_p)

准稳态的代价：低估了流量突变后的瞬态（本项目不涉及流量阶跃），
在恒定流量工况下与全动态解的差别可忽略（报告 §5.2 有对照）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from .params import CELL, CellSpec

#: 水-乙二醇冷却液比热，J/(kg·K)〔工程经验值，50/50 体积比约 3300〕
CP_COOLANT = 3300.0
#: 冷却液密度，kg/m³
RHO_COOLANT = 1050.0


@dataclass(frozen=True)
class ModuleSpec:
    """模组几何与热边界。"""

    n_rows: int = 2          # 垂直于流向的电芯排数
    n_cols: int = 6          # 沿流向的列数
    pitch_m: float = 0.022   # 电芯中心距（21700 直径 21 mm + 1 mm 间隙）
    h_w_m2k: float = 500.0   # 电芯表面换热系数
    area_fraction: float = 0.25  # 参与换热的表面积占比（底部冷板只覆盖端面/局部）
    flow_lpm: float = 2.0    # 冷却液体积流量 L/min
    g_neighbor_w_k: float = 0.5  # 相邻电芯间等效热导（经间隙/灌封/支架）
    t_in_c: float = 25.0     # 冷却液入口温度
    pcm_extra_capacity_j_k: float = 0.0
    cell: CellSpec = CELL

    @property
    def n_cells(self) -> int:
        return self.n_rows * self.n_cols

    @property
    def m_dot_kg_s(self) -> float:
        return self.flow_lpm / 60.0 / 1000.0 * RHO_COOLANT

    @property
    def a_eff_m2(self) -> float:
        return self.cell.surface_area_m2 * self.area_fraction

    def coolant_rise_k(self, total_heat_w: float) -> float:
        """冷却液总温升，K。这是"h 再大也压不下去"的那部分温差。"""
        m_dot = self.m_dot_kg_s
        return total_heat_w / (m_dot * CP_COOLANT) if m_dot > 0 else np.inf


def _cell_index(spec: ModuleSpec) -> list[tuple[int, int]]:
    return [(r, c) for r in range(spec.n_rows) for c in range(spec.n_cols)]


def simulate_module(
    t: np.ndarray,
    q_w: np.ndarray,
    spec: ModuleSpec,
    t0_c: float | None = None,
    q_scale: float = 1.0,
) -> dict:
    """给定单芯产热曲线（所有电芯同步），解模组温度场。

    ``q_scale`` 用于产热不确定性稳健性检查（见 heat.py docstring 里 1.9–4.4 倍的参数分歧）。
    """
    t = np.asarray(t, dtype=float)
    q_w = np.asarray(q_w, dtype=float) * q_scale
    t0 = float(spec.t_in_c if t0_c is None else t0_c)
    idx = _cell_index(spec)
    pos = {rc: i for i, rc in enumerate(idx)}
    n = len(idx)

    c_th = spec.cell.thermal_capacity_j_k + spec.pcm_extra_capacity_j_k
    hA = spec.h_w_m2k * spec.a_eff_m2
    mcp = spec.m_dot_kg_s * CP_COOLANT

    # 邻接矩阵（上下左右）
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for (r, c), i in pos.items():
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            j = pos.get((r + dr, c + dc))
            if j is not None:
                neighbors[i].append((j, spec.g_neighbor_w_k))

    def coolant_temps(t_cells: np.ndarray) -> np.ndarray:
        """沿流向逐列推进，返回每个电芯所在列的冷却液温度。"""
        t_cols = np.empty(spec.n_cols)
        t_cool = spec.t_in_c
        for c in range(spec.n_cols):
            t_cols[c] = t_cool
            heat = 0.0
            for r in range(spec.n_rows):
                i = pos[(r, c)]
                heat += (t_cells[i] - t_cool) * hA
            if mcp > 0:
                t_cool = t_cool + heat / mcp
        out = np.empty(n)
        for (r, c), i in pos.items():
            out[i] = t_cols[c]
        return out

    def rhs(tt, y):
        q = float(np.interp(tt, t, q_w))
        t_cool = coolant_temps(y)
        d = np.zeros(n)
        for i in range(n):
            cond = sum(g * (y[i] - y[j]) for j, g in neighbors[i])
            d[i] = (q - (y[i] - t_cool[i]) * hA - cond) / c_th
        return d

    sol = solve_ivp(
        rhs, (float(t[0]), float(t[-1])), np.full(n, t0),
        t_eval=t, method="LSODA", rtol=1e-7, atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"模组热网络积分失败: {sol.message}")

    temps = sol.y  # (n, n_t)
    t_peak = temps.max(axis=1)
    dt_max = float(np.max(temps.max(axis=0) - temps.min(axis=0)))
    t_cool_end = coolant_temps(temps[:, -1])
    return {
        "t": t,
        "temps_c": temps,
        "index": idx,
        "t_max_c": float(t_peak.max()),
        "t_min_c": float(t_peak.min()),
        "delta_t_max_k": dt_max,
        "t_cool_in_c": spec.t_in_c,
        "t_cool_out_c": float(t_cool_end.max()),
        "coolant_rise_k": float(t_cool_end.max() - spec.t_in_c),
        "n_cells": n,
    }


def consistency_report(res: dict, limit_k: float = 5.0) -> dict:
    """一致性判据：ΔT ≤ 5 K（模组设计常用口径）。"""
    return {
        "delta_t_max_k": res["delta_t_max_k"],
        "limit_k": limit_k,
        "pass": bool(res["delta_t_max_k"] <= limit_k),
        "t_max_c": res["t_max_c"],
        "coolant_rise_k": res["coolant_rise_k"],
    }


def required_flow_lpm(
    total_heat_w: float,
    spec: ModuleSpec,
    limit_k: float = 5.0,
    share: float = 0.5,
) -> float:
    """把 ΔT 预算的一半留给冷却液温升时，所需的最小流量 L/min。

    ``share``：允许冷却液温升占总一致性预算的比例。默认 0.5，即 5 K 预算里
    分 2.5 K 给"沿程升温"，其余留给单芯散热差异与接触不均。
    """
    if total_heat_w <= 0:
        return 0.0
    allowed_k = limit_k * share
    return float(total_heat_w / (allowed_k * CP_COOLANT) * 60.0 / RHO_COOLANT * 1000.0)
