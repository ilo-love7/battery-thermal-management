"""冷却方案扫描、选型 map 与 PCM 缓冲。

两条计算路径，故意都留着
------------------------
**路径 A（DFN 耦合，贵但可信）**：PyBaMM 自己解电化学-热耦合，``h`` 写进
``Total heat transfer coefficient``。只在 ≤2C 使用——3C 起 Chen2020 的 DFN 本身失效
（交付容量掉到 38%），再谈"温升"没有意义。它的作用是给出**锚点**，标定快模型。

**路径 B（两节点快模型，便宜）**：Q(t) 来自 ``heat.q_profile``，喂给
``network.simulate_two_node``。单次 < 10 ms，可以铺满 方案 × 倍率 × 环境温度 × 产热缩放。
适用域是**实测出来的**（报告 §4.3）：2C 以内与路径 A 差 < 2 K；3C 起快模型低估
自热反馈，误差写进报告，并在模组/寿命分析里只用于**相对比较**。

"需要多大 h"可以解析算
----------------------
稳态 ``ΔT = Q (R_int + 1/(h A))``，反解::

    h_req = 1 / (A (ΔT_allowed / Q - R_int))

它的价值在于说明 **h_req ∝ Q ∝ C^1.8**：倍率翻倍，所需换热系数涨 3.5 倍。
"3C 靠风冷行不行"这句话，用这条式子就能当场判死。
注意分母可能为负——那意味着 ``Q·R_int`` 已经吃掉全部温升预算，
**再大的 h 也救不回来**，此时返回 inf 并单独标注。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .heat import C_RATE_TRUSTED, q_profile, run_dfn_heat
from .network import build_cell, simulate_two_node, steady_state_rise
from .params import CELL, SCHEMES, CellSpec, CoolingScheme

#: 设计口径：电芯最高温度上限（°C）
T_LIMIT_C = 45.0
#: 预警口径（°C）
T_WARN_C = 60.0
#: 超过此温度，集总模型（不含副反应产热）不再适用，结论应交给 B3 的热失控模型
T_MODEL_MAX_C = 100.0


def _time_to(series: np.ndarray, t: np.ndarray, threshold_c: float) -> float:
    """首次达到 threshold 的时间（s）；未达到返回 inf。"""
    hit = np.flatnonzero(np.asarray(series) >= threshold_c)
    return float(t[hit[0]]) if hit.size else float("inf")


def cell_steady_rise(q_w: float, h: float, cell: CellSpec = CELL, k_radial: float | None = None) -> dict:
    """单芯稳态温升及其内外分解。"""
    net = build_cell(h=h, cell=cell, k_radial=k_radial)
    r = steady_state_rise(q_w, net)
    r.update({"h": h, "q_w": q_w})
    return r


def required_h(
    q_w: float,
    allowed_rise_k: float,
    cell: CellSpec = CELL,
    k_radial: float | None = None,
) -> float:
    """满足 ``ΔT ≤ allowed_rise_k`` 所需的换热系数 W/(m²·K)。"""
    net0 = build_cell(h=1.0, cell=cell, k_radial=k_radial)
    denom = allowed_rise_k / q_w - net0.r_int_k_w
    if denom <= 0:
        return float("inf")
    return float(1.0 / (denom * cell.surface_area_m2))


def max_rise_at_h(q_w: float, h: float, cell: CellSpec = CELL) -> float:
    """给定 h 能达到的最小稳态温升（h→∞ 时的极限由 R_int 决定）。"""
    net = build_cell(h=h, cell=cell)
    return steady_state_rise(q_w, net)["rise_total_k"]


def sweep_schemes_dfn(
    rates=(0.5, 1.0, 1.5, 2.0),
    scheme_keys=("natural", "forced_air", "cold_plate", "immersion"),
    t_amb_c: float = 25.0,
    v_min: float = 3.0,
) -> pd.DataFrame:
    """路径 A：DFN 耦合锚点。只跑 ≤2C。"""
    rows = []
    for key in scheme_keys:
        scheme = SCHEMES[key]
        for c in rates:
            if c > C_RATE_TRUSTED + 1e-9:
                continue
            res = run_dfn_heat(c, t_amb_c=t_amb_c, v_min=v_min, h=scheme.h_mid, isothermal=False)
            rows.append({
                "scheme": key,
                "scheme_name": scheme.name,
                "h": scheme.h_mid,
                "c_rate": c,
                "t_amb_c": t_amb_c,
                "q_mean_w": res.q_mean_w,
                "q_peak_w": res.q_peak_w,
                "t_max_c": t_amb_c + res.t_rise_k,
                "t_rise_k": res.t_rise_k,
                "delivered_ah": res.delivered_ah,
                "duration_s": float(res.time_s[-1]),
                "path": "dfn_coupled",
            })
    return pd.DataFrame(rows)


def sweep_schemes_network(
    rates=(0.5, 1.0, 1.5, 2.0, 3.0, 5.0),
    scheme_keys=("natural", "forced_air", "cold_plate", "immersion", "pcm"),
    ambients=(15.0, 25.0, 35.0, 45.0),
    v_min: float = 3.0,
    q_scale: float = 1.0,
    cache: dict | None = None,
) -> pd.DataFrame:
    """路径 B：快模型铺满 方案 × 倍率 × 环境温度。"""
    rows = []
    for t_amb in ambients:
        for c in rates:
            t_arr, q_arr, info = q_profile(c, t_amb_c=t_amb, v_min=v_min, cache=cache)
            q = q_arr * q_scale
            for key in scheme_keys:
                scheme = SCHEMES[key]
                net = build_cell(
                    h=scheme.h_mid,
                    pcm_extra_capacity_j_k=scheme.pcm_extra_capacity_j_k,
                )
                out = simulate_two_node(t_arr, q, t_amb, net)
                tc, ts = out["t_core_c"], out["t_surface_c"]
                t_max = float(max(tc.max(), ts.max()))
                rows.append({
                    "scheme": key,
                    "scheme_name": scheme.name,
                    "h": scheme.h_mid,
                    "area_fraction": scheme.area_fraction,
                    "c_rate": c,
                    "t_amb_c": t_amb,
                    "q_mean_w": float(np.mean(q)),
                    "q_peak_w": float(np.max(q)),
                    "q_source": info["source"],
                    "duration_s": float(t_arr[-1]),
                    "t_max_c": t_max,
                    "t_core_max_c": float(tc.max()),
                    "t_surface_max_c": float(ts.max()),
                    "core_surface_gap_k": float(tc.max() - ts.max()),
                    "t_rise_k": t_max - t_amb,
                    "t_to_45c_s": _time_to(ts, t_arr, T_LIMIT_C),
                    "t_to_60c_s": _time_to(ts, t_arr, T_WARN_C),
                    "pass_45c": bool(t_max <= T_LIMIT_C),
                    "in_model_domain": bool(t_max <= T_MODEL_MAX_C),
                    "q_scale": q_scale,
                    "path": "network",
                })
    return pd.DataFrame(rows)


def pcm_buffer(
    c_rate: float = 3.0,
    t_amb_c: float = 25.0,
    h: float = 8.0,
    pcm_capacity_j_k: float = 120.0,
    v_min: float = 3.0,
    cache: dict | None = None,
    window_s: float = 1800.0,
) -> dict:
    """PCM 的"延迟"到底值多少：对比有/无相变层的温升曲线。

    结论（报告 §5.5）：PCM 只把**升到某个温度的时间**往后推，不改变稳态温度。
    它买的是"从超温到被处理"的时间窗，不是散热能力——所以它不能替代 h，
    只能衔接两次高倍率工况之间的间歇。
    """
    t_arr, q_arr, _ = q_profile(c_rate, t_amb_c=t_amb, v_min=v_min, cache=cache)
    mask = t_arr <= min(t_arr[-1], window_s)
    t_w, q_w = t_arr[mask], q_arr[mask]
    base = simulate_two_node(t_w, q_w, t_amb_c, build_cell(h=h))
    with_pcm = simulate_two_node(
        t_w, q_w, t_amb_c, build_cell(h=h, pcm_extra_capacity_j_k=pcm_capacity_j_k)
    )
    return {
        "c_rate": c_rate,
        "t_amb_c": t_amb_c,
        "h": h,
        "window_s": float(t_w[-1]),
        "t_max_base_c": float(max(base["t_core_c"].max(), base["t_surface_c"].max())),
        "t_max_pcm_c": float(max(with_pcm["t_core_c"].max(), with_pcm["t_surface_c"].max())),
        "dt_max_reduction_k": float(
            max(base["t_core_c"].max(), base["t_surface_c"].max())
            - max(with_pcm["t_core_c"].max(), with_pcm["t_surface_c"].max())
        ),
        "t_to_45_base_s": _time_to(base["t_surface_c"], t_w, T_LIMIT_C),
        "t_to_45_pcm_s": _time_to(with_pcm["t_surface_c"], t_w, T_LIMIT_C),
        "pcm_capacity_j_k": pcm_capacity_j_k,
    }
