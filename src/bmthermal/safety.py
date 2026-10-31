"""极端工况与预警裕度：快充 + 高温 + 散热退化叠加。

边界声明（必须写清楚，否则会越界）
----------------------------------
本模块的 60 °C / 80 °C 是**工程预警阈值**，不是热失控起始温度。
真正的 T1/T2/T3 特征温度与链式反应动力学属于 B3（Hatchard 模型）的范围，
本项目不碰。这里只回答一个工程问题：

    散热退化到最坏时，电芯多久撞到 80 °C、还剩多少裕度？

散热退化怎么建模
----------------
三种失效通道统一折算成**有效 h 的折减系数**：

* 风扇失效 / 风道堵塞 → 强制风冷退化到自然对流（50 → 8，折减 0.16）
* 液冷流量下降 / 流道局部堵塞 → h 随流量 0.8 次方，流量减半 → 折减约 0.57
* 导热垫老化 / 接触热阻上升 → 同样是 h 折减，作用在面积项上

折减系数取 1.0 / 0.7 / 0.5 / 0.3 / 0.15 五档，覆盖"轻微积灰"到"风扇完全失效"。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cooling import T_MODEL_MAX_C, T_WARN_C, _time_to
from .heat import q_profile
from .network import build_cell, simulate_two_node
from .params import SCHEMES

#: 高裕度预警阈值（°C）
T_ALARM_C = 80.0
#: 散热退化折减系数网格
DEGRADATION_GRID = (1.0, 0.7, 0.5, 0.3, 0.15)


@dataclass(frozen=True)
class Scenario:
    """一个极端工况。"""

    name: str
    c_rate: float
    t_amb_c: float
    scheme_key: str = "cold_plate"
    h_factor: float = 1.0
    q_scale: float = 1.0
    v_min: float = 3.0

    @property
    def h_effective(self) -> float:
        return SCHEMES[self.scheme_key].h_mid * self.h_factor


def run_scenario(sc: Scenario, cache: dict | None = None) -> dict:
    """跑一个极端工况，返回最高温度、到达阈值的时间与裕度。"""
    t_arr, q_arr, info = q_profile(sc.c_rate, t_amb_c=sc.t_amb_c, v_min=sc.v_min, cache=cache)
    net = build_cell(h=sc.h_effective)
    out = simulate_two_node(t_arr, q_arr * sc.q_scale, sc.t_amb_c, net)
    tc, ts = out["t_core_c"], out["t_surface_c"]
    t_core_max = float(tc.max())
    return {
        "scenario": sc.name,
        "scheme": sc.scheme_key,
        "c_rate": sc.c_rate,
        "t_amb_c": sc.t_amb_c,
        "h_factor": sc.h_factor,
        "h_effective": sc.h_effective,
        "q_scale": sc.q_scale,
        "q_mean_w": float(np.mean(q_arr) * sc.q_scale),
        "q_source": info["source"],
        "duration_s": float(t_arr[-1]),
        "t_surface_max_c": float(ts.max()),
        "t_core_max_c": t_core_max,
        "core_surface_gap_k": float(tc.max() - ts.max()),
        # 到阈值时间按**芯温**算：安全边界由最热处决定，表面热电偶会低估
        # （芯-壳温差在 5C 可达 10 K 以上，见 network.py）。
        "t_to_60c_s": _time_to(tc, t_arr, T_WARN_C),
        "t_to_80c_s": _time_to(tc, t_arr, T_ALARM_C),
        "margin_to_80c_k": float(T_ALARM_C - t_core_max),
        "exceeds_80c": bool(t_core_max >= T_ALARM_C),
        "in_model_domain": bool(t_core_max <= T_MODEL_MAX_C),
    }


def degradation_map(
    c_rate: float = 3.0,
    ambients=(25.0, 35.0, 45.0),
    scheme_key: str = "natural",
    h_factors=DEGRADATION_GRID,
    q_scale: float = 1.0,
    cache: dict | None = None,
) -> pd.DataFrame:
    """散热退化 × 环境温度 → 最高温度与裕度。"""
    rows = []
    for t_amb in ambients:
        for f in h_factors:
            sc = Scenario(
                name=f"{scheme_key}|{c_rate:g}C|{t_amb:g}C|h×{f:g}",
                c_rate=c_rate, t_amb_c=t_amb, scheme_key=scheme_key,
                h_factor=f, q_scale=q_scale,
            )
            rows.append(run_scenario(sc, cache))
    return pd.DataFrame(rows)
