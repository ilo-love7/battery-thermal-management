"""PCM 相变材料：焓法（温度钉扎平台）与等效热容法的对比。

等效热容法（v1.0 用的）把潜热摊成常值热容增量——只对**不跨熔点**的
工况可靠。焓法用 H(T)=C·T+L·f(T)（f 为熔化分数的平滑阶跃），
熔化区间内温度被"钉"在 Tm 附近（热容局部放大一个数量级），
熔完后保护消失。

本模块回答：等效热容法**何时失效**。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCMProps:
    """PCM 层属性。

    ``latent_j`` 是**电芯级总潜热（J）**——H(T)=C·T+L·f(T) 里的 L，
    熔化窗口内释放。换算到等效热容增量是 L/width（J/K）。
    v1.0 的 ``pcm_extra_capacity_j_k=120``（J/K，常值）相当于
    L=120×width≈360 J 摊进同宽窗口、但**永远不释放**。
    """

    t_melt_c: float = 45.0
    width_k: float = 3.0            # 熔化窗口宽度（纯物质窄，共晶/复合材料 2–8 K）
    latent_j: float = 360.0         # 总潜热 J（电芯级）
    c_base_j_k: float = 42.78       # 电芯本体热容
    g_w_k: float = 0.05             # h·A（自然对流基底）
    t_amb_c: float = 25.0


def melt_fraction(t_c: np.ndarray, p: PCMProps) -> np.ndarray:
    """熔化分数 f(T) ∈ [0,1]，平滑阶跃（tanh 型）。"""
    x = (np.asarray(t_c, float) - p.t_melt_c) / max(p.width_k, 1e-6)
    return 0.5 * (1.0 + np.tanh(1.5 * x))


def c_eff_enthalpy(t_c: np.ndarray, p: PCMProps) -> np.ndarray:
    """焓法的有效热容 C + L·df/dT（熔化窗口内的峰）。"""
    t = np.asarray(t_c, float)
    x = (t - p.t_melt_c) / max(p.width_k, 1e-6)
    dfdt = 0.5 * 1.5 / max(p.width_k, 1e-6) * (1.0 - np.tanh(1.5 * x) ** 2)
    return p.c_base_j_k + p.latent_j * dfdt


def simulate_enthalpy(
    q_w: float,
    p: PCMProps,
    duration_s: float = 3600.0,
    dt_s: float = 1.0,
) -> dict:
    """焓法：dH/dt = Q − G(T−T_amb)，H(T)=C·T+L·f(T)（前向欧拉，dt=1 s 足够）。"""
    n = int(duration_s / dt_s) + 1
    t = np.linspace(0.0, duration_s, n)
    T = np.empty(n)
    T[0] = p.t_amb_c
    melted = np.zeros(n)
    for k in range(n - 1):
        c_here = float(c_eff_enthalpy(np.array([T[k]]), p)[0])
        dT = (q_w - p.g_w_k * (T[k] - p.t_amb_c)) / c_here * dt_s
        T[k + 1] = T[k] + dT
        melted[k + 1] = float(melt_fraction(np.array([T[k + 1]]), p)[0])
    return {
        "t_s": t, "t_c": T, "melt_fraction": melted,
        "t_peak_c": float(T.max()),
        "melt_final": float(melted[-1]),
    }


def simulate_equiv_capacity(
    q_w: float,
    p: PCMProps,
    duration_s: float = 3600.0,
    dt_s: float = 1.0,
) -> dict:
    """等效热容法（v1.0 口径）：潜热摊成常值热容增量 L/ΔT_ref。

    与 v1.0 的 ``pcm_extra_capacity_j_k`` 等价——增量不随温度变化。
    """
    n = int(duration_s / dt_s) + 1
    t = np.linspace(0.0, duration_s, n)
    T = np.empty(n)
    T[0] = p.t_amb_c
    c_here = p.c_base_j_k + p.latent_j / max(p.width_k, 1e-6)  # 摊到熔化窗口
    for k in range(n - 1):
        dT = (q_w - p.g_w_k * (T[k] - p.t_amb_c)) / c_here * dt_s
        T[k + 1] = T[k] + dT
    return {"t_s": t, "t_c": T, "t_peak_c": float(T.max())}


def divergence(
    q_w: float,
    p: PCMProps,
    duration_s: float = 3600.0,
) -> dict:
    """两种方法的峰值温度差与结论翻转判定。"""
    e = simulate_enthalpy(q_w, p, duration_s)
    v = simulate_equiv_capacity(q_w, p, duration_s)
    d_peak = v["t_peak_c"] - e["t_peak_c"]
    return {
        "q_w": q_w, "t_melt_c": p.t_melt_c, "latent_j": p.latent_j,
        "t_peak_enthalpy_c": e["t_peak_c"],
        "t_peak_equiv_c": v["t_peak_c"],
        "peak_diff_k": d_peak,
        "melt_final": e["melt_final"],
        "equiv_overestimates": d_peak > 0,
    }
