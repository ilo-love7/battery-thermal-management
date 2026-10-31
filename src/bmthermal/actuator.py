"""执行器产品化：真实泵/风机功率曲线 + 辨识脉冲排程。

为什么 P ∝ h³ 不够（scripts/21 的能耗口径局限）
------------------------------------------------
相似定律 P ∝ n³ 是理想风机/泵（无静扬程、效率恒定）的极限。真实执行器
有两个系统性偏差，都会改变"触发式冷却省多少能"的结论：

1. **空载功率** P_idle。电机铁损/驱动器功耗使 h→0 时 P 不归零；
   滞环低档不是"免费"的。
2. **静扬程/线性项**。冷板泵回路 P = ρgQ(H₀ + k_r Q²)/η 中的几何静
   扬程 H₀ 给出 ∝ Q 的线性份额——低速区功率比 h³ 高估省电。

统一模型（``ActuatorCurve``）::

    P(h_rel) = P_max · [f_idle + (1 − f_idle)·(α·h + (1 − α)·h³)]

- α = 0、f_idle = 0：理想相似定律（scripts/21 口径，对照组）；
- α = 0、f_idle ≈ 0.12–0.2：风机典型；
- α ≈ 0.3–0.4、f_idle ≈ 0.08–0.15：冷板泵典型（静扬程份额）。

P_max / f_idle / α 取文献工程典型量级（报告中标注"典型值，不冒充实测"）。

辨识脉冲排程（为什么需要"排"而不是"想到就跑"）
----------------------------------------------
scripts/20 的脉冲-弛豫辨识依赖弛豫窗内 h 恒定；scripts/21 的执行器有最短
保持时间。两者叠加的工程问题：

- **R_int 随老化漂移**（CS2：β 量级 0.5–2.5）。上次辨识的 R_int 用得越久，
  模型失配越大（scripts/20：r_scale 1.3 未校核 → 稳态偏差 −6.5 K）。
  ``ident_period_efc`` 由"允许的相对 R 误差 / β / 每 EFC 衰减"推出辨识周期。
- **辨识窗冻结控制**（时分复用）：弛豫拟合要求 h 恒定 → 辨识窗内控制律
  必须保持档位（``ident_hold_law``）。窗开在高热负荷段会放大超温风险——
  排程策略应优先放在低负荷窗口（``ident_schedule`` 支持负荷掩码）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActuatorCurve:
    """泵/风机功率曲线（相对转速/流量 h_rel ∈ [0, 1]）。"""

    p_max_w: float
    idle_frac: float = 0.0     # 空载功率占 P_max 份额
    static_frac: float = 0.0   # 线性项（静扬程）占可变功率份额

    def power_w(self, h_rel: float | np.ndarray) -> float | np.ndarray:
        h = np.clip(np.asarray(h_rel, dtype=float), 0.0, 1.0)
        var = self.static_frac * h + (1.0 - self.static_frac) * h**3
        p = self.p_max_w * (self.idle_frac + (1.0 - self.idle_frac) * var)
        return float(p) if np.ndim(h_rel) == 0 else p


#: 理想相似定律（scripts/21 口径；P_max 任意，归一化后与取值无关）
AFFINITY = ActuatorCurve(p_max_w=1.0, idle_frac=0.0, static_frac=0.0)
#: 风机典型：有空载份额，无静压线性项（文献工程典型量级）
FAN_TYPICAL = ActuatorCurve(p_max_w=40.0, idle_frac=0.15, static_frac=0.0)
#: 冷板泵典型：空载份额 + 静扬程线性份额（文献工程典型量级）
PUMP_TYPICAL = ActuatorCurve(p_max_w=25.0, idle_frac=0.10, static_frac=0.35)


def energy_from_trace(
    t_s: np.ndarray,
    h_cmd: np.ndarray,
    h_hi: float,
    curve: ActuatorCurve,
) -> float:
    """给定 h 命令轨迹的积分电能（W·s），h_rel = h_cmd / h_hi。"""
    t = np.asarray(t_s, dtype=float)
    h = np.asarray(h_cmd, dtype=float)
    dt = np.diff(t, prepend=t[0])
    dt[0] = 0.0
    p = curve.power_w(h / h_hi)
    return float(np.sum(p * dt))


def energy_norm_from_trace(
    t_s: np.ndarray,
    h_cmd: np.ndarray,
    h_lo: float,
    h_hi: float,
    curve: ActuatorCurve,
) -> float:
    """归一化能耗：同一曲线口径下「实际轨迹」/「恒满冷」。"""
    t = np.asarray(t_s, dtype=float)
    e_real = energy_from_trace(t, h_cmd, h_hi, curve)
    e_full = energy_from_trace(t, np.full_like(t, h_hi), h_hi, curve)
    return float(e_real / e_full) if e_full > 0 else np.nan


def duty_energy_norm(duty_high: float, h_lo: float, h_hi: float, curve: ActuatorCurve) -> float:
    """稳态滞环占空比下的归一化能耗（解析式，用于对拍仿真）。

    ``duty_high``=高档时间占比；低档 h_lo、高档 h_hi。
    """
    lo = float(curve.power_w(h_lo / h_hi))
    hi = float(curve.power_w(1.0))
    if hi <= 0:
        return np.nan
    return float(duty_high * hi + (1.0 - duty_high) * lo) / hi


def ident_period_efc(
    beta: float,
    fade_per_efc: float,
    r_tol: float = 0.05,
) -> float:
    """R_int 漂移约束下的辨识周期（EFC）。

    模型：R/R₀ = 1 + β·f（CS2/§18 口径），f 以 ``fade_per_efc``/EFC 线性增长。
    辨识后相对 R 误差从 0 涨到 β·Δf；令 β·Δf ≤ ``r_tol`` 反解 Δf，除以
    每 EFC 衰减即得周期。β 取 CS2 脉冲内阻下界（~0.5）时周期长，取 §18
    保守上界（2.5–3.5）时周期短——两者都落盘。
    """
    if beta <= 0 or fade_per_efc <= 0 or r_tol <= 0:
        raise ValueError("beta/fade_per_efc/r_tol 必须为正")
    df_max = r_tol / beta
    return float(df_max / fade_per_efc)


def ident_schedule(
    t_s: np.ndarray,
    period_s: float,
    window_s: float,
    q_w: np.ndarray | None = None,
    q_threshold_w: float | None = None,
) -> np.ndarray:
    """辨识窗掩码（True = 处于辨识窗，控制须冻结档位）。

    默认按固定周期铺窗；给了 ``q_w`` 与 ``q_threshold_w`` 时只在低负荷
    （q ≤ 阈值）且到达周期的时刻开窗——时分复用的"排程"策略。
    """
    t = np.asarray(t_s, dtype=float)
    n = t.size
    mask = np.zeros(n, dtype=bool)
    if period_s <= 0 or window_s <= 0:
        return mask
    next_due = t[0] + period_s
    for k in range(n):
        if t[k] < next_due:
            continue
        if q_w is not None and q_threshold_w is not None:
            if q_w[k] > q_threshold_w:
                # 高负荷：顺延到下一个采样点再试（保持周期基准不动，
                # 避免负荷长窗导致辨识饿死）
                continue
        k_end = np.searchsorted(t, t[k] + window_s, side="right")
        mask[k:k_end] = True
        next_due = t[k] + period_s
    return mask


def ident_hold_law(law, ident_mask: np.ndarray, t_s: np.ndarray):
    """把控制律包成"辨识窗内保持档位"的时分复用律。"""
    t = np.asarray(t_s, dtype=float)

    def law2(t_now: float, z, x_hat, mode: int) -> int:
        k = int(np.searchsorted(t, t_now, side="right") - 1)
        if 0 <= k < t.size and ident_mask[k]:
            return mode
        return law(t_now, z, x_hat, mode)

    return law2
