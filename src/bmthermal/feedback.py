"""老化-热耦合闭环：容量衰减 → 内阻增长 → 产热增加 → 温度上升 → 加速老化。

模型
----
既有一路（05/15）是**开环**的：温度历程 → 损伤积分 → 寿命倍数，
隐含"电芯参数恒定"。真实电芯在寿命期内内阻会涨，产热随之增加，
形成正反馈::

    衰减 f = 1 − SOH                [0, 0.2]，EOL 取 20 % 容量损失
    内阻   R(f) = R0·(1 + β·f)      [文献：LFP 到 EOL 内阻 +30–70 % → β∈1.5–3.5]
    产热   Q(f) = Q0·(1 + η·(R/R0 − 1))
           = Q0·(1 + η·β·f)         [η = 随内阻增长的产热份额（不可逆热占比）]

衰减-损伤映射取 ``f = (D/D_EOL)^p``：p=1 线性（损伤即衰减）、
p=0.5 平方根（SEI 生长主导）。两种都做。

闭环寿命怎么算（D_EOL 消掉）
----------------------------
EFC 计数域上 ``dD = d_cycle(f) dN``（每 EFC 损伤随衰减水平变化）::

    N_EOL = ∫ dD/d_cycle = D_EOL · ∫₀¹ du / d_cycle(u)      (u = f/0.2, p=1)

无反馈：``N_no = D_EOL / d_cycle(0)``。
**反馈放大系数** ``A = N_no / N_EOL``（≥1，反馈缩短寿命）与
方案间寿命比 ``N_a/N_b = [∫du/d_a] / [∫du/d_b]`` 都与 D_EOL 无关——
不需要绝对寿命标定就能回答"4.7× 会不会被反馈漂移"。

绝对 EFC 锚点：脚本层可选地锚定"基准无反馈 2000 EFC"（LFP 文献典型值，
标注为假设；scripts/19 用 CALCE CS2 实测衰减做对标）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

#: EOL 容量损失（20 % 为业界通用 EOL 判据）
FADE_EOL = 0.2

#: 内阻增长系数 β 默认与敏感性档（R/R0 = 1 + β·f，f=0.2 时 R/R0−1 = 0.2β）
BETA_DEFAULT = 2.5
BETA_GRID = (1.5, 2.5, 3.5)

#: 随内阻增长的产热份额 η 默认与档位（不可逆热占比量级，§2 分解口径）
ETA_DEFAULT = 0.4
ETA_GRID = (0.2, 0.4, 0.7)

#: 衰减-损伤映射指数 p（1=线性，0.5=SEI 平方根）
P_GRID = (0.5, 1.0)


@dataclass(frozen=True)
class FeedbackParams:
    beta: float = BETA_DEFAULT
    eta: float = ETA_DEFAULT
    fade_exponent: float = 1.0

    @property
    def q_mult_slope(self) -> float:
        """产热倍数对归一化衰减 u 的斜率：Q_mult(u) = 1 + slope·u。"""
        return self.eta * self.beta * FADE_EOL


def heat_multiplier(u: float, params: FeedbackParams) -> float:
    """归一化衰减水平 u∈[0,1]（u=1 即 f=FADE_EOL）下的产热倍数。"""
    return 1.0 + params.q_mult_slope * float(u)


def resistance_ratio(fade: float, beta: float = BETA_DEFAULT) -> float:
    """R/R0 = 1 + β·fade。"""
    return 1.0 + float(beta) * float(fade)


def life_amplification(
    d_cycle_fn: Callable[[float], float],
    params: FeedbackParams = FeedbackParams(),
    n_grid: int = 41,
) -> dict:
    """反馈放大系数 A = N_no/N_endl（≥1，反馈缩短寿命）。

    ``d_cycle_fn(u)``：归一化衰减 u=f/FADE_EOL∈[0,1] 下每 EFC 损伤。

    推导：``f = FADE_EOL·(D/D_EOL)^p``，换元 u=(D/D_EOL)^p 得
    ``N = (D_EOL/p)∫₀¹ u^(1/p−1)/d_cycle(u) du``，
    ``A = N_no/N = p / [d0·∫₀¹ u^(1/p−1)/d_cycle(u) du]``。
    p=1 退化为普通梯形积分；p=0.5 时权重 u¹ 偏向寿命后期
    （平方根衰减意味着早期衰减快、损伤预算大部分消耗在前段）。
    """
    us = np.linspace(0.0, 1.0, int(n_grid))
    d = np.array([float(d_cycle_fn(float(u))) for u in us], dtype=float)
    if np.any(d <= 0) or not np.all(np.isfinite(d)):
        raise ValueError("d_cycle_fn 必须返回有限正值")
    p = float(params.fade_exponent)
    if p <= 0:
        raise ValueError("fade_exponent 必须为正")
    # p≤1 时指数 1/p−1 ≥ 0，u=0 无奇点；p>1 才需要端点可积处理
    pw = 1.0 / p - 1.0
    weight = us**pw if pw != 0 else np.ones_like(us)
    if pw > 0:
        weight[0] = 0.0
    integral = float(np.trapezoid(weight / d, us))
    d0 = float(d[0])
    a = p / (d0 * integral)
    return {
        "amplification": a,
        "life_loss_pct": (1.0 - 1.0 / a) * 100.0,
        "d_cycle_fresh": d0,
        "d_cycle_eol": float(d[-1]),
        "d_growth_pct": (float(d[-1]) / d0 - 1.0) * 100.0,
        "u_grid": us.tolist(),
        "d_cycle_grid": d.tolist(),
        "integral": integral,
        "params": {
            "beta": params.beta, "eta": params.eta,
            "fade_exponent": p,
            "q_mult_slope": params.q_mult_slope,
        },
    }


def pair_life_ratio(
    d_cycle_a: Callable[[float], float],
    d_cycle_b: Callable[[float], float],
    params: FeedbackParams = FeedbackParams(),
    n_grid: int = 21,
) -> dict:
    """方案 a 相对 b 的闭环寿命比（含反馈），并与开环值对比。

    开环比 = d_b(0)/d_a(0)；闭环比 = [∫du/d_b]/[∫du/d_a]；
    漂移 = 闭环/开环 − 1（负 = 反馈让 a 相对更吃亏）。
    """
    ra = life_amplification(d_cycle_a, params, n_grid)
    rb = life_amplification(d_cycle_b, params, n_grid)
    open_ratio = rb["d_cycle_fresh"] / ra["d_cycle_fresh"]
    # N_x = (D_EOL/p)·I_x → N_a/N_b = I_a/I_b（与 life_ratio=d_ref/d_case 同向）
    closed_ratio = ra["integral"] / rb["integral"]
    return {
        "open_loop_ratio": open_ratio,
        "closed_loop_ratio": closed_ratio,
        "ratio_drift_pct": (closed_ratio / open_ratio - 1.0) * 100.0,
        "amplification_a": ra["amplification"],
        "amplification_b": rb["amplification"],
    }
