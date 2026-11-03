"""实验方案量化——三项硬件开放项的样本量/精度/判据预注册。

开放项收尾的「数据方案设计」可计算部分
--------------------------------------
v1.10 后只剩三项硬件开放项：① 5C 台架 m(c) 实测（LFP 5C 带
[2.30, 3.35] 定夺）；② 物理 post-mortem 位点；③ 3C/5C × 多温度
老化组。方案不写空话——用 SNL 逐芯实测方差做：

1. **功效分析**（two-sample z 检验近似）：区分 m=2.30 vs 3.10
   需要每臂 n 芯；power(n) 曲线；
2. **测量精度预算**：容量计精度 → fade 率 SE → m SE（delta 法），
   反推 m 分辨率目标下的 σ_Q 要求；
3. **判据预注册**：各实验的 go/no-go 规则（先注册后测量，防
   事后挑口径——v1.4 起的负结果纪律延伸）。

纯函数模块；数据侧统计在 scripts/45_exp_design.py。
"""
from __future__ import annotations

import math

import numpy as np

# 判据锚点（v1.9/40 定下的 LFP 5C 带）
M5_LO = 2.30          # 机械单调性下界（=3C 水平）
M5_STAR = 3.10        # 吞吐打平（γ=1）
M5_HI = 3.35          # 经验上界
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80


def z_quantile(p: float) -> float:
    """标准正态分位数（无 scipy 依赖的 Acklam 近似）。"""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    # Acklam 有理逼近（|err| < 1.15e-9）
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def power_two_group(delta: float, sd: float, n: int,
                    alpha: float = DEFAULT_ALPHA) -> float:
    """两独立组均值差 z 检验功效（每臂 n，合并 sd）。"""
    if n < 2 or sd <= 0:
        raise ValueError("n>=2 and sd>0 required")
    se = sd * math.sqrt(2.0 / n)
    z_a = z_quantile(1.0 - alpha / 2.0)
    return float(_phi(abs(delta) / se - z_a))


def sample_size_two_group(delta: float, sd: float,
                          alpha: float = DEFAULT_ALPHA,
                          power: float = DEFAULT_POWER) -> int:
    """达到目标功效的每臂最小样本量（向上取整，下限 2）。"""
    if delta <= 0 or sd <= 0:
        raise ValueError("delta>0 and sd>0 required")
    z_a = z_quantile(1.0 - alpha / 2.0)
    z_b = z_quantile(power)
    n = 2.0 * sd * sd * (z_a + z_b) ** 2 / (delta * delta)
    return max(2, int(math.ceil(n)))


def power_curve(delta: float, sd: float, n_max: int = 12,
                alpha: float = DEFAULT_ALPHA) -> dict:
    ns = list(range(2, n_max + 1))
    return {"n": ns,
            "power": [power_two_group(delta, sd, n, alpha) for n in ns]}


def slope_se(sigma_q_ah: float, efc_window: float, n_obs: int) -> float:
    """容量噪声 σ_Q → 窗口线性 fade 率 SE（等间隔 EFC 观测）。

    SE(b) = σ_Q / sqrt(Σ(x−x̄)²) = σ_Q·sqrt(12/N)/W
    """
    if sigma_q_ah < 0 or efc_window <= 0 or n_obs < 3:
        raise ValueError("sigma_q>=0, window>0, n_obs>=3")
    return float(sigma_q_ah * math.sqrt(12.0 / n_obs) / efc_window)


def required_sigma_q(rate_se_target: float, efc_window: float,
                     n_obs: int) -> float:
    """目标 fade 率 SE 反推容量计 σ_Q 要求（Ah）。"""
    if rate_se_target <= 0:
        raise ValueError("rate_se_target>0")
    return float(rate_se_target * efc_window / math.sqrt(12.0 / n_obs))


def m_ratio_se(rate_ref: float, rate_c: float,
               sd_ref: float, sd_c: float,
               n_ref: int, n_c: int) -> float:
    """m=r_c/r_ref 的 SE（delta 法，独立组）。"""
    if rate_ref <= 0 or rate_c <= 0:
        raise ValueError("rates must be positive")
    m = rate_c / rate_ref
    se2 = m * m * ((sd_c / rate_c) ** 2 / n_c + (sd_ref / rate_ref) ** 2 / n_ref)
    return float(math.sqrt(se2))


def decision_5c(m_est: float, ci_low: float, ci_high: float,
                m_lo: float = M5_LO, m_star: float = M5_STAR,
                m_hi: float = M5_HI) -> str:
    """LFP 5C 方向预注册判据（先注册后测量）。

    - CI 整体 < m_star → 吞吐收益为正（带收窄至打平点下方）
    - CI 整体 > m_star → 吞吐收益为负（衰减成本超过时间收益）
    - CI 跨 m_star → 不可判（带维持，硬件再加样本）
    CI 同时用于带界校核（是否仍贴 m_lo/m_hi）。
    """
    if not (ci_low <= m_est <= ci_high):
        raise ValueError("CI must contain estimate")
    if ci_high < m_star:
        base = "throughput_positive"
    elif ci_low > m_star:
        base = "throughput_negative"
    else:
        base = "inconclusive"
    eps = 1e-9   # 2.30+0.05 浮点不等于 2.35（边界判定加容差）
    if ci_high <= m_lo + 0.05 + eps:
        return base + "_at_lower_bound"
    if ci_low >= m_hi - 0.05 - eps:
        return base + "_at_upper_bound"
    return base


def decision_postmortem(lam_35c: float, lam_25c: float,
                        crack_35c: float, crack_25c: float) -> str:
    """post-mortem 两读法裁决判据（35 vs 25 °C 同 fade 深度配对）。

    读法 b（高温 LAM 型加速）成立的预注册条件：
    35 °C 的 LAM 指标（裂纹密度/活性材料损失）显著 > 25 °C 配对
    且曲线几何归属（share_shape）方向一致；否则读法 a（映射失效）
    得权。裂纹为 0 的位点缺失也如实记录。
    """
    lam_diff = lam_35c - lam_25c
    crack_diff = crack_35c - crack_25c
    if lam_diff > 0 and crack_diff > 0:
        return "reading_b_confirmed"
    if lam_diff <= 0 and crack_diff <= 0:
        return "reading_a_gains_weight"
    return "mixed_evidence"


def design_matrix(c_dis_list: tuple = (3.0, 5.0),
                  temps: tuple = (15.0, 25.0, 35.0),
                  chems: tuple = ("LFP", "NMC"),
                  n_per_cell: int = 4) -> list[dict]:
    """3C/5C × 多温度老化组设计矩阵（含对照 1C）。"""
    rows = []
    for chem in chems:
        for c in c_dis_list:
            for t in temps:
                rows.append({
                    "chem": chem, "c_dis": c, "temp_c": t,
                    "n_cells": n_per_cell, "role": "target",
                    "rpt_every_efc": 100,
                })
        # 对照臂：1C 同温度（m 分母）
        for t in temps:
            rows.append({
                "chem": chem, "c_dis": 1.0, "temp_c": t,
                "n_cells": n_per_cell, "role": "reference",
                "rpt_every_efc": 100,
            })
    return rows
