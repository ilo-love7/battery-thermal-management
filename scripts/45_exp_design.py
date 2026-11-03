# -*- coding: utf-8 -*-
"""WP45 三项硬件实验方案量化（v1.11 第二线）。

开放项①②③（5C 台架 / post-mortem / 高倍率×多温度组）的可计算
方案设计——不写空话方案，用 SNL 逐芯实测方差做功效分析与精度
预算，判据预注册（先注册后测量）：

1. **5C 台架 m(c)**：区分 LFP 5C 带 [2.30, 3.35] 的打平点 3.10
   ——每臂 n 节功效曲线（δ=|3.10−2.30|×r₁，sd 取 SNL 组内实测）；
   m SE ≤ 0.2 的样本量反推；
2. **测量精度预算**：容量计 σ_Q → 窗口 fade 率 SE → m SE（delta
   法），反推 m 分辨率 0.2 对应的 σ_Q 要求（与电芯间散布对比）；
3. **post-mortem 位点**：两读法裁决样本对（LFP 1C 35 °C vs 25 °C
   同 fade 深度）+ 机械份额验证组（3C）+ nuisance 对照（NMC 3C），
   decision_postmortem 预注册；
4. **3C/5C×T 组**：全矩阵（2 化学 × {3C,5C}×{15,25,35}+1C 对照）
   通道日预算与 RPT 节奏（每 100 EFC，SNL 经验）。

产物：results/metrics/45_exp_design.json、
results/figures/45a_exp_design.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal.exp_design import (  # noqa: E402
    M5_HI, M5_LO, M5_STAR,
    decision_5c, decision_postmortem, design_matrix, m_ratio_se,
    power_curve, required_sigma_q, sample_size_two_group, slope_se,
)
from bmthermal.plotting import save as save_fig, setup_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

MET = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures"
SUMMARY_CSV = ROOT / "results" / "cache" / "snl_summary.csv"

EFC_WINDOW = 240.0     # 目标 5C 寿命 ~1000 EFC 的 1–25% 窗（对齐项目口径）
N_RPT_OBS = 25         # 窗内 RPT 点数（每 ~10 EFC 一次）
M_SE_TARGET = 0.20     # m 分辨率目标（CI 半宽 ~0.4 @95%）


def main() -> None:
    t0 = time.time()
    MET.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY_CSV)
    df = df[(df.soc_lo == 0) & (df.soc_hi == 100)]

    def grp(chem: str, temp: float, c: float):
        g = df[(df.chem == chem) & (df.temp_c == temp) & (df.c_dis == c)]
        r = g["fade_rate_per_efc"].dropna().values
        return {"n": int(r.size), "mean": float(np.mean(r)) if r.size else float("nan"),
                "sd": float(np.std(r, ddof=1)) if r.size > 1 else float("nan")}

    # ---- [1] 5C 功效分析 ------------------------------------------
    print("[1/4] 5C 台架功效分析...")
    ref = grp("LFP", 25.0, 1.0)
    g3 = grp("LFP", 25.0, 3.0)
    g2 = grp("LFP", 25.0, 2.0)
    r1 = ref["mean"]
    sd_ref = ref["sd"]
    # 5C 组内 sd 无实测 → 用 3C sd ×1.2 保守代理（假设定量化，如实记）
    sd_5_assumed = (g3["sd"] if np.isfinite(g3["sd"]) else sd_ref) * 1.2
    delta_rate = abs(M5_STAR - M5_LO) * r1      # 两假说的 fade 率差
    n_needed = sample_size_two_group(delta_rate, sd_5_assumed, power=0.80)
    pc = power_curve(delta_rate, sd_5_assumed, n_max=12)
    # m SE（delta 法）随 n 变化（参考臂固定 n=4）
    mse_by_n = {}
    for n in (2, 3, 4, 6, 8, 10, 12):
        r5 = M5_STAR * r1
        mse_by_n[n] = m_ratio_se(r1, r5, sd_ref, sd_5_assumed, 4, n)
    print(f"   r1={r1:.3e}/EFC  sd5(假定)={sd_5_assumed:.3e}  "
          f"δ_rate={delta_rate:.3e}  n≥{n_needed}/臂 (power 0.8)")
    print(f"   m SE @n=4: {mse_by_n[4]:.3f}  @n=8: {mse_by_n[8]:.3f}  "
          f"(目标 ≤{M_SE_TARGET})")

    # ---- [2] 精度预算 ---------------------------------------------
    print("[2/4] 容量计精度预算...")
    # 目标：测量噪声贡献 ≤ 电芯散布 1/3（不淹没生物学信号）
    meas_rate_se_target = sd_5_assumed / 3.0 / np.sqrt(4.0)
    sigma_q_req = required_sigma_q(meas_rate_se_target, EFC_WINDOW, N_RPT_OBS)
    q_nom = float(df[df.chem == "LFP"].cap0_ah.median())
    sigma_q_rel = sigma_q_req / q_nom
    # 0.2% 典型台架容量计 → 对照
    sigma_q_typ = 0.002 * q_nom
    se_typ = slope_se(sigma_q_typ, EFC_WINDOW, N_RPT_OBS)
    print(f"   σ_Q 要求 ≤ {sigma_q_req*1000:.2f} mAh ({sigma_q_rel:.3%} of Q_nom)"
          f"；典型 0.2% 计 → SE {se_typ:.2e}/EFC "
          f"({'可' if sigma_q_typ <= sigma_q_req else '勉强/不可'}用)")

    # ---- [3] post-mortem 样本方案 ---------------------------------
    print("[3/4] post-mortem 样本方案...")
    pm_groups = [
        {"chem": "LFP", "c_dis": 1.0, "temp_c": 35.0, "n_sacrifice": 2,
         "role": "reading_b_test", "fade_trigger": 0.10,
         "measure": "SEM/EDS 正负极截面裂纹密度 + XRD + dQ/dV 对拍"},
        {"chem": "LFP", "c_dis": 1.0, "temp_c": 25.0, "n_sacrifice": 2,
         "role": "reading_b_pair_control", "fade_trigger": 0.10,
         "measure": "同上（同 fade 深度配对）"},
        {"chem": "LFP", "c_dis": 3.0, "temp_c": 25.0, "n_sacrifice": 2,
         "role": "mech_share_validate", "fade_trigger": 0.10,
         "measure": "同上 + 与 54%/62% 双估计器对拍"},
        {"chem": "NMC", "c_dis": 3.0, "temp_c": 25.0, "n_sacrifice": 2,
         "role": "nuisance_control", "fade_trigger": 0.10,
         "measure": "同上（nuisance 应无系统性 LAM 签名）"},
    ]
    # 预注册判据示例（拿到数据前先定规则）
    pm_rule_example = decision_postmortem(lam_35c=0.6, lam_25c=0.3,
                                          crack_35c=2.0, crack_25c=1.0)
    print(f"   牺牲芯 {sum(g['n_sacrifice'] for g in pm_groups)} 支 / 4 组；"
          f"判据预注册示例 → {pm_rule_example}")

    # ---- [4] 3C/5C×T 矩阵 -----------------------------------------
    print("[4/4] 3C/5C×T 组矩阵与通道预算...")
    matrix = design_matrix()
    total_cells = sum(r["n_cells"] for r in matrix)
    # 通道日：寿命 EFC ∝ 1/c（同日历长度），5C ~1000 EFC、3C ~1500、
    # 1C ~3000；周期时长 ∝ 1/c（容量/电流）→ 墙钟日数近似同量级
    life_efc = {1.0: 3000, 3.0: 1500, 5.0: 1000}
    hours_per_cycle = {1.0: 4.0, 3.0: 1.4, 5.0: 0.9}
    channel_days = 0.0
    for r in matrix:
        c = r["c_dis"]
        days = life_efc[c] * hours_per_cycle[c] / 24.0
        channel_days += days * r["n_cells"]
    print(f"   {total_cells} 芯 / {len(matrix)} 组，通道日预算 ≈ {channel_days:.0f}")

    # ---- 裁决 + 图 + JSON -----------------------------------------
    verdict = (
        f"三项硬件实验方案量化收口（开放项①②③数据方案设计）："
        f"① 5C 台架——区分 LFP 5C 打平点 m*={M5_STAR}（带 "
        f"[{M5_LO},{M5_HI}]）需每臂 n≥{n_needed}（δ_rate="
        f"{delta_rate:.2e}/EFC，sd 代理 {sd_5_assumed:.2e}=3C×1.2 假定），"
        f"n=4 时 m SE {mse_by_n[4]:.2f}（目标 {M_SE_TARGET}），"
        f"判据预注册：CI 全 <{M5_STAR} → 吞吐正 / 全 > → 负 / 跨 → "
        f"不可判。② post-mortem——35 vs 25 °C 同 fade 配对 2+2（两读法"
        f"裁决）+ 3C 验证 2 + NMC nuisance 对照 2 = 8 牺牲芯，"
        f"SEM/EDS+XRD+dQ/dV。③ 3C/5C×T——{total_cells} 芯/"
        f"{len(matrix)} 组全矩阵（含 1C 对照），通道日 ≈ {channel_days:.0f}，"
        f"RPT 每 100 EFC。测量精度：σ_Q ≤ {sigma_q_req*1000:.1f} mAh"
        f"（{sigma_q_rel:.2%} Q_nom）方可不淹没电芯散布。"
        f"⚠️ 5C sd 是 3C×1.2 代理（无 5C 实测方差）；功效数字是方案"
        f"下界参考，首组 5C 数据到手后应回填真实 sd 重算。"
    )
    print(verdict)

    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(pc["n"], pc["power"], "o-", color="#2980b9", lw=1.6)
    ax.axhline(0.8, color="#7f8c8d", ls=":")
    ax.axvline(n_needed, color="#c0392b", ls="--", lw=1,
               label=f"n*={n_needed}/臂")
    ax.set_xlabel("每臂样本量 n")
    ax.set_ylabel("功效")
    ax.set_title(f"a) 区分 m={M5_LO} vs {M5_STAR} 的功效（LFP 5C）")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ns = list(mse_by_n.keys())
    ax.plot(ns, [mse_by_n[n] for n in ns], "s-", color="#8e44ad", lw=1.6)
    ax.axhline(M_SE_TARGET, color="#7f8c8d", ls=":",
               label=f"目标 m SE={M_SE_TARGET}")
    ax.set_xlabel("5C 臂样本量 n（参考臂固定 4）")
    ax.set_ylabel("m 的 SE（delta 法）")
    ax.set_title("b) m 分辨率随样本量")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    rels = np.array([0.0005, 0.001, 0.002, 0.005, 0.01])
    ses = [slope_se(r * q_nom, EFC_WINDOW, N_RPT_OBS) for r in rels]
    ax.loglog(rels * 100, ses, "o-", color="#e67e22", lw=1.6,
              label="测量噪声贡献")
    ax.axhline(sd_5_assumed / 3.0, color="#7f8c8d", ls=":",
               label="电芯散布 /3（信号线）")
    ax.axvline(sigma_q_rel * 100, color="#c0392b", ls="--", lw=1,
               label=f"σ_Q 要求 {sigma_q_rel:.2%}")
    ax.set_xlabel("容量计 σ_Q [% of Q_nom]")
    ax.set_ylabel("fade 率 SE [/EFC]")
    ax.set_title("c) 测量精度预算")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        "预注册判据（先注册后测量）",
        "",
        f"5C LFP:  CI 全<{M5_STAR} → 吞吐正（带收窄）",
        f"         CI 全>{M5_STAR} → 吞吐负",
        f"         跨 {M5_STAR} → 不可判（加样本）",
        "",
        "post-mortem: 35C LAM 与裂纹均 > 25C 配对",
        "         → 读法 b（高温 LAM 加速）确认",
        "         双 ≤ → 读法 a（映射失效）得权",
        "         混合 → mixed（如实记）",
        "",
        f"3C×T: m(3C;T) CI 半宽 ≤0.5 替代 2C 代理带",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, va="top", family="sans-serif",
            bbox=dict(boxstyle="round", facecolor="#eaf2f8",
                      edgecolor="#2980b9"))
    ax.set_title("d) 判据预注册")

    fig.suptitle("45a 三项硬件实验方案量化（功效 / 精度 / 判据）", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "45a_exp_design.png")

    out = {
        "variance_source": {
            "file": str(SUMMARY_CSV),
            "ref_group": {"chem": "LFP", "temp_c": 25.0, "c_dis": 1.0, **ref},
            "g3": g3, "g2": g2,
            "sd_5_assumed": sd_5_assumed,
            "sd_5_note": "5C 无实测方差 → 3C sd×1.2 代理（假设定量化）",
        },
        "bench_5c": {
            "delta_rate_per_efc": delta_rate,
            "hypotheses_m": [M5_LO, M5_STAR, M5_HI],
            "n_per_arm_power80": n_needed,
            "power_curve": pc,
            "m_se_by_n": {str(k): v for k, v in mse_by_n.items()},
            "m_se_target": M_SE_TARGET,
            "decision_rule": {
                "positive": f"CI 整体 < {M5_STAR}",
                "negative": f"CI 整体 > {M5_STAR}",
                "inconclusive": "CI 跨打平点 → 加样本",
            },
        },
        "precision_budget": {
            "efc_window": EFC_WINDOW, "n_rpt_obs": N_RPT_OBS,
            "sigma_q_req_ah": sigma_q_req, "sigma_q_req_rel": sigma_q_rel,
            "q_nom_ah": q_nom,
            "typical_0p2pct_se": se_typ,
            "adequate": bool(sigma_q_typ <= sigma_q_req),
        },
        "postmortem_plan": {
            "groups": pm_groups,
            "total_sacrifice": sum(g["n_sacrifice"] for g in pm_groups),
            "decision_fn": "decision_postmortem(lam_35c, lam_25c, crack_35c, crack_25c)",
            "rule_example": pm_rule_example,
        },
        "rate_temp_matrix": {
            "groups": matrix, "total_cells": total_cells,
            "channel_days": channel_days,
            "rpt_every_efc": 100,
            "decision_rule": "m(3C;T) CI 半宽 ≤ 0.5 替代 2C 代理带",
        },
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (MET / "45_exp_design.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {out['elapsed_s']}s")


if __name__ == "__main__":
    main()
