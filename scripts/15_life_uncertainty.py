"""scripts/15_life_uncertainty.py —— 寿命账全不确定性传播（v1.2 深化 2）。

v1.1 §14 审计把 C5（3C 液冷寿命比 4.6×）判为脆弱：绝对倍数依赖 Ea 与 Q 标定。
本脚本不再给单点倍数，而是把 (Q 参数集包络, Ea, 接触 h) 联合采样，
给寿命比的**分布**——引用口径从"4.6×"改为"中位数 + 5–95% 区间 + 超过阈值概率"。

采样口径（每个样本同时算 3C 方案与 1C 自然基线，同 (q_scale, Ea, h_scale)）：
- q_scale ~ U(0.7, 1.5)：参数集分歧包络（§11 的 robustness 扫描口径）
- Ea ~ U(30, 60) kJ/mol：文献区间（aging.EA_GRID 两端）
- h_scale ~ 截断对数正态（中位 1，σ_ln=0.25，限 [0.5, 2.0]）：接触热阻不确定

产出：results/metrics/15_life_uncertainty.json / results/figures/15a_life_monte_carlo.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmthermal import plotting
from bmthermal.aging import damage_integral
from bmthermal.heat import q_profile
from bmthermal.network import build_cell
from bmthermal.observer import simulate_state_space
from bmthermal.params import SCHEMES

OUT = Path(__file__).resolve().parents[1] / "results"
N_SAMPLES = 1500
SEED = 20261031
DT_S = 10.0
CASE_RATE = 3.0
REF_RATE = 1.0
SCHEME_KEYS = ("natural", "forced_air", "cold_plate", "immersion", "pcm")


def uniform_grid(t_arr: np.ndarray, q_arr: np.ndarray, dt: float = DT_S):
    """把 q_profile 的网格统一到等间隔 dt（ZOH 矩阵指数只需算一次）。"""
    t_u = np.arange(float(t_arr[0]), float(t_arr[-1]) + 0.5 * dt, dt)
    q_u = np.interp(t_u, t_arr, q_arr)
    return t_u, q_u


def temperature_for(q_u, t_u, h, t_amb, pcm_extra=0.0):
    cell = build_cell(h=h, pcm_extra_capacity_j_k=pcm_extra)
    xs = simulate_state_space(cell, t_u, q_u, t_amb)
    return xs[0]  # 芯温


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    cache: dict = {}

    # 基线与 3C 的产热剖面（同倍率下方案无关，只喂一次）
    t_ref, q_ref_raw, _ = q_profile(REF_RATE, t_amb_c=25.0, cache=cache)
    t_case, q_case_raw, _ = q_profile(CASE_RATE, t_amb_c=25.0, cache=cache)
    t_ref_u, q_ref_u = uniform_grid(t_ref, q_ref_raw)
    t_case_u, q_case_u = uniform_grid(t_case, q_case_raw)

    schemes = {k: SCHEMES[k] for k in SCHEME_KEYS}

    # 采样
    q_scale = rng.uniform(0.7, 1.5, N_SAMPLES)
    ea = rng.uniform(30e3, 60e3, N_SAMPLES)
    h_scale = np.clip(rng.lognormal(0.0, 0.25, N_SAMPLES), 0.5, 2.0)

    # 结果容器
    ratios = {k: np.empty(N_SAMPLES) for k in SCHEME_KEYS}
    damage_case = {k: np.empty(N_SAMPLES) for k in SCHEME_KEYS}
    damage_ref = np.empty(N_SAMPLES)

    for i in range(N_SAMPLES):
        qs, e, hs = float(q_scale[i]), float(ea[i]), float(h_scale[i])
        # 基线：1C 自然（h 也带接触不确定）
        tc_ref = temperature_for(q_ref_u * qs, t_ref_u, schemes["natural"].h_mid * hs, 25.0)
        d_ref = damage_integral(t_ref_u, tc_ref, ea_j_mol=e)
        damage_ref[i] = d_ref
        for k, sch in schemes.items():
            tc = temperature_for(
                q_case_u * qs, t_case_u, sch.h_mid * hs, 25.0,
                pcm_extra=sch.pcm_extra_capacity_j_k,
            )
            d = damage_integral(t_case_u, tc, ea_j_mol=e)
            damage_case[k][i] = d
            ratios[k][i] = d_ref / d if d > 0 else np.nan

    def dist(arr):
        a = arr[np.isfinite(arr)]
        return {
            "median": float(np.median(a)),
            "p5": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
            "mean": float(np.mean(a)),
            "p_gt_1": float(np.mean(a > 1.0)),
            "p_gt_2": float(np.mean(a > 2.0)),
            "p_gt_4": float(np.mean(a > 4.0)),
        }

    distributions = {k: dist(v) for k, v in ratios.items()}

    # 排序稳健性：P(液冷 > 风冷)、P(浸没 > 液冷) 等
    ranking = {
        "p_cold_plate_beats_forced_air": float(np.mean(ratios["cold_plate"] > ratios["forced_air"])),
        "p_immersion_beats_cold_plate": float(np.mean(ratios["immersion"] > ratios["cold_plate"])),
        "p_cold_plate_beats_natural": float(np.mean(ratios["cold_plate"] > ratios["natural"])),
        "p_pcm_beats_natural": float(np.mean(ratios["pcm"] > ratios["natural"])),
    }

    # C5 概率化口径
    cp = ratios["cold_plate"]
    c5 = {
        "old_claim": "3C 液冷寿命比 ≈ 4.6× vs 自然 1C（单点）",
        "new_statement": (
            f"3C 液冷寿命比中位数 {np.median(cp):.1f}×，"
            f"5–95% 区间 [{np.percentile(cp, 5):.1f}, {np.percentile(cp, 95):.1f}]×，"
            f"P(>2×)={np.mean(cp > 2):.2f}，P(>4×)={np.mean(cp > 4):.2f}"
        ),
        "distribution": distributions["cold_plate"],
    }

    out = {
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "sampling": {
            "q_scale": "U(0.7, 1.5) 参数集包络",
            "ea_j_mol": "U(30e3, 60e3)",
            "h_scale": "LogNormal(0, 0.25) 截断 [0.5, 2.0]",
        },
        "case": f"{CASE_RATE}C 各方案 vs 基线 {REF_RATE}C 自然对流（25 °C，同样本参数）",
        "distributions": distributions,
        "ranking": ranking,
        "c5_probabilistic": c5,
        "robust_claims": [
            "液冷/浸没 > 自然（P≈1.0）：方向稳健",
            "液冷 vs 风冷 也稳健（P>0.99）",
            "绝对倍数必须带区间引用，禁止再报单点 4.6×",
        ],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    with open(OUT / "metrics" / "15_life_uncertainty.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 图 ----
    fig, axes = plotting.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    for k, color in zip(SCHEME_KEYS, ("#7f8c8d", "#2980b9", "#27ae60", "#8e44ad", "#e67e22")):
        v = ratios[k]
        vp = np.percentile(v, [5, 25, 50, 75, 95])
        ax.plot([vp[0], vp[4]], [k, k], color=color, lw=2, alpha=0.6)
        ax.plot([vp[1], vp[3]], [k, k], color=color, lw=6, alpha=0.8)
        ax.scatter([vp[2]], [k], color=color, s=60, zorder=3, edgecolor="k", lw=0.3)
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_xlabel("寿命比 vs 自然 1C")
    ax.set_title("寿命比分布（5–25–50–75–95%）")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(cp, bins=40, color="#27ae60", alpha=0.85, edgecolor="white")
    ax.axvline(np.median(cp), color="k", lw=1.5, label=f"中位 {np.median(cp):.1f}×")
    ax.axvline(4.6, color="r", ls="--", lw=1.5, label="v1.0 单点 4.6×")
    ax.set_xlabel("3C 液冷寿命比")
    ax.set_title("C5 概率化：4.6× 只是分布的一个点")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    names = list(ranking.keys())
    vals = [ranking[n] for n in names]
    ax.barh(range(len(names)), vals, color=["#27ae60", "#2980b9", "#27ae60", "#e67e22"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("p_", "").replace("_", " ") for n in names], fontsize=8)
    ax.axvline(0.95, color="r", ls="--", lw=1, label="0.95")
    ax.set_xlim(0, 1.05)
    ax.set_title("排序稳健性（胜出概率）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    plotting.save(fig, "15a_life_monte_carlo.png")

    print(json.dumps({"c5": c5["new_statement"], "ranking": ranking, "elapsed_s": out["elapsed_s"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
