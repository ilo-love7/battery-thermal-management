"""18 · 老化-热耦合闭环：内阻增长正反馈会不会漂移寿命比？

开环（05/15）假设电芯参数恒定。本脚本补上正反馈环：
衰减 → 内阻 R/R0=1+βf → 产热 Q×(1+η(βf)) → 温度↑ → 衰减加速。

三问三答：
1. 反馈让单方案寿命缩水多少？（放大系数 A，寿命损失 %）
2. **3C 液冷 vs 1C 自然 的寿命比（C5 的 4.7×）会不会被反馈漂移？**
3. 结论对 β/η/p 的敏感性——漂移是否稳健？

绝对 EFC 锚点（标注为假设）：基准（自然对流 1C）无反馈 2000 EFC 到 EOL，
用于把放大系数换算成 EFC 数字；寿命比本身不需要这个锚点（D_EOL 消掉）。

产出：
  results/metrics/18_feedback.json
  results/figures/18a_aging_feedback.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmthermal import plotting  # noqa: E402
from bmthermal.aging import damage_integral  # noqa: E402
from bmthermal.feedback import (  # noqa: E402
    BETA_GRID,
    ETA_GRID,
    FADE_EOL,
    FeedbackParams,
    P_GRID,
    life_amplification,
    pair_life_ratio,
)
from bmthermal.heat import q_profile  # noqa: E402
from bmthermal.network import build_cell, simulate_two_node  # noqa: E402
from bmthermal.params import SCHEMES  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MET = OUT / "metrics"
MET.mkdir(parents=True, exist_ok=True)

#: 假设锚点（LFP 文献典型，非实测）：基准无反馈寿命
ANCHOR_EFC = 2000.0

# 代表性工况：(倍率, 方案) —— 覆盖 C5 关键对比的两端
CASES = [
    (1.0, "natural"),      # 基准
    (3.0, "natural"),      # 热端
    (3.0, "forced_air"),
    (3.0, "cold_plate"),   # C5 液冷端
    (5.0, "cold_plate"),
]


def d_cycle_factory(c: float, scheme_key: str, params: FeedbackParams,
                    cache: dict, fn_cache: dict | None = None):
    """返回 d_cycle_fn(u)：归一化衰减 u 下该工况每 EFC 损伤（s）。

    d_cycle 只依赖产热斜率 η·β（p 不进 Q 缩放），按斜率缓存避免重复仿真。
    """
    slope = round(params.q_mult_slope, 9)
    if fn_cache is not None and (c, scheme_key, slope) in fn_cache:
        return fn_cache[(c, scheme_key, slope)]
    t_arr, q_arr, _ = q_profile(c, t_amb_c=25.0, cache=cache)
    scheme = SCHEMES[scheme_key]
    net = build_cell(h=scheme.h_mid,
                     pcm_extra_capacity_j_k=scheme.pcm_extra_capacity_j_k)

    def d_cycle_fn(u: float) -> float:
        mult = 1.0 + slope * u
        out = simulate_two_node(t_arr, q_arr * mult, 25.0, net)
        return damage_integral(t_arr, out["t_core_c"])

    if fn_cache is not None:
        fn_cache[(c, scheme_key, slope)] = d_cycle_fn
    return d_cycle_fn


def main() -> None:
    t0 = time.time()
    cache: dict = {}
    fn_cache: dict = {}

    print("[1/3] 基线场景（β=2.5, η=0.4, p=1）...")
    base_params = FeedbackParams()
    results = {}
    fns = {}
    for c, key in CASES:
        fn = d_cycle_factory(c, key, base_params, cache, fn_cache)
        fns[(c, key)] = fn
        r = life_amplification(fn, base_params)
        results[f"{c:g}C_{key}"] = r
        print(f"   {c:g}C {key:12s} 产热+{r['d_growth_pct']:5.1f}% @EOL  "
              f"放大 A={r['amplification']:.4f}  寿命损失 {r['life_loss_pct']:.2f}%")

    print("[2/3] 关键对比：3C 液冷 vs 1C 自然（C5）...")
    pair = pair_life_ratio(fns[(3.0, "cold_plate")], fns[(1.0, "natural")],
                           base_params)
    print(f"   开环比 {pair['open_loop_ratio']:.3f}  "
          f"闭环比 {pair['closed_loop_ratio']:.3f}  "
          f"漂移 {pair['ratio_drift_pct']:+.2f}%")

    pairs = {
        "3C_cold_plate_vs_1C_natural": pair,
        "3C_natural_vs_1C_natural": pair_life_ratio(
            fns[(3.0, "natural")], fns[(1.0, "natural")], base_params),
        "5C_cold_plate_vs_1C_natural": pair_life_ratio(
            fns[(5.0, "cold_plate")], fns[(1.0, "natural")], base_params),
        "3C_cold_plate_vs_3C_natural": pair_life_ratio(
            fns[(3.0, "cold_plate")], fns[(3.0, "natural")], base_params),
    }

    print("[3/3] 敏感性 β×η×p ...")
    sens = []
    for beta in BETA_GRID:
        for eta in ETA_GRID:
            for p in P_GRID:
                par = FeedbackParams(beta=beta, eta=eta, fade_exponent=p)
                # 产热斜率变了，d_cycle_fn 必须重建（只有 Q 缩放依赖参数）
                fn_cold = d_cycle_factory(3.0, "cold_plate", par, cache, fn_cache)
                fn_base = d_cycle_factory(1.0, "natural", par, cache, fn_cache)
                pr = pair_life_ratio(fn_cold, fn_base, par)
                ra = life_amplification(fn_cold, par)
                sens.append({
                    "beta": beta, "eta": eta, "p": p,
                    "q_mult_slope": par.q_mult_slope,
                    "amp_3c_cold": ra["amplification"],
                    "pair_open": pr["open_loop_ratio"],
                    "pair_closed": pr["closed_loop_ratio"],
                    "ratio_drift_pct": pr["ratio_drift_pct"],
                })
    drifts = [s["ratio_drift_pct"] for s in sens]
    print(f"   漂移范围 [{min(drifts):+.2f}%, {max(drifts):+.2f}%]，"
          f"中位 {np.median(drifts):+.2f}%")

    # EFC 锚点换算（假设）
    d_base0 = results["1C_natural"]["d_cycle_fresh"]
    efc_rows = []
    for k, r in results.items():
        n_no = ANCHOR_EFC * d_base0 / r["d_cycle_fresh"]
        n_fb = n_no / r["amplification"]
        efc_rows.append({
            "case": k,
            "efc_no_feedback": n_no,
            "efc_with_feedback": n_fb,
            "amp": r["amplification"],
        })

    drift_pair = pair["ratio_drift_pct"]
    same_sign = all(d > 0 for d in drifts) or all(d < 0 for d in drifts)
    robust = same_sign and abs(drift_pair) < 10.0 and max(abs(d) for d in drifts) < 15.0
    direction = "液冷相对优势略增" if drift_pair > 0 else "液冷相对优势略减"
    verdict = (
        f"反馈放大系数 A≈{results['3C_cold_plate']['amplification']:.3f}"
        f"（液冷 3C）～{results['3C_natural']['amplification']:.3f}（自然 3C）——"
        f"热端付更多反馈代价；C5 寿命比漂移 {drift_pair:+.1f}%"
        f"（开环 {pair['open_loop_ratio']:.2f} → 闭环 {pair['closed_loop_ratio']:.2f}，"
        f"{direction}），β×η×p 敏感性 [{min(drifts):+.1f}%, {max(drifts):+.1f}%]；"
        + ("方向稳健、幅度温和，4–5× 区间口径在含反馈下仍成立（数值偏区间上限）"
           if robust else "漂移幅度或方向不稳健，引用寿命比须显式声明反馈口径")
    )
    print("判词:", verdict)

    payload = {
        "model": {
            "fade_eol": FADE_EOL,
            "anchor_efc": ANCHOR_EFC,
            "anchor_note": "基准（1C 自然）无反馈 2000 EFC 到 EOL，LFP 文献典型值（假设，非实测）",
            "r_growth": "R/R0 = 1 + β·f",
            "q_growth": "Q_mult = 1 + η·β·f",
            "base_params": {"beta": 2.5, "eta": 0.4, "p": 1.0},
        },
        "cases": results,
        "pairs": pairs,
        "sensitivity": sens,
        "sensitivity_drift_range_pct": [float(min(drifts)), float(max(drifts))],
        "efc_anchor_table": efc_rows,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (MET / "18_feedback.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- 图 ----------------
    fig, axes = plotting.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for (c, key), style in [((1.0, "natural"), "-"), ((3.0, "natural"), "-"),
                            ((3.0, "cold_plate"), "--"), ((5.0, "cold_plate"), ":")]:
        r = results[f"{c:g}C_{key}"]
        ax.plot(r["u_grid"], r["d_cycle_grid"], style, lw=1.8,
                label=f"{c:g}C {SCHEMES[key].name}")
    ax.set_xlabel("归一化衰减 u（1 = EOL）")
    ax.set_ylabel("每 EFC 损伤 [s]")
    ax.set_title("衰减过程中每 EFC 损伤的增长（正反馈强度）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    names = list(results.keys())
    amps = [results[k]["amplification"] for k in names]
    colors = ["#c0392b" if "natural" in k else "#2980b9" for k in names]
    ax.bar(range(len(names)), amps, color=colors, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("放大系数 A = N_no/N_endl")
    ax.set_title("反馈寿命损失（A>1 即反馈缩短寿命）")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    ds = [s["ratio_drift_pct"] for s in sens]
    ax.hist(ds, bins=15, color="#8e44ad", alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(float(np.median(ds)), color="#e67e22", lw=1.5,
               label=f"中位 {np.median(ds):+.1f}%")
    ax.set_xlabel("C5 寿命比漂移 [%]（闭环/开环 − 1）")
    ax.set_ylabel("敏感性组合数")
    ax.set_title("β×η×p 全组合下的寿命比漂移分布")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    labels = [r["case"] for r in efc_rows]
    n_no = [r["efc_no_feedback"] for r in efc_rows]
    n_fb = [r["efc_with_feedback"] for r in efc_rows]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, n_no, 0.36, label="无反馈", color="#95a5a6")
    ax.bar(x + 0.18, n_fb, 0.36, label="含反馈", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("到 EOL 的 EFC（假设锚点 2000）")
    ax.set_title("绝对寿命（锚点换算，仅量级参考）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("18 · 老化-热耦合闭环：内阻增长正反馈", fontsize=13)
    plotting.save(fig, "18a_aging_feedback.png")

    print(f"完成，耗时 {payload['elapsed_s']} s")


if __name__ == "__main__":
    main()
