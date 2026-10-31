"""02 · 冷却方案扫参与选型 map（博远基础版阶段 2 + 定安 D4）。

两条路径（见 cooling.py docstring）：
  A. DFN 耦合锚点：schemes × 倍率 ≤2C，贵但可信
  B. 两节点快模型：schemes × 0.5–5C × 环境温度，铺满选型矩阵
并解析求解"满足 ΔT ≤ 45 °C 所需的 h"——这是选型 map 的解析骨架。

产出：
  results/metrics/02_dfn_anchor.csv / .json
  results/metrics/02_network_matrix.csv
  results/metrics/02_required_h.json
  results/figures/02a_scheme_compare.png     各方案 Q 与 T 对比
  results/figures/02b_selection_map.png      h × C → 稳态温升 map（主图）
  results/figures/02c_required_h.png         所需 h 随倍率
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from bmthermal import plotting  # noqa: E402
from bmthermal.cooling import (  # noqa: E402
    T_LIMIT_C,
    max_rise_at_h,
    required_h,
    sweep_schemes_dfn,
    sweep_schemes_network,
)
from bmthermal.heat import predict_q, q_law  # noqa: E402
from bmthermal.params import CELL, SCHEMES  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MET = OUT / "metrics"
MET.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    cache: dict = {}

    # ---- 1. DFN 耦合锚点 ----
    print("[1/3] 路径 A：DFN 耦合锚点（≤2C）...")
    dfn = sweep_schemes_dfn(
        rates=(0.5, 1.0, 2.0),
        scheme_keys=("natural", "forced_air", "cold_plate", "immersion"),
    )
    dfn.to_csv(MET / "02_dfn_anchor.csv", index=False)
    for _, r in dfn.iterrows():
        print(f"   {r['scheme']:12s} {r['c_rate']:g}C  Q={r['q_mean_w']:5.2f}W  "
              f"Tmax={r['t_max_c']:5.1f}C  rise={r['t_rise_k']:5.1f}K")

    # ---- 2. 快模型铺满 ----
    print("[2/3] 路径 B：两节点快模型矩阵（0.5–5C × 15–45 °C）...")
    net = sweep_schemes_network(
        rates=(0.5, 1.0, 2.0, 3.0, 5.0),
        scheme_keys=("natural", "forced_air", "cold_plate", "immersion", "pcm"),
        ambients=(15.0, 25.0, 35.0, 45.0),
        cache=cache,
    )
    net.to_csv(MET / "02_network_matrix.csv", index=False)
    sel = net[(net.t_amb_c == 25.0)]
    for _, r in sel.iterrows():
        flag = "PASS" if r["pass_45c"] else "FAIL"
        print(f"   {r['scheme']:12s} {r['c_rate']:g}C  T={r['t_max_c']:6.1f}C  {flag}")

    # ---- 3. 解析所需 h ----
    print("[3/3] 解析所需 h（ΔT ≤ 45 °C 预算）...")
    law = q_law()
    req = []
    for c in (0.5, 1.0, 2.0, 3.0, 5.0):
        q = predict_q(c, law)
        # 温升预算 = 45 - 25（按 25 °C 环境）
        h_req = required_h(q, T_LIMIT_C - 25.0)
        rises = {k: max_rise_at_h(q, SCHEMES[k].h_mid) for k in SCHEMES}
        req.append({"c_rate": c, "q_pred_w": q, "h_required_w_m2k": h_req,
                    "rise_at_scheme_mid": rises})
        print(f"   {c:g}C  Q={q:5.2f}W  h_req={h_req:8.1f} W/(m²·K)")

    (MET / "02_required_h.json").write_text(
        json.dumps({"power_law": law, "allowed_rise_k": T_LIMIT_C - 25.0, "required": req},
                   indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (MET / "02_dfn_anchor.json").write_text(
        json.dumps(dfn.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # ---------------- 图 ----------------
    plotting.setup_style()
    colors = plotting.SCHEME_COLORS

    # 2a: 方案对比（25 °C，各方案在 1/3/5C 的 Tmax）
    sel25 = net[net.t_amb_c == 25.0]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    schemes = list(colors)
    xs = np.arange(len(schemes))
    for j, c in enumerate((1.0, 3.0, 5.0)):
        sub = sel25[sel25.c_rate == c].set_index("scheme")
        vals = [sub.loc[s, "t_max_c"] if s in sub.index else np.nan for s in schemes]
        ax.bar(xs + (j - 1) * 0.26, vals, width=0.26, label=f"{c:g}C",
               color=["#9CA3AF", "#6B7280", "#374151"][j])
    ax.axhline(T_LIMIT_C, color="#DC2626", ls="--", lw=1, label=f"{T_LIMIT_C:g} °C 上限")
    ax.set_xticks(xs)
    ax.set_xticklabels([SCHEMES[s].name for s in schemes], fontsize=8)
    ax.set_ylabel("最高温度 (°C)"); ax.set_title("冷却方案 × 倍率（环境 25 °C）")
    ax.legend(fontsize=8)
    plotting.save(fig, "02a_scheme_compare.png")

    # 2b: 选型 map（主图）：h → 稳态温升，标出各方案区间
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    h_grid = np.logspace(0.7, 3.4, 120)
    for c in (1.0, 2.0, 3.0, 5.0):
        q = predict_q(c, law)
        rise = [max_rise_at_h(q, h) for h in h_grid]
        ax.plot(h_grid, rise, label=f"{c:g}C ({q:.1f} W)")
    for key, scheme in SCHEMES.items():
        ax.axvspan(scheme.h_lo, scheme.h_hi, alpha=0.12, color=colors[key])
        ax.text(np.sqrt(scheme.h_lo * scheme.h_hi), ax.get_ylim()[1] * 0.97,
                scheme.name, rotation=90, fontsize=7, va="top", ha="center", color=colors[key])
    ax.axhline(T_LIMIT_C - 25.0, color="#DC2626", ls="--", lw=1.2,
               label=f"温升预算 {T_LIMIT_C - 25:g} K（25 °C 环境）")
    ax.set_xscale("log"); ax.set_xlabel("等效换热系数 h (W/(m²·K))")
    ax.set_ylabel("稳态温升 (K)")
    ax.set_title("选型 map：h × 倍率 → 稳态温升（含方案适用区间）")
    ax.legend(fontsize=8, loc="lower left")
    plotting.save(fig, "02b_selection_map.png")

    # 2c: 所需 h
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    cs = [r["c_rate"] for r in req]
    hs = [r["h_required_w_m2k"] for r in req]
    ax.plot(cs, hs, "o-", color="#1D4ED8")
    for key, scheme in SCHEMES.items():
        ax.axhspan(scheme.h_lo, scheme.h_hi, alpha=0.10, color=colors[key])
    ax.set_yscale("log"); ax.set_xlabel("倍率 (C)")
    ax.set_ylabel("所需 h (W/(m²·K))")
    ax.set_title(f"满足 ΔT ≤ {T_LIMIT_C - 25:g} K 所需换热系数")
    plotting.save(fig, "02c_required_h.png")

    print(f"完成，用时 {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
