"""01 · 产热分析：倍率扫描、产热分解、Q∝C^α 标度律与跨参数集交叉验证。

对应博远基础版阶段 1「产热分析」+ 报告 §2、§3。
产出：
  results/metrics/01_heat_sweep.json      倍率 × 产热分解全表
  results/metrics/01_power_law.json       标度律拟合与外推
  results/figures/01a_heat_decomposition.png   Q(t) 与分解
  results/figures/01b_shares_vs_rate.png       三部分占比随倍率
  results/figures/01c_q_soc.png                Q(SOC) 对齐曲线
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
from bmthermal.heat import (  # noqa: E402
    C_RATE_TRUSTED,
    HIGH_RATE_SET,
    PRIMARY_SET,
    fit_power_law,
    heat_at_soc,
    predict_q,
    run_dfn_heat,
)

OUT = Path(__file__).resolve().parents[1] / "results"
MET = OUT / "metrics"
MET.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    rates_trusted = [0.5, 1.0, 1.5, 2.0]

    # ---- 1. 主参数集（Chen2020）实测点：分解 ----
    print("[1/3] Chen2020 0.5–2C 产热分解 ...")
    rows = []
    curves = {}
    for c in rates_trusted:
        r = run_dfn_heat(c, isothermal=True)
        rows.append(r.summary())
        curves[c] = r
        print(f"   {c:g}C  Q={r.q_mean_w:6.3f} W  peak={r.q_peak_w:6.3f} W  "
              f"Ah={r.delivered_ah:5.2f}  份额 ohm={r.shares()['ohmic']:.2f} "
              f"irr={r.shares()['irreversible']:.2f} rev={r.shares()['reversible']:.2f}")

    qs = [curves[c].q_mean_w for c in rates_trusted]
    law = fit_power_law(rates_trusted, qs)
    print(f"   标度律 Q = {law['k']:.4f} · C^{law['alpha']:.3f}   R²={law['r2']:.4f}")

    # ---- 2. 高倍率交叉验证（Mohtat2020 形态 + 标度律幅值）----
    print("[2/3] 高倍率交叉验证 3C/5C ...")
    high = []
    for c in (3.0, 4.0, 5.0):
        r_m = run_dfn_heat(c, isothermal=True, parameter_set=HIGH_RATE_SET)
        q_pred = predict_q(c, law)
        dev = (q_pred / r_m.q_mean_w - 1.0) * 100.0
        high.append({
            "c_rate": c,
            "mohtat_q_mean_w": r_m.q_mean_w,
            "mohtat_q_peak_w": r_m.q_peak_w,
            "mohtat_delivered_ah": r_m.delivered_ah,
            "power_law_pred_w": q_pred,
            "dev_pct": dev,
            "mohtat_shares": r_m.shares(),
        })
        rows.append(r_m.summary())
        print(f"   {c:g}C  Mohtat Q={r_m.q_mean_w:6.3f} W  标度律 {q_pred:6.3f} W  偏差 {dev:+.1f}%")

    # ---- 3. Q(SOC) 对齐 ----
    print("[3/3] Q(SOC) 对齐与出图 ...")
    soc_grid = np.linspace(1.0, 0.0, 101)
    q_soc = {}
    for c in rates_trusted:
        _, q_on_grid = heat_at_soc(curves[c], soc_grid)
        q_soc[str(c)] = q_on_grid.tolist()

    payload = {
        "primary_set": PRIMARY_SET,
        "high_rate_set": HIGH_RATE_SET,
        "trusted_rate_max": C_RATE_TRUSTED,
        "sweep": rows,
        "power_law": law,
        "power_law_measured_q": {str(c): qs[i] for i, c in enumerate(rates_trusted)},
        "high_rate_cross_check": high,
        "q_soc_grid": soc_grid.tolist(),
        "q_soc": q_soc,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (MET / "01_heat_sweep.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---------------- 图 ----------------
    plotting.setup_style()

    # 1a: Q(t) 分解（2C 为代表）
    r = curves[2.0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(r.time_s, r.q_total_w, label="总产热", color="#111827")
    ax[0].plot(r.time_s, r.q_ohmic_w, label="欧姆", color="#B45309")
    ax[0].plot(r.time_s, r.q_irrev_w, label="不可逆极化", color="#1D4ED8")
    ax[0].plot(r.time_s, r.q_rev_w, label="可逆熵热", color="#047857")
    ax[0].set_xlabel("时间 (s)"); ax[0].set_ylabel("产热功率 (W)")
    ax[0].set_title("2C 放电产热分解（Chen2020）"); ax[0].legend(fontsize=8)
    for c in rates_trusted:
        ax[1].plot(curves[c].time_s, curves[c].q_total_w, label=f"{c:g}C")
    ax[1].set_xlabel("时间 (s)"); ax[1].set_ylabel("总产热 (W)")
    ax[1].set_title("倍率对 Q(t) 的影响"); ax[1].legend(fontsize=8)
    plotting.save(fig, "01a_heat_decomposition.png")

    # 1b: 份额 vs 倍率
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    cs = rates_trusted
    oh = [curves[c].shares()["ohmic"] for c in cs]
    ir = [curves[c].shares()["irreversible"] for c in cs]
    rv = [curves[c].shares()["reversible"] for c in cs]
    ax.stackplot(cs, oh, ir, rv, labels=["欧姆", "不可逆极化", "可逆熵热"],
                 colors=["#B45309", "#1D4ED8", "#047857"], alpha=0.85)
    ax.set_xlabel("倍率 (C)"); ax.set_ylabel("能量占比")
    ax.set_title("产热构成随倍率的变化"); ax.legend(fontsize=8, loc="lower right")
    plotting.save(fig, "01b_shares_vs_rate.png")

    # 1c: Q(SOC)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for c in rates_trusted:
        ax.plot(soc_grid, q_soc[str(c)], label=f"{c:g}C")
    ax.set_xlabel("SOC"); ax.set_ylabel("产热功率 (W)")
    ax.set_title("按 SOC 对齐的产热曲线"); ax.legend(fontsize=8)
    plotting.save(fig, "01c_q_soc.png")

    print(f"完成，用时 {payload['elapsed_s']} s")


if __name__ == "__main__":
    main()
