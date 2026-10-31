"""06 · 极端工况与安全裕度（定安 D5，衔接 B3）。

问题：**散热退化到最坏时，电芯多久撞到 80 °C、还剩多少裕度？**
（60/80 °C 是工程预警阈值，不是热失控起始温度——热失控归 B3。）

三种失效通道统一折算成 h 折减系数（见 safety.py）：
  风扇失效 → 自然对流；流道堵塞 → h ∝ ṁ^0.8；导热垫老化 → 面积项退化。

产出：
  results/metrics/06_safety.csv / .json
  results/figures/06a_degradation_map.png  散热退化 × 环境温度 → 最高温度（主图之一）
  results/figures/06b_margin.png           80 °C 裕度
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bmthermal import plotting  # noqa: E402
from bmthermal.safety import DEGRADATION_GRID, T_ALARM_C, Scenario, degradation_map, run_scenario

OUT = Path(__file__).resolve().parents[1] / "results"
MET = OUT / "metrics"
MET.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    cache: dict = {}

    print("[1/2] 散热退化 × 环境温度（3C 快充场景）...")
    frames = []
    for scheme_key in ("natural", "forced_air", "cold_plate"):
        df = degradation_map(
            c_rate=3.0, ambients=(25.0, 35.0, 45.0),
            scheme_key=scheme_key, h_factors=DEGRADATION_GRID, cache=cache,
        )
        frames.append(df)
        for _, r in df.iterrows():
            flag = "SAFE" if not r["exceeds_80c"] else "ALARM"
            print(f"   {scheme_key:12s} {r['t_amb_c']:4.0f}C h×{r['h_factor']:.2f}  "
                  f"Tcore={r['t_core_max_c']:6.1f} C  裕度={r['margin_to_80c_k']:+6.1f} K  {flag}")
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(MET / "06_safety.csv", index=False)

    print("[2/2] 组合场景（5C + 45 °C + 散热退化）...")
    combos = []
    for f in (1.0, 0.5, 0.3, 0.15):
        sc = Scenario(
            name=f"5C|45C|cold_plate|h×{f:g}",
            c_rate=5.0, t_amb_c=45.0, scheme_key="cold_plate", h_factor=f,
        )
        r = run_scenario(sc, cache)
        combos.append(r)
        print(f"   h×{f:.2f}  Tcore={r['t_core_max_c']:6.1f} C  "
              f"到80C: {r['t_to_80c_s']:7.0f} s  裕度={r['margin_to_80c_k']:+6.1f} K")

    payload = {
        "alarm_c": T_ALARM_C,
        "degradation_grid": list(DEGRADATION_GRID),
        "degradation_map": all_df.to_dict(orient="records"),
        "extreme_combos": combos,
        "boundary_note": (
            "60/80 °C 为工程预警阈值，非热失控起始温度；"
            "T1/T2/T3 特征温度与链式反应动力学属 B3 范围，本项目不建模。"
        ),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (MET / "06_safety.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # ---------------- 图 ----------------
    plotting.setup_style()
    colors = plotting.SCHEME_COLORS

    # 6a: 退化 map（热图：h_factor × ambient → T_core_max，逐方案小图）
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, scheme_key in zip(axes, ("natural", "forced_air", "cold_plate")):
        sub = all_df[all_df.scheme == scheme_key]
        piv = sub.pivot_table(index="h_factor", columns="t_amb_c", values="t_core_max_c")
        im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto", origin="lower",
                       vmin=25, vmax=120)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"×{v:g}" for v in piv.index])
        ax.set_xlabel("环境温度 (°C)")
        ax.set_title(f"{scheme_key}（3C）")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.values[i, j]
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color="black" if val < 90 else "white")
    axes[0].set_ylabel("散热能力折减 h×")
    fig.colorbar(im, ax=axes, shrink=0.85, label="芯温峰值 (°C)")
    fig.suptitle("极端工况安全边界：散热退化 × 环境温度（红线以上=超 80 °C 预警）", y=1.02)
    plotting.save(fig, "06a_degradation_map.png")

    # 6b: 裕度
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    labels, margins = [], []
    for _, r in all_df.iterrows():
        labels.append(f"{r['scheme'][:4]}|{r['t_amb_c']:.0f}C|×{r['h_factor']:.2f}")
        margins.append(r["margin_to_80c_k"])
    cols = ["#DC2626" if m < 0 else ("#B45309" if m < 20 else "#047857") for m in margins]
    ax.barh(range(len(margins)), margins, color=cols)
    ax.axvline(0, color="#111827", lw=1)
    ax.set_yticks(range(len(margins)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("距 80 °C 的裕度 (K)")
    ax.set_title("各场景到 80 °C 预警线的裕度（负值=已超）")
    plotting.save(fig, "06b_margin.png")

    print(f"完成，用时 {payload['elapsed_s']} s")


if __name__ == "__main__":
    main()
