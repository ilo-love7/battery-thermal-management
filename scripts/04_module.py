"""04 · 模组级温度场与一致性（定安 D1）。

单芯分析回答"这颗电芯会到多少度"，模组回答"最热与最冷差多少"。
ΔT ≤ 5 K 是模组设计常用一致性判据（定安 D1 明确给出）。

场景：2×6 圆柱模组 + 底部液冷板，0.5–3C 恒流 + 环境 25 °C。
同时扫描流量，回答"要把沿程温升压进预算需要多少流量"。

产出：
  results/metrics/04_module.csv / .json
  results/figures/04a_module_field.png     模组温度场
  results/figures/04b_flow_sweep.png       流量 ↔ 一致性
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
from bmthermal.heat import q_profile  # noqa: E402
from bmthermal.module import (  # noqa: E402
    ModuleSpec,
    consistency_report,
    required_flow_lpm,
    simulate_module,
)

OUT = Path(__file__).resolve().parents[1] / "results"
MET = OUT / "metrics"
MET.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    cache: dict = {}
    rows = []
    field_snapshot = None

    # ---- 1. 基准工况：倍率 × 流量 ----
    print("[1/2] 模组一致性扫描（倍率 × 流量）...")
    for c in (0.5, 1.0, 2.0, 3.0):
        t_arr, q_arr, info = q_profile(c, t_amb_c=25.0, cache=cache)
        for flow in (0.5, 1.0, 2.0, 4.0):
            spec = ModuleSpec(h_w_m2k=500.0, area_fraction=0.25, flow_lpm=flow, t_in_c=25.0)
            res = simulate_module(t_arr, q_arr, spec, q_scale=1.0)
            cons = consistency_report(res)
            rows.append({
                "c_rate": c, "flow_lpm": flow,
                "q_mean_w": float(np.mean(q_arr)),
                "coolant_rise_k": res["coolant_rise_k"],
                "delta_t_max_k": res["delta_t_max_k"],
                "t_max_c": res["t_max_c"],
                "t_cool_out_c": res["t_cool_out_c"],
                "pass_5k": cons["pass"],
                "required_flow_lpm": required_flow_lpm(float(np.mean(q_arr)) * spec.n_cells, spec),
            })
            print(f"   {c:g}C flow={flow:g} L/min  ΔT={res['delta_t_max_k']:5.2f} K  "
                  f"Tmax={res['t_max_c']:5.1f} C  冷却液升温={res['coolant_rise_k']:5.2f} K  "
                  f"{'PASS' if cons['pass'] else 'FAIL'}")
            if c == 2.0 and flow == 2.0:
                field_snapshot = res

    # ---- 2. 产热不确定性（q_scale）----
    print("[2/2] 产热不确定性稳健性（q × 0.7 / 1.0 / 1.5）...")
    robust = []
    t_arr, q_arr, _ = q_profile(2.0, t_amb_c=25.0, cache=cache)
    spec = ModuleSpec(h_w_m2k=500.0, area_fraction=0.25, flow_lpm=2.0, t_in_c=25.0)
    for s in (0.7, 1.0, 1.5):
        res = simulate_module(t_arr, q_arr, spec, q_scale=s)
        robust.append({"q_scale": s, "delta_t_max_k": res["delta_t_max_k"],
                       "t_max_c": res["t_max_c"], "coolant_rise_k": res["coolant_rise_k"]})
        print(f"   q×{s:g}  ΔT={res['delta_t_max_k']:5.2f} K  Tmax={res['t_max_c']:5.1f} C")

    payload = {
        "module": {"n_rows": 2, "n_cols": 6, "h_w_m2k": 500.0, "area_fraction": 0.25,
                   "limit_k": 5.0},
        "sweep": rows,
        "robustness_q_scale": robust,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (MET / "04_module.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    import pandas as pd
    pd.DataFrame(rows).to_csv(MET / "04_module.csv", index=False)

    # ---------------- 图 ----------------
    plotting.setup_style()

    # 4a: 温度场（2C, 2 L/min）
    if field_snapshot is not None:
        temps = np.asarray(field_snapshot["temps_c"])  # (n, n_t)
        idx = field_snapshot["index"]
        final = temps[:, -1].reshape(2, 6)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        im = ax[0].imshow(final, cmap="YlOrRd", aspect="auto")
        for r in range(2):
            for cc in range(6):
                ax[0].text(cc, r, f"{final[r, cc]:.1f}", ha="center", va="center", fontsize=9)
        ax[0].set_xticks(range(6)); ax[0].set_yticks(range(2))
        ax[0].set_xlabel("列（沿流向 →）"); ax[0].set_ylabel("排")
        ax[0].set_title("模组末端电芯温度 (°C)（2C, 2 L/min）")
        fig.colorbar(im, ax=ax[0], shrink=0.8)
        # 时序：最热/最冷
        t_max_series = temps.max(axis=0); t_min_series = temps.min(axis=0)
        ax[1].plot(field_snapshot["t"], t_max_series, label="最热电芯", color="#DC2626")
        ax[1].plot(field_snapshot["t"], t_min_series, label="最冷电芯", color="#1D4ED8")
        ax[1].fill_between(field_snapshot["t"], t_min_series, t_max_series,
                           alpha=0.15, color="#6B7280", label="温差带")
        ax[1].set_xlabel("时间 (s)"); ax[1].set_ylabel("温度 (°C)")
        ax[1].set_title(f"模组温差演化（ΔT_max={field_snapshot['delta_t_max_k']:.2f} K）")
        ax[1].legend(fontsize=8)
        plotting.save(fig, "04a_module_field.png")

    # 4b: 流量扫描
    import pandas as pd
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for c in (0.5, 1.0, 2.0, 3.0):
        sub = df[df.c_rate == c]
        ax.plot(sub["flow_lpm"], sub["delta_t_max_k"], "o-", label=f"{c:g}C")
    ax.axhline(5.0, color="#DC2626", ls="--", lw=1, label="ΔT ≤ 5 K 判据")
    ax.set_xlabel("冷却液流量 (L/min)"); ax.set_ylabel("模组最大温差 (K)")
    ax.set_title("流量对一致性的影响（2×6 模组）"); ax.legend(fontsize=8)
    plotting.save(fig, "04b_flow_sweep.png")

    print(f"完成，用时 {payload['elapsed_s']} s")


if __name__ == "__main__":
    main()
