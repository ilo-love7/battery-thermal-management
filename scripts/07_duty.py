"""07 · 动态工况复测（博远基础版阶段 4）：US06 × 冷却方案选型矩阵。

恒流扫参回答"稳态能力够不够"，US06 回答"脉冲扛不扛得住"。
本脚本用**真实测量的 US06 电流剖面**（CALCE A123，见 duty.py）做三件事：

1. DFN 等温算 US06 的 Q(t)：脉冲形态（峰值 1.46C → 峰值产热 ~1.5 W）；
2. 两节点网络铺满 方案 × 环境温度 的动态温度矩阵；
3. 恒流等效对照：US06（均值 0.21C）vs 恒流 0.21C，量化"动态惩罚"。

产出：
  results/metrics/07_duty.json
  results/figures/07a_duty_profile.png   US06 电流与产热
  results/figures/07b_duty_matrix.png    动态工况选型矩阵
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
from bmthermal.duty import (  # noqa: E402
    US06_PROVENANCE,
    load_us06,
    run_dfn_duty,
    run_network_duty,
)
from bmthermal.network import build_cell  # noqa: E402
from bmthermal.params import SCHEMES  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MET = OUT / "metrics"
MET.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()

    print("[1/3] DFN 等温：US06 产热 Q(t) ...")
    r = run_dfn_duty(scale=1.0, isothermal=True, duration_s=600.0)
    print(f"   Q_mean={r['q_mean_w']:.3f} W  Q_peak={r['q_peak_w']:.3f} W  "
          f"倍率峰值={np.max(r['c_rate']):.2f}C")

    # 3 个重复冲击（1800 s）：看热累积
    print("[2/3] 两节点：方案 × 环境温度（600 s 单循环）...")
    t_p, c_p = load_us06()
    rows = []
    for t_amb in (25.0, 35.0, 45.0):
        for key, scheme in SCHEMES.items():
            net = build_cell(h=scheme.h_mid, pcm_extra_capacity_j_k=scheme.pcm_extra_capacity_j_k)
            out = run_network_duty(r["t_s"], r["q_total_w"], t_amb, net)
            tc, ts = out["t_core_c"], out["t_surface_c"]
            t_max = float(max(tc.max(), ts.max()))
            rows.append({
                "duty": "US06", "scheme": key, "scheme_name": scheme.name,
                "h": scheme.h_mid, "t_amb_c": t_amb,
                "t_max_c": t_max, "t_rise_k": t_max - t_amb,
                "core_surface_gap_k": float((tc - ts).max()),
                "pass_45c": bool(t_max <= 45.0),
            })
            print(f"   {t_amb:4.0f}C {scheme.name:20s} Tmax={t_max:5.1f} C  "
                  f"rise={t_max - t_amb:5.1f} K  {'PASS' if t_max <= 45 else 'FAIL'}")

    # 恒流等效对照（0.21C 均值）
    print("[3/3] 恒流等效对照 ...")
    from bmthermal.heat import predict_q, q_law
    law = q_law()
    q_mean = r["q_mean_w"]
    equiv = []
    for key in ("natural", "forced_air", "cold_plate"):
        scheme = SCHEMES[key]
        net = build_cell(h=scheme.h_mid)
        out_dyn = run_network_duty(r["t_s"], r["q_total_w"], 25.0, net)
        out_cc = run_network_duty(r["t_s"], np.full_like(r["t_s"], q_mean), 25.0, net)
        equiv.append({
            "scheme": key,
            "t_max_dynamic_c": float(out_dyn["t_core_c"].max()),
            "t_max_const_c": float(out_cc["t_core_c"].max()),
            "dynamic_extra_k": float(out_dyn["t_core_c"].max() - out_cc["t_core_c"].max()),
        })
        print(f"   {key:12s} 动态 {out_dyn['t_core_c'].max():5.2f} C  "
              f"恒流等效 {out_cc['t_core_c'].max():5.2f} C  "
              f"差 {out_dyn['t_core_c'].max() - out_cc['t_core_c'].max():+.2f} K")

    # 重复冲击：单循环 600 s 对 5 Ah 芯几乎无热累积，3 循环（1800 s）才看得出
    # "脉冲 + 间歇散热"的分工。等温 DFN 取 Q(t)，两节点给温度。
    print("[2b/3] 重复冲击（1800 s = 3×US06）...")
    r3 = run_dfn_duty(scale=1.0, isothermal=True, duration_s=1800.0)
    repeat = []
    for t_amb in (35.0, 45.0):
        for key in ("natural", "forced_air", "cold_plate", "pcm"):
            scheme = SCHEMES[key]
            net = build_cell(h=scheme.h_mid, pcm_extra_capacity_j_k=scheme.pcm_extra_capacity_j_k)
            out1 = run_network_duty(r["t_s"], r["q_total_w"], t_amb, net)
            out3 = run_network_duty(r3["t_s"], r3["q_total_w"], t_amb, net)
            repeat.append({
                "scheme": key, "t_amb_c": t_amb,
                "t_max_1cycle_c": float(out1["t_core_c"].max()),
                "t_max_3cycle_c": float(out3["t_core_c"].max()),
                "accumulation_k": float(out3["t_core_c"].max() - out1["t_core_c"].max()),
            })
            print(f"   {t_amb:4.0f}C {scheme.name:20s} 1循环={out1['t_core_c'].max():5.1f} C  "
                  f"3循环={out3['t_core_c'].max():5.1f} C  累积={out3['t_core_c'].max() - out1['t_core_c'].max():+4.1f} K")

    payload = {
        "provenance": US06_PROVENANCE,
        "profile_stats": {
            "mean_c": float(np.mean(c_p)), "peak_c": float(np.max(c_p)),
            "regen_min_c": float(np.min(c_p)), "duration_s": 600.0,
        },
        "dfn_q": {"q_mean_w": r["q_mean_w"], "q_peak_w": r["q_peak_w"]},
        "matrix": rows,
        "repeat_stress": repeat,
        "dynamic_vs_constant": equiv,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (MET / "07_duty.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # ---------------- 图 ----------------
    plotting.setup_style()

    # 7a: 电流与产热
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    ax[0].plot(r["t_s"], r["c_rate"], color="#1D4ED8", lw=1.2)
    ax[0].axhline(0, color="#111827", lw=0.7)
    ax[0].set_ylabel("倍率 (C，放电为正)")
    ax[0].set_title(f"US06 动态工况（{US06_PROVENANCE[:30]}…）")
    ax[1].plot(r["t_s"], r["q_total_w"], color="#B45309", lw=1.2, label="瞬时产热")
    ax[1].axhline(r["q_mean_w"], color="#111827", ls="--", lw=1,
                  label=f"均值 {r['q_mean_w']:.2f} W")
    ax[1].set_xlabel("时间 (s)"); ax[1].set_ylabel("产热 (W)")
    ax[1].legend(fontsize=8)
    plotting.save(fig, "07a_duty_profile.png")

    # 7b: 矩阵
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    schemes = list(SCHEMES)
    xs = np.arange(len(schemes))
    for j, t_amb in enumerate((25.0, 35.0, 45.0)):
        sub = {row["scheme"]: row for row in rows if row["t_amb_c"] == t_amb}
        vals = [sub[s]["t_max_c"] for s in schemes]
        ax.bar(xs + (j - 1) * 0.26, vals, width=0.26, label=f"{t_amb:g} °C",
               color=["#6B7280", "#B45309", "#DC2626"][j])
    ax.axhline(45.0, color="#111827", ls="--", lw=1, label="45 °C 上限")
    ax.set_xticks(xs)
    ax.set_xticklabels([SCHEMES[s].name for s in schemes], fontsize=8)
    ax.set_ylabel("US06 单循环最高温度 (°C)")
    ax.set_title("动态工况（US06）冷却方案选型矩阵")
    ax.legend(fontsize=8)
    plotting.save(fig, "07b_duty_matrix.png")

    print(f"完成，用时 {payload['elapsed_s']} s")


if __name__ == "__main__":
    main()
