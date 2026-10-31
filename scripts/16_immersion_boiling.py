"""scripts/16_immersion_boiling.py —— 浸没式两相边界与等效 h 适用域（v1.2 深化 3）。

问题：v1.0–v1.1 的浸没式统一折算为等效 h=1000–2000（单相口径）。
这个假设什么时候失效？—— 壁温达到沸点（须换两相曲线）或局部热流撞 CHF。

产出：results/metrics/16_immersion.json / results/figures/16a_boiling_boundary.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmthermal import plotting
from bmthermal.boiling import (
    FLUID,
    chf_zuber,
    effective_h_boiling,
    hotspot_factor_to_chf,
    immersion_validity,
    nucleate_boiling_curve,
)
from bmthermal.params import CELL

OUT = Path(__file__).resolve().parents[1] / "results"
METRICS = Path(__file__).resolve().parents[1] / "results" / "metrics" / "01_heat_sweep.json"


def q_law(c_rate: float) -> float:
    try:
        law = json.loads(METRICS.read_text(encoding="utf-8"))["power_law"]
        k, alpha = law["k"], law["alpha"]
    except Exception:
        k, alpha = 0.7821775390520517, 1.7909636972736782
    return k * abs(c_rate) ** alpha


def main():
    t0 = time.time()
    area = CELL.surface_area_m2
    q_chf = chf_zuber(FLUID)
    rates = (0.5, 1.0, 2.0, 3.0, 5.0)

    nominal = []       # h_sp = 1500（方案标称）
    degraded = []      # h_sp = 200（循环变差）
    poor = []          # h_sp = 50（接近自然对流的池）
    for c in rates:
        q = q_law(c)
        nominal.append(immersion_validity(q, area, h_sp=1500.0, t_fluid_c=25.0))
        degraded.append(immersion_validity(q, area, h_sp=200.0, t_fluid_c=25.0))
        poor.append(immersion_validity(q, area, h_sp=50.0, t_fluid_c=25.0))

    # 热点放大：撞 CHF 需要多大放大倍数
    hotspot = [
        {"c_rate": c, "q_avg_wm2": q_law(c) / area, "factor_to_chf": hotspot_factor_to_chf(q_law(c), area)}
        for c in rates
    ]

    # 反解：5C 下壁温到沸点的临界 h_sp
    q5 = q_law(5.0) / area
    h_sp_onset_5c = q5 / (FLUID.t_sat_c - 25.0)
    q3 = q_law(3.0) / area
    h_sp_onset_3c = q3 / (FLUID.t_sat_c - 25.0)

    # 核态沸腾有效 h（若真的沸腾，h_eff 是多少）
    boiling_h = [effective_h_boiling(q_law(c) / area) for c in rates]

    findings = [
        (
            f"CHF（Zuber 大平板池沸腾）= {q_chf/1e4:.1f} W/cm²，"
            f"与 FC-72 类介电体文献量级（11–15 W/cm²）一致；"
            f"电芯平均热流 5C 仅 {q5:.0f} W/m²，CHF 裕度 {q_chf/q5:.0f}×——"
            "CHF 不是电芯级约束"
        ),
        (
            f"等效 h=1500（单相标称）：全倍率壁温 ≤ "
            f"{max(r['t_wall_single_phase_c'] for r in nominal):.1f} °C < T_sat 56 °C，"
            "单相假设成立；"
            f"循环退化到 h_sp=50 时 5C 壁温 {poor[-1]['t_wall_single_phase_c']:.1f} °C 逼近沸点"
        ),
        (
            f"失效边界（反解）：3C 下 h_sp < {h_sp_onset_3c:.0f}、"
            f"5C 下 h_sp < {h_sp_onset_5c:.0f} W/(m²·K) 时壁温达沸点，"
            "必须换两相模型——即'循环丧失'而不是'热流过大'触发两相"
        ),
        (
            f"局部热点放大到 {hotspot[-1]['factor_to_chf']:.0f}× 才撞 CHF；"
            "两相设计的必要性来自**均匀性与循环可靠性**，不是平均热流"
        ),
        (
            "若进入核态沸腾，h_eff 是负荷的函数（3C 约 "
            f"{boiling_h[3]['h_eff']:.0f} W/(m²·K)），不再是常数 1500——"
            "这就是'等效 h'表达在两相区失效的机制"
        ),
    ]

    out = {
        "fluid": FLUID.name,
        "fluid_note": "文献典型值（FC-72/Novec 类），非特定牌号实测",
        "q_chf_wm2": q_chf,
        "cell_area_m2": area,
        "nominal_h_sp_1500": nominal,
        "degraded_h_sp_200": degraded,
        "poor_h_sp_50": poor,
        "hotspot_to_chf": hotspot,
        "onset_threshold_h_sp": {"3c": h_sp_onset_3c, "5c": h_sp_onset_5c},
        "boiling_h_eff": dict(zip((str(r) for r in rates), boiling_h)),
        "findings": findings,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    with open(OUT / "metrics" / "16_immersion.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 图 ----
    fig, axes = plotting.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    dt_e = np.linspace(0.1, 30.0, 200)
    q_nb = nucleate_boiling_curve(dt_e)
    mask = q_nb <= q_chf * 1.5
    ax.plot(dt_e[mask], q_nb[mask] / 1e4, color="#c0392b", lw=2, label="Rohsenow 核态沸腾")
    ax.axhline(q_chf / 1e4, color="k", ls="--", lw=1.5, label=f"CHF {q_chf/1e4:.0f} W/cm²")
    colors = ("#7f8c8d", "#2980b9", "#27ae60", "#e67e22", "#8e44ad")
    for c, col in zip(rates, colors):
        q_avg = q_law(c) / area
        # 单相工作点：等效过热度按沸腾曲线反查（画在曲线上的等价位置）
        bo = effective_h_boiling(q_avg)
        if bo["regime"] == "nucleate_boiling":
            ax.scatter([bo["dt_e_k"]], [q_avg / 1e4], color=col, s=70, zorder=3, edgecolor="k", lw=0.3, label=f"{c:g}C 平均热流")
    ax.set_yscale("log")
    ax.set_xlabel("壁面过热度 ΔT_e [K]")
    ax.set_ylabel("热流密度 [W/cm²]")
    ax.set_title("沸腾曲线与电芯热流工作点")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for series, label, col in (
        (nominal, "h_sp=1500（标称）", "#27ae60"),
        (degraded, "h_sp=200（循环变差）", "#e67e22"),
        (poor, "h_sp=50（弱循环）", "#c0392b"),
    ):
        tw = [r["t_wall_single_phase_c"] for r in series]
        ax.plot(rates, tw, "o-", color=col, lw=2, label=label)
    ax.axhline(FLUID.t_sat_c, color="k", ls="--", lw=1.5, label="T_sat 56 °C")
    ax.axhline(45.0, color="#7f8c8d", ls=":", lw=1, label="45 °C 设计上限")
    ax.set_xlabel("倍率 [C]")
    ax.set_ylabel("单相预测壁温 [°C]")
    ax.set_title("等效 h 假设的失效：壁温何时达沸点")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    plotting.save(fig, "16a_boiling_boundary.png")

    print(json.dumps({"findings": findings, "elapsed_s": out["elapsed_s"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
