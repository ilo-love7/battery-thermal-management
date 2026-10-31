"""05 · 热-寿命耦合选型（定安 D2，本项目最强记忆点）。

把"选冷却方案"升级为"算寿命账"：同一工况下不同冷却方案的温度历程，
经 Arrhenius 损伤积分折算成相对寿命倍数。

三条输出：
1. 各方案在 1C/3C/5C 下的相对寿命倍数（以自然对流 25 °C 为基准 = 1.0）；
2. Ea 敏感性（30/45/60 kJ/mol）——这个参数会改结论量级，必须一起报；
3. "用平均温度代替温度历程"的系统性偏差（Jensen 不等式项）。

产出：
  results/metrics/05_aging.json
  results/figures/05a_life_ratio.png      寿命倍数对比（主图之一）
  results/figures/05b_mean_temp_bias.png  平均温度偏差
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
from bmthermal.aging import (  # noqa: E402
    EA_GRID,
    damage_integral,
    life_ratio,
    mean_temperature_bias,
)
from bmthermal.heat import q_profile  # noqa: E402
from bmthermal.network import build_cell, simulate_two_node  # noqa: E402
from bmthermal.params import SCHEMES  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MET = OUT / "metrics"
MET.mkdir(parents=True, exist_ok=True)


def run_case(c: float, scheme_key: str, t_amb: float, cache: dict) -> dict:
    t_arr, q_arr, _ = q_profile(c, t_amb_c=t_amb, cache=cache)
    scheme = SCHEMES[scheme_key]
    net = build_cell(h=scheme.h_mid, pcm_extra_capacity_j_k=scheme.pcm_extra_capacity_j_k)
    out = simulate_two_node(t_arr, q_arr, t_amb, net)
    # 用芯温算老化（热电偶在表面，老化由卷芯温度决定——两节点的卖点）
    t_core = out["t_core_c"]
    return {
        "t_s": t_arr, "t_core_c": t_core,
        "t_surface_c": out["t_surface_c"],
        "t_max_core_c": float(t_core.max()),
    }


def main() -> None:
    t0 = time.time()
    cache: dict = {}

    print("[1/3] 方案 × 倍率 的损伤积分与寿命倍数 ...")
    cases = {}
    for c in (1.0, 3.0, 5.0):
        for key in SCHEMES:
            cases[(c, key)] = run_case(c, key, 25.0, cache)

    # 基准：自然对流 1C（最温和常规工况）
    base = cases[(1.0, "natural")]
    d_base = damage_integral(base["t_s"], base["t_core_c"])
    rows = []
    for (c, key), res in cases.items():
        d = damage_integral(res["t_s"], res["t_core_c"])
        lr = life_ratio(d, d_base)
        row = {
            "c_rate": c, "scheme": key, "scheme_name": SCHEMES[key].name,
            "t_max_core_c": res["t_max_core_c"],
            "damage_s": d, "life_ratio_vs_natural_1c": lr,
        }
        # Ea 敏感性
        for ea in EA_GRID:
            d_ea = damage_integral(res["t_s"], res["t_core_c"], ea_j_mol=ea)
            row[f"life_ratio_ea{int(ea/1000)}k"] = life_ratio(d_ea, damage_integral(
                base["t_s"], base["t_core_c"], ea_j_mol=ea))
        rows.append(row)
        print(f"   {c:g}C {SCHEMES[key].name:20s} Tmax={res['t_max_core_c']:5.1f} C  "
              f"寿命倍数={lr:6.3f}")

    print("[2/3] 平均温度偏差（Jensen 项）...")
    bias_rows = []
    for (c, key), res in cases.items():
        if key not in ("natural", "cold_plate"):
            continue
        b = mean_temperature_bias(res["t_s"], res["t_core_c"])
        bias_rows.append({"c_rate": c, "scheme": key, **b})
        print(f"   {c:g}C {key:12s} 平均T={b['t_mean_c']:.2f} C  等效T={b['t_eff_c']:.2f} C  "
              f"偏差={b['bias_k']:+.2f} K  寿命误差={b['life_error_pct']:+.1f}%")

    print("[3/3] 恒流 vs US06 动态的损伤对比 ...")
    from bmthermal.duty import run_network_duty
    from bmthermal.duty import load_us06
    duty_rows = []
    t_p, c_p = load_us06()
    # 用 Q ∝ C² 的准稳态映射把倍率剖面折成产热（峰 1.46C ≈ 2.13×2C 的热）
    from bmthermal.heat import predict_q, q_law
    law = q_law()
    q_dyn = np.array([predict_q(max(abs(cc), 0.02), law) * (1 if cc >= 0 else 0.2) for cc in c_p])
    for key in ("natural", "forced_air", "cold_plate"):
        scheme = SCHEMES[key]
        net = build_cell(h=scheme.h_mid, pcm_extra_capacity_j_k=scheme.pcm_extra_capacity_j_k)
        out = run_network_duty(t_p, q_dyn, 25.0, net)
        d_dyn = damage_integral(t_p, out["t_core_c"])
        # 等平均产热的恒流（0.21C 等效）
        q_avg = float(np.mean(q_dyn))
        t_eq = np.linspace(0, t_p[-1], 200)
        out_eq = run_network_duty(t_eq, np.full_like(t_eq, q_avg), 25.0, net)
        d_eq = damage_integral(t_eq, out_eq["t_core_c"])
        duty_rows.append({
            "scheme": key,
            "t_max_core_c": float(out["t_core_c"].max()),
            "damage_dynamic_s": d_dyn, "damage_mean_equiv_s": d_eq,
            "dynamic_penalty_pct": (d_dyn / d_eq - 1.0) * 100.0,
        })
        print(f"   {key:12s} 动态 Tmax={out['t_core_c'].max():5.1f} C  "
              f"损伤惩罚 +{(d_dyn / d_eq - 1) * 100:.1f}%")

    payload = {
        "baseline": "natural convection @ 1C, 25 °C ambient",
        "ea_grid_j_mol": list(EA_GRID),
        "life_table": rows,
        "mean_temperature_bias": bias_rows,
        "duty_vs_constant": duty_rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (MET / "05_aging.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # ---------------- 图 ----------------
    plotting.setup_style()
    colors = plotting.SCHEME_COLORS

    # 5a: 寿命倍数（3C，各方案，三条 Ea）
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    schemes = list(SCHEMES)
    xs = np.arange(len(schemes))
    width = 0.26
    for j, ea in enumerate(EA_GRID):
        vals = []
        for key in schemes:
            row = next(r for r in rows if r["c_rate"] == 3.0 and r["scheme"] == key)
            vals.append(row[f"life_ratio_ea{int(ea/1000)}k"])
        ax.bar(xs + (j - 1) * width, vals, width=width,
               label=f"Ea={int(ea/1000)} kJ/mol",
               color=["#9CA3AF", "#1D4ED8", "#B45309"][j])
    ax.set_xticks(xs)
    ax.set_xticklabels([SCHEMES[s].name for s in schemes], fontsize=8)
    ax.set_ylabel("相对寿命倍数（基准=自然对流 1C）")
    ax.set_title("3C 工况下冷却方案的寿命账（含 Ea 敏感性）")
    ax.legend(fontsize=8)
    plotting.save(fig, "05a_life_ratio.png")

    # 5b: 平均温度偏差
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    labels = [f"{b['c_rate']:g}C/{b['scheme']}" for b in bias_rows]
    vals = [b["life_error_pct"] for b in bias_rows]
    ax.barh(labels, vals, color="#B45309")
    ax.axvline(0, color="#111827", lw=0.8)
    ax.set_xlabel("用平均温度算寿命的相对误差 (%)")
    ax.set_title("平均温度 vs 损伤等效温度（Jensen 偏差）")
    plotting.save(fig, "05b_mean_temp_bias.png")

    print(f"完成，用时 {payload['elapsed_s']} s")


if __name__ == "__main__":
    main()
