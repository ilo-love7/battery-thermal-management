"""30 · 模组级寿命一致性：温差 → 损伤散布 → 最差芯 EOL 惩罚。

背景（WP24 模组控制 + WP29 倍率效应）
------------------------------------
WP24 给了 12 芯冷板的温差结构（设计流量 ΔT_max≈0.5 K、低流量 1.5 K、
沿程梯度主导、布点=合规裕度）。本脚本把温差翻译成**寿命一致性**：
按 Arrhenius 损伤（Ea=30/45/60 kJ/mol 敏感性档）算芯间每 EFC 损伤散布，
模组 EOL 由最差芯决定 → 一致性惩罚 = k(T_worst)/<k(T)> − 1。

口径：
- 模组内倍率相同 → m(c) 消掉（WP29 结论），一致性只由温度场决定；
- 稳态解（steady_state_cells）给的是保守-中性口径：控制瞬态（WP24
  damage_max_s）会叠加占空比波动，但稳态散布是下界结构；
- 布点/流量的价值用"惩罚变化"表达（与 §32 合规裕度口径一致）。

产出：
  results/metrics/30_module_life.json
  results/figures/30a_module_life.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal import plotting  # noqa: E402
from bmthermal.aging import EA_GRID, arrhenius_factor  # noqa: E402
from bmthermal.heat import predict_q, q_law  # noqa: E402
from bmthermal.module import ModuleSpec  # noqa: E402
from bmthermal.module_control import steady_state_cells  # noqa: E402

OUT = ROOT / "results"
MET = OUT / "metrics"
FIG = OUT / "figures"
for p in (MET, FIG):
    p.mkdir(parents=True, exist_ok=True)

BASE_C = 2.0
SIGMA_Q = 0.08          # WP24 产热散布
MISMATCH = 0.15         # WP24 +15% 半模组失配
FLOWS = {"design_2.0lpm": 2.0, "low_0.25lpm": 0.25}


def case_cells(spec: ModuleSpec, q_w: np.ndarray, ea: float) -> dict:
    """一组 q 分布 → 温度场与损伤散布。"""
    t = steady_state_cells(spec, q_w, spec.h_w_m2k)
    k = arrhenius_factor(t, ea)
    k_mean = float(k.mean())
    i_worst = int(np.argmax(k))
    return {
        "t_cells_c": t.tolist(),
        "delta_t_max_k": float(t.max() - t.min()),
        "t_max_c": float(t.max()),
        "k_mean": k_mean,
        "k_worst": float(k.max()),
        "worst_idx": i_worst,
        "damage_ratio_worst_mean": float(k.max() / k_mean),
        "eol_penalty_pct": float((k.max() / k_mean - 1.0) * 100.0),
        "damage_rel_cells": (k / k_mean).tolist(),
    }


def main() -> None:
    t0 = time.time()
    law = q_law(t_amb_c=25.0)
    q_base = float(predict_q(BASE_C, law))
    print(f"[1/3] 基准产热 {q_base:.2f} W/芯（{BASE_C}C 标度律）")

    rng = np.random.default_rng(42)
    n = 12
    q_uniform = np.full(n, q_base)
    q_spread = q_base * (1.0 + SIGMA_Q * rng.standard_normal(n))
    q_mismatch = q_spread.copy()
    q_mismatch[: n // 2] *= 1.0 + MISMATCH   # 上游半模组 +15%

    cases_cfg = {
        "design/uniform": (FLOWS["design_2.0lpm"], q_uniform),
        "design/spread8%": (FLOWS["design_2.0lpm"], q_spread),
        "design/mismatch+15%": (FLOWS["design_2.0lpm"], q_mismatch),
        "low/uniform": (FLOWS["low_0.25lpm"], q_uniform),
        "low/spread8%": (FLOWS["low_0.25lpm"], q_spread),
        "low/mismatch+15%": (FLOWS["low_0.25lpm"], q_mismatch),
    }

    print("[2/3] 稳态温度场与损伤散布 ...")
    results = {}
    for name, (flow, q_w) in cases_cfg.items():
        spec = ModuleSpec(flow_lpm=flow, t_in_c=25.0)
        per_ea = {str(int(ea / 1000)): case_cells(spec, q_w, ea) for ea in EA_GRID}
        results[name] = {
            "flow_lpm": flow,
            "q_mean_w": float(q_w.mean()),
            "coolant_rise_k": spec.coolant_rise_k(float(q_w.sum())),
            "by_ea": per_ea,
        }
        e45 = per_ea["45"]
        print(f"   {name:22s} ΔT={e45['delta_t_max_k']:.2f} K  "
              f"惩罚(Ea45)={e45['eol_penalty_pct']:.1f}%  "
              f"最差芯 idx={e45['worst_idx']}")

    # 流量扫掠（mismatch 场景）：惩罚 vs 流量
    print("   流量扫掠 ...")
    flow_sweep = []
    spec0 = ModuleSpec(t_in_c=25.0)
    for flow in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        spec = ModuleSpec(flow_lpm=flow, t_in_c=25.0)
        r = case_cells(spec, q_mismatch, 45000.0)
        flow_sweep.append({
            "flow_lpm": flow,
            "delta_t_max_k": r["delta_t_max_k"],
            "eol_penalty_pct": r["eol_penalty_pct"],
        })
        print(f"   flow={flow:4.1f} lpm  ΔT={r['delta_t_max_k']:.2f} K  "
              f"惩罚={r['eol_penalty_pct']:.1f}%")

    # WP24 控制瞬态参照（damage_max_s 的散布无法从 JSON 还原，只引用 max）
    ref24 = None
    try:
        d24 = json.loads((MET / "24_module_control.json").read_text(encoding="utf-8"))
        ref24 = {k: v.get("damage_max_s") for k, v in d24["results"].items()
                 if v.get("damage_max_s") is not None}
    except Exception:
        pass

    print("[3/3] 图与指标 ...")
    plotting.setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    idx = np.arange(12)
    for name, color, off in (("design/uniform", "#7f8c8d", -0.27),
                             ("low/spread8%", "#c0392b", 0.0),
                             ("low/mismatch+15%", "#2980b9", 0.27)):
        rel = results[name]["by_ea"]["45"]["damage_rel_cells"]
        ax.bar(idx + off, rel, 0.27, label=name, color=color)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("电芯序号（沿流向）")
    ax.set_ylabel("每 EFC 损伤（模组均值=1）")
    ax.set_title("a) 芯间损伤散布（Ea=45 kJ/mol）")
    ax.legend(fontsize=7)

    ax = axes[1]
    eas = [int(ea / 1000) for ea in EA_GRID]
    for name in ("design/spread8%", "low/spread8%", "low/mismatch+15%"):
        vals = [results[name]["by_ea"][str(e)]["eol_penalty_pct"] for e in eas]
        ax.plot(eas, vals, "o-", label=name)
    ax.set_xlabel("损伤活化能 Ea (kJ/mol)")
    ax.set_ylabel("最差芯 EOL 惩罚 (%)")
    ax.set_title("b) 惩罚 vs Ea 敏感性")
    ax.legend(fontsize=7)

    ax = axes[2]
    fl = [r["flow_lpm"] for r in flow_sweep]
    pen = [r["eol_penalty_pct"] for r in flow_sweep]
    ax.plot(fl, pen, "o-", color="#8e44ad")
    ax.set_xscale("log")
    ax.set_xlabel("流量 (L/min，对数轴)")
    ax.set_ylabel("最差芯 EOL 惩罚 (%)")
    ax.set_title("c) 惩罚 vs 流量（mismatch+15%）")
    fig.tight_layout()
    plotting.save(fig, "30a_module_life")

    d45 = results["low/mismatch+15%"]["by_ea"]["45"]
    verdict = (
        f"模组寿命一致性惩罚很小：设计流量 ΔT≲0.5 K → 惩罚 <1%（Ea=45）；"
        f"低流量+15% 失配 ΔT={d45['delta_t_max_k']:.2f} K → 惩罚 {d45['eol_penalty_pct']:.1f}%，"
        f"Ea=60 时上限 {results['low/mismatch+15%']['by_ea']['60']['eol_penalty_pct']:.1f}%。"
        "WP24 的布点/流量选择对**合规**（超限）是 1.5 K 量级、对**寿命一致性**只是 % 级——"
        "一致性不是布点优化的理由，合规裕度才是（与 §32 判词一致）。"
    )
    print("   " + verdict)

    out = {
        "base_c": BASE_C,
        "q_base_w": q_base,
        "sigma_q": SIGMA_Q,
        "mismatch": MISMATCH,
        "results": results,
        "flow_sweep": flow_sweep,
        "wp24_damage_max_s_ref": ref24,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (MET / "30_module_life.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成，{time.time() - t0:.1f}s → 30_module_life.json / 30a_module_life.png")


if __name__ == "__main__":
    main()
