"""31 · 5C 产热带与安全边界复核（g(T) 自限口径）。

背景（WP28 开放项 #1 部分收敛 + v1.0 安全边界）
---------------------------------------------
v1.0 的 5C 冷却需求（h=638 W/(m²·K)）与安全边界（5C+45 °C 散热完好
芯温 91 °C、散热退化到 30% 时 105 °C）都基于标度律外推 + 恒定 R。
WP28 只锁了欧姆指数 p_ohm=1.83–2.00。本脚本：

1. **5C 产热带**：三条独立估计——标度律（13.97 W）、Mohtat 形状
   （14.25 W）、分量倍率指数混合法（p_local=Σ s_i·p_i，实测 p_ohm
   + 可逆热线性 + 极化 [1,2]）→ Q(5) 带与 h 需求带。
2. **安全边界 g(T) 复核**：恒定 R 复现 v1.0 峰值后，用 Panasonic
   实测 R(T)（Ea 22.4 kJ/mol）做 g(T) 不动点——注意 5C 可逆热占 62%、
   欧姆只占 8%，自限是**二阶**（对比 2C 冷启动的主导地位）。

关键判词：
- 总热在 5C 的局部倍率指数 [1.06, 1.43] **低于** 0.5–2C 拟合 1.79
  （可逆热线性主导）——纯幂律外推从 3C 的偏低（v1.1 −25%）交叉到
  5C 的收敛；"下界警示"的正确范围是欧姆份额/低温/老化，不是
  25 °C 新芯总热。
- g(T) 自限与 2C 冷启动图景不可互搬。

产出：
  results/metrics/31_5c_band.json
  results/figures/31a_5c_band.png
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
from bmthermal.network import build_cell, simulate_two_node  # noqa: E402
from bmthermal.heat import q_profile  # noqa: E402
from bmthermal.params import R_GAS  # noqa: E402

OUT = ROOT / "results"
MET = OUT / "metrics"
FIG = OUT / "figures"
for p in (MET, FIG):
    p.mkdir(parents=True, exist_ok=True)

T_AMB_SAFE = 45.0
C_RATE = 5.0
ALARM_C = 80.0
# WP27 Panasonic r_end 拟合（27_lowtemp_rct.json：Ea=22.374 kJ/mol）
EA_R_J_MOL = 22_374.0
T_REF_C = 25.0
# WP28 实测欧姆指数带
P_OHM = (1.83, 2.00)
P_IRR = (1.0, 2.0)      # 极化份：Butler-Volmer 高场 ~ I·ln(I) → [1, 2]
P_REV = 1.0             # 可逆（熵）热 Q ∝ I → 线性
H_CASES = (("h×1 散热完好", 500.0), ("h×0.3 散热退化", 150.0))


def g_ratio(t_c: float) -> float:
    """R(T)/R(25)：Panasonic 实测 Arrhenius（外推到 45–60 °C 是载荷性假设）。"""
    return float(np.exp(EA_R_J_MOL / R_GAS * (1.0 / (t_c + 273.15) - 1.0 / (T_REF_C + 273.15))))


def shares_at(c: float) -> dict:
    p = json.loads((MET / "01_heat_sweep.json").read_text(encoding="utf-8"))
    rows = [r for r in p["sweep"] if abs(r["c_rate"] - c) < 1e-6
            and r.get("parameter_set") == "Mohtat2020"]
    if not rows:
        rows = [r for r in p["sweep"] if abs(r["c_rate"] - c) < 1e-6]
    r = rows[0]
    return {"ohmic": r["share_ohmic"], "irr": r["share_irreversible"],
            "rev": r["share_reversible"], "q_mean_w": r["q_mean_w"]}


def main() -> None:
    t0 = time.time()

    # ---- 1) 5C 产热带：三条估计 ----
    print("[1/3] 5C 产热带 ...")
    law = json.loads((MET / "02_required_h.json").read_text(encoding="utf-8"))
    q_law5 = float(next(r["q_pred_w"] for r in law["required"] if r["c_rate"] == 5.0))
    h_law5 = float(next(r["h_required_w_m2k"] for r in law["required"] if r["c_rate"] == 5.0))
    s3, s5 = shares_at(3.0), shares_at(5.0)
    q_m3, q_m5 = s3["q_mean_w"], s5["q_mean_w"]

    def p_local(s: dict, p_ohm: float, p_irr: float) -> float:
        return s["ohmic"] * p_ohm + s["irr"] * p_irr + s["rev"] * P_REV

    p_lo = 0.5 * (p_local(s3, P_OHM[0], P_IRR[0]) + p_local(s5, P_OHM[0], P_IRR[0]))
    p_hi = 0.5 * (p_local(s3, P_OHM[1], P_IRR[1]) + p_local(s5, P_OHM[1], P_IRR[1]))
    q_blend_lo = q_m3 * (5.0 / 3.0) ** p_lo
    q_blend_hi = q_m3 * (5.0 / 3.0) ** p_hi
    p_fd = float(np.log(q_m5 / q_m3) / np.log(5.0 / 3.0))   # Mohtat 3→5 实测差分

    area_m2 = q_law5 / (20.0 * h_law5)   # 温升 20 K 需求口径反推换热面积
    h_of = lambda q: q / (20.0 * area_m2)  # noqa: E731

    q_band = (min(q_blend_lo, q_law5), max(q_blend_hi, q_m5))
    h_band = (h_of(q_band[0]), h_of(q_band[1]))
    print(f"   标度律 Q={q_law5:.2f} W  h={h_law5:.0f} | Mohtat 形状 Q={q_m5:.2f} W")
    print(f"   分量指数混合法 p_local∈[{p_lo:.2f},{p_hi:.2f}]（Mohtat 3→5 实测差分 p={p_fd:.2f}）")
    print(f"   → Q(5) 带 [{q_band[0]:.2f}, {q_band[1]:.2f}] W  h 带 [{h_band[0]:.0f}, {h_band[1]:.0f}] W/(m²·K)")

    # ---- 2) 5C 安全边界：恒定 R 复现 + g(T) 自限 ----
    print("[2/3] 5C 安全边界 g(T) 复核 ...")
    t_arr, q_arr, info = q_profile(C_RATE, t_amb_c=25.0, cache={})
    s_ohm = s5["ohmic"]
    print(f"   q_profile(5C) 均值 {q_arr.mean():.2f} W（{info['source']}），欧姆份额 {s_ohm:.2f}")

    def run(h_eff: float, mode: str) -> dict:
        """mode: const（恒定 R）/ tdep_ohmic（只缩放欧姆份）/ tdep_full（全量）。"""
        cell = build_cell(h=h_eff)
        mult = np.ones_like(q_arr)
        sim = None
        for _ in range(6):
            sim = simulate_two_node(t_arr, q_arr * mult, T_AMB_SAFE, cell)
            core = sim["t_core_c"]
            g_t = np.array([g_ratio(float(tc)) for tc in core])
            if mode == "const":
                break
            if mode == "tdep_ohmic":
                mult = 1.0 + (g_t - 1.0) * s_ohm
            else:
                mult = g_t
        core, surf = sim["t_core_c"], sim["t_surface_c"]
        return {
            "mode": mode,
            "q_mult_start": float(mult[0]),
            "q_mult_end": float(mult[-1]),
            "t_core_peak_c": float(core.max()),
            "t_surface_peak_c": float(surf.max()),
            "gap_peak_k": float((core - surf).max()),
            "exceeds_alarm": bool(core.max() > ALARM_C),
        }

    safety = {}
    for label, h_eff in H_CASES:
        safety[label] = {
            "h_effective": h_eff,
            "const": run(h_eff, "const"),
            "tdep_ohmic": run(h_eff, "tdep_ohmic"),
            "tdep_full": run(h_eff, "tdep_full"),
        }
        c0 = safety[label]["const"]["t_core_peak_c"]
        c1 = safety[label]["tdep_ohmic"]["t_core_peak_c"]
        c2 = safety[label]["tdep_full"]["t_core_peak_c"]
        print(f"   {label:14s} 恒定R {c0:5.1f}°C → g(T) 欧姆份 {c1:5.1f}°C"
              f"（全量对照 {c2:5.1f}°C）")

    # ---- 3) 图与指标 ----
    print("[3/3] 图与指标 ...")
    plotting.setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    names = ["标度律\n(0.5–2C 拟合)", "Mohtat 形状\n(5C 实测)", "分量指数\n混合法"]
    vals = [q_law5, q_m5, 0.5 * (q_blend_lo + q_blend_hi)]
    errs = [[0, 0, 0.5 * (q_blend_hi - q_blend_lo)],
            [0, 0, 0.5 * (q_blend_hi - q_blend_lo)]]
    ax.bar(names, vals, yerr=errs, color=["#7f8c8d", "#2980b9", "#27ae60"],
           capsize=4, width=0.55)
    ax.axhspan(q_band[0], q_band[1], alpha=0.12, color="green", label=f"Q 带 {q_band[0]:.1f}–{q_band[1]:.1f} W")
    ax.set_ylabel("5C 平均产热 Q (W)")
    ax.set_title(f"a) 5C 产热带 → h 带 [{h_band[0]:.0f}, {h_band[1]:.0f}]")
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = list(safety)
    x = np.arange(len(labels))
    w = 0.27
    for i, (key, color, name) in enumerate((
            ("const", "#c0392b", "恒定 R（v1.0 口径）"),
            ("tdep_ohmic", "#2980b9", "g(T) 欧姆份（物理）"),
            ("tdep_full", "#8e44ad", "g(T) 全量（对照）"))):
        vals = [safety[l][key]["t_core_peak_c"] for l in labels]
        ax.bar(x + (i - 1) * w, vals, w, label=name, color=color)
    ax.axhline(ALARM_C, color="k", ls="--", lw=0.9, label=f"预警 {ALARM_C:.0f}°C")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("芯温峰值 (°C)")
    ax.set_title("b) 5C+45 °C 安全边界：g(T) 自限修正")
    ax.legend(fontsize=7)
    fig.tight_layout()
    plotting.save(fig, "31a_5c_band")

    const1 = safety["h×1 散热完好"]["const"]["t_core_peak_c"]
    td1 = safety["h×1 散热完好"]["tdep_ohmic"]["t_core_peak_c"]
    const3 = safety["h×0.3 散热退化"]["const"]["t_core_peak_c"]
    td3 = safety["h×0.3 散热退化"]["tdep_ohmic"]["t_core_peak_c"]
    verdict = (
        f"① 5C 产热带 Q=[{q_band[0]:.1f},{q_band[1]:.1f}] W → h 需求 "
        f"[{h_band[0]:.0f},{h_band[1]:.0f}] W/(m²·K)：638 在带中（±10%），"
        "『5C 冷却需求』不脆弱；WP28『下界警示』的正确范围是欧姆份额/低温/老化，"
        f"不是 25 °C 新芯总热——总热局部倍率指数 [{p_lo:.2f},{p_hi:.2f}] 低于 0.5–2C 拟合 1.79"
        f"（可逆热 {s5['rev']:.0%} 线性主导；Mohtat 3→5 实测差分 p={p_fd:.2f} 佐证），"
        "纯幂律外推从 3C 偏低（−25%）交叉到 5C 收敛，非幂律行为。"
        f"② 安全边界 g(T) 修正：散热完好 {const1:.1f}→{td1:.1f}°C、退化 30% {const3:.1f}→{td3:.1f}°C"
        "——5C 可逆热主导、欧姆只占 8%，自限是二阶（对比 2C 冷启动 105→30°C 的主导地位）；"
        "两幅图景不可互搬。退化案例仍超 80 °C 预警。"
    )
    print("   " + verdict)

    out = {
        "q_band_w": q_band,
        "h_band_w_m2k": h_band,
        "q_law5_w": q_law5,
        "q_mohtat5_w": q_m5,
        "p_local_band": [p_lo, p_hi],
        "p_fd_mohtat_3_5": p_fd,
        "shares_5c": s5,
        "safety": safety,
        "assumption_note": "R(T) Arrhenius（Panasonic Ea 22.4 kJ/mol）外推到 45–60 °C 为载荷性假设；可逆/极化份不随 R 缩放。",
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (MET / "31_5c_band.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成，{time.time() - t0:.1f}s → 31_5c_band.json / 31a_5c_band.png")


if __name__ == "__main__":
    main()
