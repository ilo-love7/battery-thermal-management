"""36 · 联合口径 s_cal 敏感性：C5 随识别集收窄的定量前瞻（v1.8, WP36）。

开放项 #4 的定量版（NMC/NCA 干净日历数据未到位前的前瞻）：

联合口径（v1.7 §34）：m_flux=(m_meas−s·r_t)/(1−s)、br3=calendar_bracket
(s, span3, span1)，C5_joint = d1/(d3·m_flux·br3)。s_cal 目前只有识别集
[0, 0.61]（§26），点估计被跨对不自洽挡住。

本脚本回答三个问题：
1. C5_joint 对 s∈[0, 0.65] 的敏感性曲线（LFP/NMC）；
2. **翻转临界 s***：LFP 的 C5_joint=1 在哪个 s（3C 液冷输给 1C 自然的门槛）；
3. **前瞻表**：若未来 NMC/NCA 日历数据给出 s 点估计 x，C5 收窄到多少。

产出：
  results/metrics/36_joint_sens.json
  results/figures/36a_joint_sens.png
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
from bmthermal.calendar import calendar_bracket  # noqa: E402

OUT = ROOT / "results"
MET = OUT / "metrics"
FIG = OUT / "figures"

R_T_SNL = 7248.0 / 9970.0      # SNL LFP 3C/1C dt 中位（§34 同源）
SPAN_1 = 3340.15               # 1C 模型时长 s（scripts/23）
SPAN_3 = 1091.56
S_LO, S_HI = 0.0, 0.65
S_ID_LO, S_ID_HI = 0.0, 0.61   # §26 识别集
S_MAIN = 0.40                  # CS2 主口径 rebase


def load_metrics(name: str) -> dict:
    with open(MET / name, encoding="utf-8") as fh:
        return json.load(fh)


def m_flux(m_meas: float, s: float, r_t: float = R_T_SNL) -> float:
    """m_flux=(m_meas−s·r_t)/(1−s)，带数值护栏。"""
    if s >= 1.0:
        return float("nan")
    v = (m_meas - s * r_t) / (1.0 - s)
    return float(max(v, 0.05))


def c5_joint(d1: float, d3: float, m_meas: float, s: float) -> dict:
    mf = m_flux(m_meas, s)
    br3 = calendar_bracket(s, SPAN_3, SPAN_1)
    ratio = d1 / (d3 * mf * br3) if (d3 * mf * br3) > 0 else float("nan")
    return {"s": float(s), "m_flux_3c": mf, "bracket_3c": float(br3),
            "c5_joint": float(ratio)}


def main() -> None:
    t0 = time.time()
    print("[1/3] 载入 29/34 指标 ...")
    m29 = load_metrics("29_rate_effect.json")
    m34 = load_metrics("34_life_ledger.json")

    # d1/d3 从 34 ledger 行取（热学部分与体系无关）
    row0 = next(r for r in m34["ledger_rows"]
                if r["chem"] == "LFP" and r["c_rate"] == 1.0
                and r["scheme"] == "natural")
    row3 = next(r for r in m34["ledger_rows"]
                if r["chem"] == "LFP" and r["c_rate"] == 3.0
                and r["scheme"] == "cold_plate")
    d1, d3 = row0["d_s_25c_s"], row3["d_s_25c_s"]

    m_map: dict[tuple[str, float], dict] = {}
    for p in m29["rate_points"]:
        if abs(p["temp_c"] - 25.0) > 1e-6:
            continue
        m_map[(p["chem"], float(p["c_dis"]))] = p

    chems = ("LFP", "NMC")
    m3 = {c: float(m_map[(c, 3.0)]["m_per_efc"]) for c in chems}
    roles = {c: m_map[(c, 3.0)]["role"] for c in chems}
    print(f"   d1={d1:.0f} s·d3={d3:.0f} s·m3 LFP={m3['LFP']:.2f} NMC={m3['NMC']:.2f}")

    # WP29 口径参考线（m_meas×热损伤，不带日历换算）
    wp29 = {}
    for c in chems:
        rr = next(r for r in m34["c5_joint_mflux_bracket"]["rows"]
                  if r["chem"] == c and r["s_tag"] == "main_0.40")
        wp29[c] = rr["c5_joint_ratio"]  # 联合口径 @0.40（对照用）
    wp29_convention = {"LFP": 2.00, "NMC": 7.16}   # §43 WP29 口径（m_meas 直用）

    # ---- [2/3] 扫描 ----
    print("[2/3] s 扫描与临界点 ...")
    s_grid = np.round(np.arange(S_LO, S_HI + 1e-9, 0.01), 4)
    curves = {c: [c5_joint(d1, d3, m3[c], float(s)) for s in s_grid]
              for c in chems}

    # 临界 s*：C5_joint=1（线性插值）
    crit = {}
    for c in chems:
        ys = np.array([r["c5_joint"] for r in curves[c]])
        ss = np.array([r["s"] for r in curves[c]])
        below = np.nonzero(ys < 1.0)[0]
        if below.size == 0:
            crit[c] = {"s_star": float("nan"),
                       "note": "全扫描区间 C5>1，不翻转"}
        else:
            i = int(below[0])
            if i == 0:
                crit[c] = {"s_star": 0.0, "note": "s=0 即 C5<1"}
            else:
                y0, y1 = ys[i - 1], ys[i]
                s0, s1 = ss[i - 1], ss[i]
                s_star = float(s0 + (1.0 - y0) / (y1 - y0) * (s1 - s0))
                crit[c] = {"s_star": s_star,
                           "note": f"C5_joint 跨 1 的临界 s（{c}）"}
        print(f"   {c}: s* = {crit[c]['s_star']}  ({crit[c]['note']})")

    # 前瞻表：若点估计落在 x，C5 及 ±0.05 窗
    forecast = []
    for x in (0.0, 0.20, S_MAIN, 0.55, S_ID_HI):
        for c in chems:
            cen = c5_joint(d1, d3, m3[c], x)["c5_joint"]
            lo = c5_joint(d1, d3, m3[c], max(0.0, x - 0.05))["c5_joint"]
            hi = c5_joint(d1, d3, m3[c], min(S_HI, x + 0.05))["c5_joint"]
            forecast.append({
                "chem": c, "s_point": x,
                "c5_at_point": cen,
                "c5_window_pm005": [float(min(lo, hi)), float(max(lo, hi))],
                "inside_identified_set": bool(S_ID_LO <= x <= S_ID_HI),
            })
            print(f"   前瞻 {c} s={x:.2f}: C5={cen:.2f} "
                  f"[{min(lo, hi):.2f}, {max(lo, hi):.2f}]")

    # s→1 奇异性护栏说明（m_flux 发散）
    diverge = []
    for c in chems:
        for s in (0.8, 0.9, 0.95):
            r = c5_joint(d1, d3, m3[c], s)
            diverge.append({"chem": c, **r})

    # ---- [3/3] 图与产物 ----
    print("[3/3] 图与产物 ...")
    plotting.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    for c, color in (("LFP", "#c0392b"), ("NMC", "#2980b9")):
        ss = [r["s"] for r in curves[c]]
        ys = [r["c5_joint"] for r in curves[c]]
        ax.plot(ss, ys, "-", color=color, lw=1.8, label=f"{c} 联合口径")
        ax.axhline(wp29_convention[c], color=color, ls=":", lw=1.1,
                   label=f"{c} WP29 口径 {wp29_convention[c]:.2f}")
    ax.axhline(1.0, color="k", ls="--", lw=0.9, label="打平线 C5=1")
    ax.axvspan(S_ID_LO, S_ID_HI, color="#f6d55c", alpha=0.18, label="识别集 [0, 0.61]")
    if np.isfinite(crit["LFP"]["s_star"]):
        ax.axvline(crit["LFP"]["s_star"], color="#c0392b", ls="--", lw=1.0)
        ax.text(crit["LFP"]["s_star"], ax.get_ylim()[0], f" s*={crit['LFP']['s_star']:.2f}",
                fontsize=8, va="bottom")
    ax.set_xlabel("s_cal（rebase 参考时长）")
    ax.set_ylabel("C5_joint（3C 液冷 / 1C 自然）")
    ax.set_title("a) C5 联合口径 × s_cal")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for c, color in (("LFP", "#c0392b"), ("NMC", "#2980b9")):
        ss = [r["s"] for r in curves[c]]
        mfs = [r["m_flux_3c"] for r in curves[c]]
        ax.plot(ss, mfs, "-", color=color, lw=1.8, label=f"{c} m_flux(3C)")
    brs = [calendar_bracket(float(s), SPAN_3, SPAN_1) for s in s_grid]
    ax.plot(s_grid, brs, "k--", lw=1.4, label="bracket(3C)")
    ax.axvspan(S_ID_LO, S_ID_HI, color="#f6d55c", alpha=0.18)
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("s_cal")
    ax.set_ylabel("无量纲倍率")
    ax.set_title("b) 拆解：m_flux 与 bracket（s→1 发散护栏）")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    xs = np.arange(len(forecast) // 2)
    lfp = [r for r in forecast if r["chem"] == "LFP"]
    nmc = [r for r in forecast if r["chem"] == "NMC"]
    w = 0.35
    ax.bar(xs - w / 2, [r["c5_at_point"] for r in lfp], w, color="#c0392b",
           label="LFP")
    ax.bar(xs + w / 2, [r["c5_at_point"] for r in nmc], w, color="#2980b9",
           label="NMC")
    ax.axhline(1.0, color="k", ls="--", lw=0.9)
    ax.set_xticks(xs, [f"s={r['s_point']:.2f}" for r in lfp], fontsize=8)
    ax.set_ylabel("C5_joint（点估计）")
    ax.set_title("c) 前瞻：点估计若落在 s=x")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    s_draw = np.linspace(0, 0.95, 200)
    for c, color in (("LFP", "#c0392b"), ("NMC", "#2980b9")):
        mfs = [m_flux(m3[c], float(s)) for s in s_draw]
        ax.plot(s_draw, mfs, "-", color=color, lw=1.8, label=f"{c}")
    ax.axvline(S_ID_HI, color="#8e44ad", ls=":", lw=1.2, label="识别集上界 0.61")
    ax.set_xlabel("s_cal")
    ax.set_ylabel("m_flux(3C)")
    ax.set_title("d) m_flux(s)：s→1 数值发散（护栏 0.05）")
    ax.legend(fontsize=8)

    fig.suptitle("36a · 联合口径 s_cal 敏感性：C5 收窄前瞻", fontsize=12)
    fig.tight_layout()
    plotting.save(fig, FIG / "36a_joint_sens.png")
    plt.close(fig)

    lfp_crit = crit["LFP"]["s_star"]
    nmc_hi = max(r["c5_joint"] for r in curves["NMC"])
    nmc_lo = min(r["c5_joint"] for r in curves["NMC"])
    verdict = (
        f"联合口径 C5 对 s_cal 的敏感性收口：识别集 [0, 0.61] 内 C5 带 "
        f"LFP [{min(r['c5_joint'] for r in curves['LFP']):.2f}, "
        f"{max(r['c5_joint'] for r in curves['LFP']):.2f}]、"
        f"NMC [{nmc_lo:.2f}, {nmc_hi:.2f}]；"
        f"LFP 翻转临界 s*={lfp_crit:.2f}（C5=1，3C 液冷打平 1C 自然）"
        f"{'，识别集内即翻转' if np.isfinite(lfp_crit) and lfp_crit <= S_ID_HI else ''}。"
        f"NMC 全区间 C5>1（m=0.64 助推）。未来日历数据若给出 s 点估计，"
        f"C5 可收窄到 ±0.05 窗（前瞻表）；NCA m(3C) 仍 no_data 不进主结论。"
        f"m_flux 在 s→1 发散是公式奇异性，识别集外仅作护栏演示。"
    )
    metrics = {
        "dataset": "WP29 m(c) × WP34 热损伤 × §26 s_cal 识别集（组合口径，无新数据）",
        "inputs": {
            "d_1c_natural_s": d1, "d_3c_cold_s": d3,
            "m_3c": m3, "m_3c_role": roles,
            "r_t_snl_3c_1c": R_T_SNL, "span_1c_s": SPAN_1, "span_3c_s": SPAN_3,
            "identified_set": [S_ID_LO, S_ID_HI], "s_main": S_MAIN,
            "wp29_convention_c5": wp29_convention,
        },
        "curves": {c: curves[c] for c in chems},
        "critical_s": crit,
        "forecast_table": forecast,
        "divergence_guard": diverge,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    with open(MET / "36_joint_sens.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=1)

    print(f"done in {metrics['elapsed_s']:.0f}s -> 36_joint_sens.json / 36a_joint_sens.png")
    print("verdict:", verdict[:240])


if __name__ == "__main__":
    main()
