"""33 · 表观 Ea 与 β 重标：冷却寿命收益的参数敏感性。

背景（v1.6 开放项 #3 的参数侧）
------------------------------
损伤模型的温度标度 Ea 默认 45 kJ/mol（aging.py，文献 30–60 中位），
反馈模型的内阻增长 β 默认 2.5（feedback.py，文献 [1.5, 3.5]）。
两个参数都直接决定"冷却值多少寿命"：C5（3C 液冷 vs 1C 自然）的
寿命比随 Ea 升、随 β 的反馈漂移微调。SNL 15/25/35 °C 与 CS2 内阻
给了实测对账的机会。

三问三答：
1. **fade 的表观温度敏感度是多少？** SNL 同倍率温度对（n≥2 双侧）
   给 Ea_app：LFP 25→35 干净对 ≈28 kJ/mol（dt 几乎不变，日历时长
   混杂可排除）；NCA 1C ≈23；NMC 温度轴被批次异质性污染（35 °C 组
   反而慢）→ 资格否决。表观 Ea **低于** 默认 45 → 冷却寿命收益在
   45 假设下偏乐观。
2. **β 能重标吗？** SNL 内阻字段 0/61 全空（硬负结果）；CS2 脉冲
   内阻实测 β 中位 0.48（r²≥0.3 组），低于文献 [1.5, 3.5]。
   按 β=0.48 重算反馈放大与 C5 漂移——闭环偏移缩小，§18 的
   +3.9% 漂移是保守上界。
3. **C5 对 Ea 多敏感？** 主口径 s_rebased=0.40 下，Ea 22→60 kJ/mol
   的 C5 闭环比区间；与识别集上界 s=0.61 组合成二维敏感性。

产出：
  results/metrics/33_ea_beta.json
  results/figures/33a_ea_beta.png
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal import plotting  # noqa: E402
from bmthermal.aging import EA_DEFAULT, damage_integral  # noqa: E402
from bmthermal.calendar import calendar_bracket, rebase_share  # noqa: E402
from bmthermal.feedback import (  # noqa: E402
    BETA_DEFAULT, BETA_GRID, FeedbackParams, life_amplification, pair_life_ratio,
)
from bmthermal.heat import q_profile  # noqa: E402
from bmthermal.network import build_cell, simulate_two_node  # noqa: E402
from bmthermal.params import SCHEMES  # noqa: E402
from bmthermal.snl_aging import load_dataset  # noqa: E402

OUT = ROOT / "results"
MET = OUT / "metrics"
CACHE = OUT / "cache"
FIG = OUT / "figures"
for p in (MET, CACHE, FIG):
    p.mkdir(parents=True, exist_ok=True)

SNL_ZIP = Path("D:/项目/02_数据集/01_国际公开集/Sandia_SNL/SNL.zip")
SNL_CSV = CACHE / "snl_summary.csv"
R_GAS = 8.314462618
MIN_N = 2                      # 温度对资格：双侧 n≥2
CS2_BETA_MEDIAN = 0.4758       # scripts/19 实测（r²≥0.3 组）
CS2_BETA_CELLS = (0.3407, 0.5972, 0.3829, 0.5686)
EA_GRID_J = (22_000.0, 28_000.0, EA_DEFAULT, 60_000.0)
S_MAIN = 0.40                  # v1.4 主口径 rebase
S_HI = 0.61                    # v1.5 识别集上界
CS2_S_CLOSED = 0.6928
CS2_DT_FAST_S = 3.1176 * 3600


def ea_from_pair(f1: float, t1_c: float, f2: float, t2_c: float) -> float:
    """Ea = R·ln(f2/f1) / (1/T1 − 1/T2)。"""
    if not (np.isfinite(f1) and np.isfinite(f2)) or f1 <= 0 or f2 <= 0:
        return float("nan")
    t1, t2 = t1_c + 273.15, t2_c + 273.15
    return float(R_GAS * np.log(f2 / f1) / (1.0 / t1 - 1.0 / t2))


def apparent_ea_rows(df: pd.DataFrame) -> list[dict]:
    """SNL 同倍率温度对 → 表观 Ea（资格：双侧 n≥MIN_N）。"""
    rows = []
    g0 = df[(df.soc_lo == 0) & (df.soc_hi == 100)]
    for chem in ("LFP", "NMC", "NCA"):
        rates = sorted(g0[g0.chem == chem].c_dis.unique())
        for c_dis in rates:
            temps = sorted(g0[(g0.chem == chem) & (g0.c_dis == c_dis)].temp_c.unique())
            for t1, t2 in itertools.combinations(temps, 2):
                s1 = g0[(g0.chem == chem) & (g0.c_dis == c_dis) & (g0.temp_c == t1)]
                s2 = g0[(g0.chem == chem) & (g0.c_dis == c_dis) & (g0.temp_c == t2)]
                if len(s1) < MIN_N or len(s2) < MIN_N:
                    continue
                f1, f2 = float(s1.fade_rate_per_efc.mean()), float(s2.fade_rate_per_efc.mean())
                dt1, dt2 = float(s1.dt_s_per_cycle.median()), float(s2.dt_s_per_cycle.median())
                ea = ea_from_pair(f1, t1, f2, t2)
                eligible = bool(ea > 0)   # 负 Ea = 温度轴异质性，不进汇总
                rows.append({
                    "chem": chem, "c_dis": float(c_dis),
                    "t_lo_c": float(t1), "t_hi_c": float(t2),
                    "n_lo": int(len(s1)), "n_hi": int(len(s2)),
                    "fade_lo": f1, "fade_hi": f2,
                    "fade_ratio": float(f2 / f1),
                    "dt_lo_s": dt1, "dt_hi_s": dt2,
                    "dt_ratio": float(dt2 / dt1),
                    "ea_j_mol": ea,
                    "ea_kj_mol": ea / 1000.0,
                    "eligible": eligible,
                    "dt_confounded": bool(abs(dt2 / dt1 - 1.0) > 0.15),
                })
    return rows


def local_d_cycle_factory(c: float, scheme_key: str, params: FeedbackParams,
                          ea_j_mol: float, cache: dict, fn_cache: dict):
    """d_cycle_fn(u)，Ea 可注入（scripts/18 版本固定默认 Ea）。"""
    slope = round(params.q_mult_slope, 9)
    key = (c, scheme_key, slope, ea_j_mol)
    if key in fn_cache:
        return fn_cache[key]
    t_arr, q_arr, _ = q_profile(c, t_amb_c=25.0, cache=cache)
    scheme = SCHEMES[scheme_key]
    net = build_cell(h=scheme.h_mid,
                     pcm_extra_capacity_j_k=scheme.pcm_extra_capacity_j_k)

    def d_cycle_fn(u: float) -> float:
        mult = 1.0 + slope * u
        out = simulate_two_node(t_arr, q_arr * mult, 25.0, net)
        return damage_integral(t_arr, out["t_core_c"], ea_j_mol=ea_j_mol)

    fn_cache[key] = d_cycle_fn
    return d_cycle_fn


def main() -> None:
    t0 = time.time()

    print("[1/4] SNL 表观 Ea ...")
    df = load_dataset(SNL_ZIP, SNL_CSV)
    df = df[df["fade_rate_per_efc"].notna()].copy()
    ea_rows = apparent_ea_rows(df)
    elig = [r for r in ea_rows if r["eligible"]]
    for r in ea_rows:
        flag = "OK " if r["eligible"] else "否决"
        conf = " dt混杂" if r["dt_confounded"] else ""
        print(f"   [{flag}] {r['chem']} {r['c_dis']:g}C {r['t_lo_c']:.0f}->{r['t_hi_c']:.0f}C: "
              f"Ea={r['ea_kj_mol']:+.1f} kJ/mol  ratio={r['fade_ratio']:.2f}  "
              f"n={r['n_lo']}+{r['n_hi']}{conf}")

    # 按体系汇总（只计资格对）
    ea_by_chem = {}
    for chem in ("LFP", "NMC", "NCA"):
        vals = [r["ea_kj_mol"] for r in elig if r["chem"] == chem]
        clean = [r["ea_kj_mol"] for r in elig
                 if r["chem"] == chem and not r["dt_confounded"]]
        ea_by_chem[chem] = {
            "n_eligible_pairs": len(vals),
            "ea_median_kj": float(np.median(vals)) if vals else float("nan"),
            "ea_range_kj": [float(np.min(vals)), float(np.max(vals))] if vals else [],
            "ea_median_dt_clean_kj": float(np.median(clean)) if clean else float("nan"),
            "n_dt_clean": len(clean),
        }
        if vals:
            print(f"   {chem} 表观 Ea 中位 {np.median(vals):.1f} kJ/mol "
                  f"（dt 干净对 {np.median(clean) if clean else float('nan'):.1f}，"
                  f"n={len(clean)}/{len(vals)}）")

    lfp_ea = ea_by_chem["LFP"].get("ea_median_dt_clean_kj")
    if not np.isfinite(lfp_ea):
        lfp_ea = ea_by_chem["LFP"].get("ea_median_kj", float("nan"))

    # ---- β 对账 ----
    print("[2/4] β 对账与反馈重算 ...")
    beta_rows = []
    cache: dict = {}
    fn_cache: dict = {}
    for beta in (CS2_BETA_MEDIAN,) + tuple(BETA_GRID):
        par = FeedbackParams(beta=beta)
        fn_cold = local_d_cycle_factory(3.0, "cold_plate", par, EA_DEFAULT, cache, fn_cache)
        fn_base = local_d_cycle_factory(1.0, "natural", par, EA_DEFAULT, cache, fn_cache)
        pr = pair_life_ratio(fn_cold, fn_base, par)
        ra = life_amplification(fn_cold, par)
        rb = life_amplification(fn_base, par)
        beta_rows.append({
            "beta": float(beta),
            "source": "CS2 实测中位" if abs(beta - CS2_BETA_MEDIAN) < 1e-6
                      else ("文献默认" if beta == BETA_DEFAULT else "文献档"),
            "amp_3c_cold": ra["amplification"],
            "amp_1c_natural": rb["amplification"],
            "c5_open": pr["open_loop_ratio"],
            "c5_closed": pr["closed_loop_ratio"],
            "c5_drift_pct": pr["ratio_drift_pct"],
        })
        print(f"   β={beta:.2f}: A(3C液冷)={ra['amplification']:.3f} "
              f"A(1C自然)={rb['amplification']:.3f}  C5 闭环 {pr['closed_loop_ratio']:.3f} "
              f"漂移 {pr['ratio_drift_pct']:+.2f}%")

    snl_r_scan = {"cells_scanned": 61, "cells_with_resistance": 0,
                  "verdict": "SNL internal_resistance_in_ohm 字段 0/61 全空——"
                             "β 无法从 SNL 实测重标（硬负结果）；"
                             "CS2 脉冲内阻是唯一实测锚，NMC/NCA β 仍停留在文献带。"}

    # ---- C5 × Ea 敏感性 ----
    print("[3/4] C5 × Ea 敏感性 ...")
    t_arr, _, _ = q_profile(1.0, t_amb_c=25.0, cache=cache)
    spans = {}
    for c in (1.0, 3.0):
        ta, _, _ = q_profile(c, t_amb_c=25.0, cache=cache)
        spans[c] = float(ta[-1] - ta[0])
    t_ref = spans[1.0]
    s_rebased = rebase_share(CS2_S_CLOSED, t_ref, CS2_DT_FAST_S)

    ea_grid_rows = []
    for ea in EA_GRID_J:
        par = FeedbackParams(beta=CS2_BETA_MEDIAN)
        for s_tag, s_val in (("main_0.40", S_MAIN), ("id_hi_0.61", S_HI)):
            br = {c: calendar_bracket(s_val, spans[c], t_ref) for c in (1.0, 3.0)}
            fn_c = local_d_cycle_factory(3.0, "cold_plate", par, ea, cache, fn_cache)
            fn_b = local_d_cycle_factory(1.0, "natural", par, ea, cache, fn_cache)
            pr = pair_life_ratio(lambda u: fn_c(u) * br[3.0],
                                 lambda u: fn_b(u) * br[1.0], par)
            ea_grid_rows.append({
                "ea_kj_mol": ea / 1000.0, "s_tag": s_tag, "s_rebased": s_val,
                "open_loop_ratio": pr["open_loop_ratio"],
                "closed_loop_ratio": pr["closed_loop_ratio"],
            })
            print(f"   Ea={ea/1000:.0f} kJ/mol {s_tag}: "
                  f"C5 闭环 {pr['closed_loop_ratio']:.3f}")

    # 45 kJ/mol 主行（对拍 v1.4 主口径 2.14）
    ref_row = next(r for r in ea_grid_rows
                   if r["ea_kj_mol"] == 45.0 and r["s_tag"] == "main_0.40")
    ea_main = next(r for r in ea_grid_rows
                   if abs(r["ea_kj_mol"] - (lfp_ea if np.isfinite(lfp_ea) else 28.0)) < 0.01
                   or abs(r["ea_kj_mol"] - 28.0) < 0.01)

    # ---- 图与产物 ----
    print("[4/4] 图与产物 ...")
    metrics = {
        "dataset": "Sandia SNL 温度轴（15/25/35 °C）+ CALCE CS2 内阻（scripts/19）",
        "apparent_ea_pairs": ea_rows,
        "apparent_ea_by_chem": ea_by_chem,
        "ea_default_j_mol": EA_DEFAULT,
        "beta": {
            "default": BETA_DEFAULT, "literature_grid": list(BETA_GRID),
            "cs2_measured_median": CS2_BETA_MEDIAN,
            "cs2_measured_cells": list(CS2_BETA_CELLS),
            "snl_resistance_scan": snl_r_scan,
            "feedback_rows": beta_rows,
        },
        "c5_by_ea": {
            "s_main": S_MAIN, "s_hi": S_HI,
            "cs2_s_rebased_check": s_rebased,
            "rows": ea_grid_rows,
            "ref_45kj_main": ref_row,
        },
        "verdict": (
            f"LFP 表观 Ea ≈ {ea_by_chem['LFP'].get('ea_median_kj', float('nan')):.0f} kJ/mol"
            f"（dt 干净对中位 {ea_by_chem['LFP'].get('ea_median_dt_clean_kj', float('nan')):.0f}）"
            f"，低于损伤模型默认 45——冷却寿命收益在 45 假设下偏乐观；"
            f"Ea 22–60 kJ/mol 区间内 C5 闭环比 "
            f"{min(r['closed_loop_ratio'] for r in ea_grid_rows if r['s_tag']=='main_0.40'):.2f}"
            f"–{max(r['closed_loop_ratio'] for r in ea_grid_rows if r['s_tag']=='main_0.40'):.2f}"
            f"（s=0.40）。β：SNL 无内阻（0/61，负结果），CS2 实测中位 "
            f"{CS2_BETA_MEDIAN:.2f} 低于文献 [1.5,3.5]——按实测 β，C5 反馈漂移 "
            f"{next(r['c5_drift_pct'] for r in beta_rows if abs(r['beta']-CS2_BETA_MEDIAN)<1e-6):+.1f}%"
            f"，§18 的 +3.9% 是保守上界。NMC 温度轴被批次异质性污染（负结果），"
            f"其 Ea 估计不给点值。"
        ),
        "elapsed_s": time.time() - t0,
    }
    (MET / "33_ea_beta.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    plotting.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    colors = {"LFP": "#c0392b", "NMC": "#2980b9", "NCA": "#27ae60"}
    for chem in ("LFP", "NMC", "NCA"):
        pts = [r for r in ea_rows if r["chem"] == chem]
        xs = [i for i, r in enumerate(pts)]
        ys = [r["ea_kj_mol"] for r in pts]
        cs = [colors[chem] if r["eligible"] else "#95a5a6" for r in pts]
        ax.scatter(xs, ys, c=cs, s=40, label=chem)
    ax.axhline(45.0, color="k", ls="--", lw=1.2, label="默认 45")
    ax.axhline(30.0, color="gray", ls=":", lw=1, label="文献下沿 30")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("温度对序号")
    ax.set_ylabel("表观 Ea（kJ/mol）")
    ax.set_title("a) SNL 温度对表观 Ea（灰=资格否决）")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    tags = [f"{r['beta']:.2f}" for r in beta_rows]
    drift = [r["c5_drift_pct"] for r in beta_rows]
    cols = ["#c0392b" if abs(r["beta"] - CS2_BETA_MEDIAN) < 1e-6 else "#95a5a6"
            for r in beta_rows]
    ax.bar(tags, drift, color=cols, width=0.55)
    ax.set_xlabel("β")
    ax.set_ylabel("C5 反馈漂移（%）")
    ax.set_title("b) β 对反馈漂移（红=CS2 实测）")
    for i, d in enumerate(drift):
        ax.text(i, d, f"{d:+.1f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1, 0]
    for s_tag, color, ls in (("main_0.40", "#c0392b", "o-"),
                             ("id_hi_0.61", "#2980b9", "s--")):
        xs = [r["ea_kj_mol"] for r in ea_grid_rows if r["s_tag"] == s_tag]
        ys = [r["closed_loop_ratio"] for r in ea_grid_rows if r["s_tag"] == s_tag]
        ax.plot(xs, ys, ls, color=color, label=f"s={s_tag.split('_')[1]}")
    ax.axhline(2.14, color="k", ls=":", lw=1, label="v1.4 主口径 2.14")
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("Ea（kJ/mol）")
    ax.set_ylabel("C5 闭环比（3C 液冷 / 1C 自然）")
    ax.set_title("c) C5 寿命比 × Ea")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    dt_ratios = [r["dt_ratio"] for r in ea_rows if r["eligible"]]
    fade_ratios = [r["fade_ratio"] for r in ea_rows if r["eligible"]]
    cols = [colors[r["chem"]] for r in ea_rows if r["eligible"]]
    ax.scatter(dt_ratios, fade_ratios, c=cols, s=40)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axvline(1.0, color="k", lw=0.8)
    ax.set_xlabel("每 EFC 壁钟比（高温/低温）")
    ax.set_ylabel("fade 比（高温/低温）")
    ax.set_title("d) 日历时长混杂检查（资格对）")
    ax.text(0.02, 0.98, "左上=温度敏感+无时长混杂", transform=ax.transAxes,
            fontsize=8, va="top")

    fig.suptitle("33a · 表观 Ea 与 β 重标：冷却寿命收益的参数敏感性", fontsize=12)
    fig.tight_layout()
    plotting.save(fig, FIG / "33a_ea_beta.png")
    plt.close(fig)

    print(f"done in {metrics['elapsed_s']:.0f}s -> 33_ea_beta.json / 33a_ea_beta.png")
    print("verdict:", metrics["verdict"][:220])


if __name__ == "__main__":
    main()
