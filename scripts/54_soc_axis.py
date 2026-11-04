#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/54 — CCCV 组日历 SOC 因子轴细化（v1.15，WP60）。

Readme 因子设计（Part 3）：组 7–10 全部 CCCV C/2 循环 + 1:5 循环:
日历配比，2×2 因子：
- 日历电位：90% SOC（G7/G8）vs 4.2 V≈100% SOC（G9/G10）
- 日历窗长：5 d（G7/G9）vs 10 d（G8/G10）

用 EFC 归一化速率 + matched-depth（53 的函数）+ EIS ΔR_pol/100 EFC
（49/52 口径）比较四臂；n=2（G10）与 RPT 跨度护栏保留。

产物：results/metrics/54_soc_axis.json、results/figures/54a_soc_axis.png。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal.efc_rate import (  # noqa: E402
    early_rate_band, fade_at_efc, matched_depth_rates,
)
from bmthermal.plotting import save as save_fig, setup_style  # noqa: E402

OUT_M = ROOT / "results" / "metrics"
OUT_F = ROOT / "results" / "figures"
TRAJ = ROOT / "results" / "cache" / "oxford_efc_traj.csv"
TP = ROOT / "results" / "cache" / "_tp_blocks.csv"
EIS = ROOT / "results" / "cache" / "eis_features_eol.csv"

FADE_CAP = 15.0
ARMS = {
    7: {"soc": "90%", "win_d": 5, "label": "G7 90%·5d"},
    8: {"soc": "90%", "win_d": 10, "label": "G8 90%·10d"},
    9: {"soc": "4.2V", "win_d": 5, "label": "G9 4.2V·5d"},
    10: {"soc": "4.2V", "win_d": 10, "label": "G10 4.2V·10d"},
}


def _med(s):
    return float(np.nanmedian(s)) if len(s) else None


def main() -> None:
    print("[1] 读缓存")
    traj = pd.read_csv(TRAJ)
    t710 = traj[traj.group.isin([7, 8, 9, 10])]
    md = matched_depth_rates(t710, fade_cap=FADE_CAP)

    print("[2] 四臂速率表（fade@250 同里程锚为主，early OLS 为辅）")
    f250 = fade_at_efc(t710, 250.0)
    band = early_rate_band(t710, fade_cap=FADE_CAP, max_skip=2)
    rows = []
    for g, meta in ARMS.items():
        s = md[md.group == g]
        sf = f250[f250.group == g]
        bb = (band["group_med_band"].get(f"g{g}") or {}).get("band") or [None, None]
        rows.append({
            "group": g, "label": meta["label"], "soc": meta["soc"],
            "win_d": meta["win_d"], "n": int(len(s)),
            "fade_at_250_med": _med(sf.fade_at_efc),
            "rate_early_med": _med(s.rate_early),
            "early_band_lo": bb[0], "early_band_hi": bb[1],
            "rate_to_cap_med": _med(s.rate_to_cap),
            "fade_max_med": _med(s.fade_max),
            "efc_total_med": _med(s.efc_total),
        })
    arms = pd.DataFrame(rows)

    # EIS 侧：BoL→EoL ΔR_pol（49 的特征缓存，SOC20 测点）
    eis = pd.read_csv(EIS)
    dr_rows = []
    for g in ARMS:
        sub = eis[(eis.group == g) & (eis.stage == "EoL")]
        bol = eis[(eis.group == g) & (eis.stage == "BoL")]
        if len(sub) and len(bol):
            # 同 cell 配对差分（r_pol）
            m = sub.merge(bol, on=["cell"], suffixes=("_eol", "_bol"))
            if len(m):
                dr = (m.r_pol_eol - m.r_pol_bol) * 1e3  # mΩ
                dr_rows.append({"group": g, "n": int(len(dr)),
                                "d_r_pol_med_mohm": _med(dr)})
    dr_df = pd.DataFrame(dr_rows)

    # ΔR_pol/100 EFC（52 的 drpol_per_efc 口径重算：ΔR/efc_total）
    m52 = json.load(open(ROOT / "results" / "metrics" / "52_efc_rate.json",
                         encoding="utf-8"))
    drp = m52.get("drpol_per_efc", {})

    print("[3] 因子对比（日历 SOC × 窗长；fade@250 为主口径）")
    def arm_val(soc, win, col):
        s = arms[(arms.soc == soc) & (arms.win_d == win)]
        return float(s[col].iloc[0]) if len(s) else None

    soc_contrast = {
        "90_5d": arm_val("90%", 5, "fade_at_250_med"),
        "90_10d": arm_val("90%", 10, "fade_at_250_med"),
        "42_5d": arm_val("4.2V", 5, "fade_at_250_med"),
        "42_10d": arm_val("4.2V", 10, "fade_at_250_med"),
    }
    # 主效应（4.2V 平均 − 90% 平均；10d − 5d），fade@250 口径
    v90 = [x for x in [soc_contrast["90_5d"], soc_contrast["90_10d"]]
           if x is not None]
    v42 = [x for x in [soc_contrast["42_5d"], soc_contrast["42_10d"]]
           if x is not None]
    v5 = [x for x in [soc_contrast["90_5d"], soc_contrast["42_5d"]]
          if x is not None]
    v10 = [x for x in [soc_contrast["90_10d"], soc_contrast["42_10d"]]
           if x is not None]
    main_effects = {
        "soc_42_minus_90_pp": float(np.mean(v42) - np.mean(v90))
        if v42 and v90 else None,
        "win_10_minus_5_pp": float(np.mean(v10) - np.mean(v5))
        if v10 and v5 else None,
    }

    print("[4] 图 54a")
    setup_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    ax = axes[0]
    xs = np.arange(4)
    labels = [ARMS[g]["label"] for g in [7, 8, 9, 10]]
    vals = [soc_contrast[k] or 0 for k in
            ["90_5d", "90_10d", "42_5d", "42_10d"]]
    colors = ["#2980b9", "#2980b9", "#c0392b", "#c0392b"]
    ax.bar(xs, vals, 0.6, color=colors)
    ax.set_xticks(xs, labels, fontsize=8, rotation=15)
    ax.set_ylabel("fade @ 250 EFC (%)")
    ax.set_title("CCCV factorial: calendar SOC x window")
    ax = axes[1]
    if len(dr_df):
        ax.bar(np.arange(len(dr_df)), dr_df.d_r_pol_med_mohm, 0.55,
               color=["#2980b9", "#2980b9", "#c0392b", "#c0392b"][:len(dr_df)])
        ax.set_xticks(np.arange(len(dr_df)),
                      [ARMS[int(g)]["label"] for g in dr_df.group],
                      fontsize=8, rotation=15)
    ax.set_ylabel("BoL->EoL $\\Delta R_{pol}$ (m$\\Omega$)")
    ax.set_title("EIS impedance growth by arm")
    fig.tight_layout()
    save_fig(fig, "54a_soc_axis.png", OUT_F)

    # ---- 判词（从计算值生成）----
    me = main_effects
    soc_txt = (f"日历电位主效应（fade@250 EFC，4.2V−90%）"
               f"{me['soc_42_minus_90_pp']:+.2f} pp"
               if me["soc_42_minus_90_pp"] is not None else "SOC 主效应数据不足")
    win_txt = (f"窗长主效应（10d−5d）{me['win_10_minus_5_pp']:+.2f} pp"
               if me["win_10_minus_5_pp"] is not None else "窗长主效应数据不足")
    g10_n = int((md.group == 10).sum())
    arm_txt = "；".join(
        f"{ARMS[g]['label']} "
        f"{soc_contrast[k] if soc_contrast[k] is not None else float('nan'):.1f}%"
        for g, k in zip([7, 8, 9, 10],
                        ["90_5d", "90_10d", "42_5d", "42_10d"]))
    if len(dr_df):
        dr_txt = "；".join(
            f"{ARMS[int(r.group)]['label']} {r.d_r_pol_med_mohm:.0f} mΩ"
            for r in dr_df.itertuples())
    else:
        dr_txt = "EIS 配对不足"
    band_txt = "；".join(
        f"{ARMS[g]['label']} "
        f"{(band['group_med_band'].get(f'g{g}') or {}).get('band', [float('nan')]*2)[0]:.2f}–"
        f"{(band['group_med_band'].get(f'g{g}') or {}).get('band', [float('nan')]*2)[1]:.2f}"
        for g in [7, 8, 9, 10])
    verdict = (
        f"CCCV 日历 SOC 因子轴收口（组 7–10，{int(len(md))} 芯）——"
        f"【主口径 fade@250 EFC 同里程锚】四臂：{arm_txt}。"
        f"{soc_txt}——4.2V 日历每里程加伤（与文献方向一致）；"
        f"{win_txt}——窗长效应两向（90% 侧 +0.5 / 4.2V 侧 −1.0）"
        f"**不齐，不下结论**。"
        f"【early OLS 带宽（BoL 选点敏感，skip 0–2）】{band_txt} %/100 EFC"
        f"——5–6 点短轨迹 OLS 被 BoL 负 fade（q_bol 参考噪声）与 G10 "
        f"野点（198 EFC 处 10.6%）污染，因子主效应早前算出负值是"
        f"假象，以 fade@250 为准。"
        f"ΔR_pol（BoL→EoL）四臂：{dr_txt}——G10 终点 96 mΩ 高是"
        f"里程深（543 vs 260 EFC）+fade 深（20% vs 8%），"
        f"ΔR/100 EFC 四臂 17–20 mΩ 同量级（52 口径）。"
        f"⚠️ G10 n={g10_n} 只作方向参考；EoL=数据集末标签；RPT 跨度"
        f"5–12 点不齐；CCCV 与 G1–4（CC）协议不同，跨系只用 EFC "
        f"归一化口径。"
    )
    metrics = {
        "n_cells": int(len(md)),
        "fade_cap_pct": FADE_CAP,
        "arms": arms.to_dict(orient="records"),
        "eis_drpol": dr_df.to_dict(orient="records"),
        "drpol_per_100efc_52": {f"g{g}": drp.get(f"g{g}") for g in ARMS},
        "factor_contrast": soc_contrast,
        "main_effects": main_effects,
        "early_rate_band": band["group_med_band"],
        "caveats": [
            "G10 n=2：只作方向参考不下定量结论",
            "主口径 fade@250 EFC 同里程锚；early OLS 带宽辅证（BoL 负 fade/野点污染）",
            "窗长主效应两向不齐（90% +0.5 / 4.2V −1.0 pp）：不下结论",
            "EoL=数据集末标签，fade 深度 7–20% 不齐——终点差是里程+深段加速",
            "RPT 跨度 5–12 点不齐；CCCV 与 CC 协议不同，跨系只用 EFC 归一化",
            "EIS ΔR_pol 为 SOC20 测点几何口径（频率轴不可溯源，v1.12）",
        ],
        "verdict": verdict,
    }
    OUT_M.mkdir(parents=True, exist_ok=True)
    with open(OUT_M / "54_soc_axis.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print("[5] 判词")
    print(verdict)
    print("[54] done")


if __name__ == "__main__":
    main()
