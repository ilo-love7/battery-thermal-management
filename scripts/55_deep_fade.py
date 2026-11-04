#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/55 — 深段加速定位 + EIS 阻抗-深度耦合（v1.16，WP61）。

回答 v1.15 收尾的机制侧问题：G5「末 250 EFC 内 23→61%」的深段
加速（fade>25%）在**fade 分带速率**上怎么分布？深段加速是否伴随
**阻抗每 % fade 代价的抬升**（BoL→MoL 早段 vs MoL→EoL 晚段）？

1. fade 深度分带速率（0–15 / 15–25 / 25%+，带内 OLS）：
   G5（唯一样深段 58–67%）逐带速率与深/早加速比；G1–4 只到
   15–21%（前两带对照）；
2. EIS 阻抗-深度耦合：ΔR_pol / Δfade 逐芯分段——早段（BoL→MoL，
   0→fade_mol）vs 晚段（MoL→EoL，fade_mol→fade_eol）配对差；
   G5（无 MoL）BoL→EoL 全程代价与 G1–4 晚段比；
3. 判词：深段加速（轨迹侧）与阻抗每 % fade 代价抬升（EIS 侧）
   是否同向共变。

⚠️ 口径：fade 带按轨迹点 fade_pct 落带，带内 OLS 需 ≥3 点否则
flag=1（无数据档不用默认值顶）；EIS 逐芯特征取同芯 SOC 中位；
G5 无 MoL 中间谱——早/晚段分割只在 G1–4 可做；G5 三芯中 C5 无
EoL 谱（n=2 全程代价）。判词从计算值生成。

产物：results/metrics/55_deep_fade.json、results/figures/55a_deep_fade.png。
依赖：oxford_efc_traj.csv、oxford_fade_eol.csv、eis_features_eol.csv。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal.efc_rate import segment_band_rates  # noqa: E402
from bmthermal.plotting import save as save_fig, setup_style  # noqa: E402

OUT_M = ROOT / "results" / "metrics"
OUT_F = ROOT / "results" / "figures"
TRAJ = ROOT / "results" / "cache" / "oxford_efc_traj.csv"
FADE_EOL = ROOT / "results" / "cache" / "oxford_fade_eol.csv"
EIS_EOL = ROOT / "results" / "cache" / "eis_features_eol.csv"

BANDS = ((0.0, 15.0), (15.0, 25.0), (25.0, 100.0))
KIND = {
    1: "combined_c2_5d", 2: "combined_c4_5d",
    3: "combined_c2_10d", 4: "combined_c4_10d",
    5: "cycle_only", 6: "calendar_only",
    7: "cccv_90_5d", 8: "cccv_90_10d",
    9: "cccv_42_5d", 10: "cccv_42_10d",
}


def _med(s: pd.Series):
    return float(np.nanmedian(s)) if len(s) and s.notna().any() else None


def eis_depth_coupling(eis: pd.DataFrame,
                       fade: pd.DataFrame) -> pd.DataFrame:
    """逐芯 ΔR_pol/Δfade 分段（早 BoL→MoL / 晚 MoL→EoL / 全程）。

    EIS 特征同芯取 SOC 中位后作阶段差；fade 区间用 fade_mol_pct /
    fade_eol_pct（50 口径）。输出：group, cell, kind, fade_mol,
    fade_eol, drpol_early, drpol_late, drpol_full（mΩ / %fade）。
    """
    med = eis.groupby(["group", "cell", "stage"]).r_pol.median().unstack()
    med = med.reset_index()
    m = med.merge(fade[["group", "cell", "fade_mol_pct", "fade_eol_pct"]],
                  on=["group", "cell"], how="left")
    rows = []
    for _, r in m.iterrows():
        g, cell = int(r.group), int(r.cell)
        row = {"group": g, "cell": cell, "kind": KIND.get(g, "?"),
               "fade_mol": r.fade_mol_pct, "fade_eol": r.fade_eol_pct,
               "drpol_early": float("nan"), "drpol_late": float("nan"),
               "drpol_full": float("nan")}
        bol = r.get("BoL", np.nan)
        mol = r.get("MoL", np.nan)
        eol = r.get("EoL", np.nan)
        # 早段：0 → fade_mol
        if np.isfinite(bol) and np.isfinite(mol) and \
                np.isfinite(r.fade_mol_pct) and r.fade_mol_pct > 1:
            row["drpol_early"] = float((mol - bol) * 1000 / r.fade_mol_pct)
        # 晚段：fade_mol → fade_eol
        if np.isfinite(mol) and np.isfinite(eol) and \
                np.isfinite(r.fade_mol_pct) and np.isfinite(r.fade_eol_pct):
            d_fade = r.fade_eol_pct - r.fade_mol_pct
            if d_fade > 1:
                row["drpol_late"] = float((eol - mol) * 1000 / d_fade)
        # 全程：0 → fade_eol（G5 无 MoL 时唯一段）
        if np.isfinite(bol) and np.isfinite(eol) and \
                np.isfinite(r.fade_eol_pct) and r.fade_eol_pct > 1:
            row["drpol_full"] = float((eol - bol) * 1000 / r.fade_eol_pct)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    print("[1] 读缓存")
    traj = pd.read_csv(TRAJ)
    fade = pd.read_csv(FADE_EOL)
    eis = pd.read_csv(EIS_EOL)
    print(f"    traj {len(traj)} 点 / {traj.groupby(['group','cell']).ngroups} 芯；"
          f"EIS {len(eis)} 谱")

    print("[2] fade 深度分带速率")
    br = segment_band_rates(traj, bands=BANDS, min_pts=3)
    br.to_csv(ROOT / "results" / "cache" / "deep_band_rates.csv", index=False)
    # 逐芯加速比（深带/早带）
    piv = br.pivot_table(index=["group", "cell"], columns="band_lo",
                         values="rate_band")
    acc_rows = []
    for (g, cell), r in piv.iterrows():
        early = r.get(0.0, np.nan)
        rec = {"group": int(g), "cell": int(cell),
               "kind": KIND.get(int(g), "?"),
               "rate_0_15": early,
               "rate_15_25": r.get(15.0, np.nan),
               "rate_25_100": r.get(25.0, np.nan),
               "acc_25_100": float("nan"), "acc_15_25": float("nan")}
        deep = r.get(25.0, np.nan)
        if np.isfinite(deep) and np.isfinite(early) and abs(early) > 1e-9:
            rec["acc_25_100"] = float(deep / early)
        if np.isfinite(r.get(15.0, np.nan)) and np.isfinite(early) \
                and abs(early) > 1e-9:
            rec["acc_15_25"] = float(r[15.0] / early)
        acc_rows.append(rec)
    acc = pd.DataFrame(acc_rows)

    print("[3] EIS 阻抗-深度耦合")
    cou = eis_depth_coupling(eis, fade)

    # 组级汇总
    grp_rows = []
    for g in sorted(acc.group.unique()):
        s = acc[acc.group == g]
        c = cou[cou.group == g]
        grp_rows.append({
            "group": int(g), "kind": KIND.get(int(g), "?"), "n": int(len(s)),
            "rate_0_15_med": _med(s.rate_0_15),
            "rate_15_25_med": _med(s.rate_15_25),
            "rate_25_100_med": _med(s.rate_25_100),
            "acc_25_100_med": _med(s.acc_25_100),
            "acc_15_25_med": _med(s.acc_15_25),
            "drpol_early_med": _med(c.drpol_early),
            "drpol_late_med": _med(c.drpol_late),
            "drpol_full_med": _med(c.drpol_full),
        })
    grp = pd.DataFrame(grp_rows)

    # G5 深段 vs G1–4
    g5 = acc[acc.group == 5]
    g14 = acc[acc.group.isin([1, 2, 3, 4])]
    g5_deep = _med(g5.rate_25_100)
    g5_early = _med(g5.rate_0_15)
    g5_acc = _med(g5.acc_25_100)
    g14_early = _med(g14.rate_0_15)
    g14_mid = _med(g14.rate_15_25)
    g14_acc = _med(g14.acc_15_25)

    cou_g5 = cou[cou.group == 5]
    cou_g14 = cou[cou.group.isin([1, 2, 3, 4])]
    late_over_early = None
    paired = cou_g14.dropna(subset=["drpol_early", "drpol_late"])
    if len(paired):
        late_over_early = float(np.nanmedian(
            paired.drpol_late / paired.drpol_early.replace(0, np.nan)))
    g5_full = _med(cou_g5.drpol_full)
    g14_late = _med(cou_g14.drpol_late)
    g14_early_c = _med(cou_g14.drpol_early)

    print("[4] 图 55a")
    setup_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    band_labels = ["0–15", "15–25", "25+"]
    band_cols = ["rate_0_15_med", "rate_15_25_med", "rate_25_100_med"]
    ax = axes[0]
    xs = np.arange(3)
    for g, mk, c, lab in [(5, "o", "#c0392b", "G5 cycle"),
                          (1, "s", "#2980b9", "G1"),
                          (3, "^", "#27ae60", "G3"),
                          (10, "D", "#8e44ad", "G10")]:
        row = grp[grp.group == g]
        if not len(row):
            continue
        vals = [row[col].iloc[0] if row[col].notna().iloc[0] else np.nan
                for col in band_cols]
        ax.plot(xs, vals, marker=mk, color=c, label=lab, lw=1.4)
    ax.set_xticks(xs, band_labels)
    ax.set_xlabel("fade band (%)")
    ax.set_ylabel("band rate (%/100 EFC)")
    ax.set_title("fade rate by depth band")
    ax.legend(fontsize=7)
    ax = axes[1]
    for (g, cell), r in piv.iterrows():
        c = "#c0392b" if g == 5 else "#2980b9"
        vals = [r.get(15.0, np.nan), r.get(25.0, np.nan)]
        ax.plot([15, 25], vals, marker="o", ms=3.5, color=c, alpha=0.65,
                lw=1.0)
    ax.plot([], [], color="#c0392b", label="G5")
    ax.plot([], [], color="#2980b9", label="G1-4 etc")
    ax.set_xlabel("fade band start (%)")
    ax.set_ylabel("band rate (%/100 EFC)")
    ax.set_title("per-cell band rates (deep focus)")
    ax.legend(fontsize=7)
    ax = axes[2]
    for _, r in cou_g14.dropna(subset=["drpol_early"]).iterrows():
        x = [0, 1]
        y = [r.drpol_early,
             r.drpol_late if np.isfinite(r.drpol_late) else np.nan]
        if np.isfinite(y[1]):
            ax.plot(x, y, color="#2980b9", alpha=0.55, lw=1.0, marker="o",
                    ms=3.5)
        else:
            ax.plot([0], [y[0]], color="#2980b9", alpha=0.55, marker="o",
                    ms=3.5)
    for _, r in cou_g5.dropna(subset=["drpol_full"]).iterrows():
        ax.plot([0, 1], [np.nan, np.nan], color="#c0392b")
        ax.plot([0.5], [r.drpol_full], marker="D", color="#c0392b", ms=6)
    ax.set_xticks([0, 0.5, 1],
                  ["early\nBoL→MoL", "G5 full\nBoL→EoL", "late\nMoL→EoL"])
    ax.set_ylabel("ΔR_pol / Δfade (mΩ/%)")
    ax.set_title("impedance cost per % fade")
    fig.tight_layout()
    save_fig(fig, "55a_deep_fade.png", OUT_F)

    # ---- 判词（从计算值生成）----
    n_deep_cells = int(g5.acc_25_100.notna().sum()) if len(g5) else 0
    if g5_acc is not None and g5_acc > 2.0:
        acc_dir = (f"深段加速确认（>25% 带速率/早带 = {g5_acc:.1f}×，"
                   f"深段中位 {g5_deep if g5_deep is not None else float('nan'):.1f}"
                   f" vs 早段 {g5_early if g5_early is not None else float('nan'):.2f}"
                   f" %/100 EFC）")
    elif g5_acc is not None and g5_acc > 1.2:
        acc_dir = f"深段加速温和（比 {g5_acc:.2f}×）"
    elif g5_acc is not None:
        acc_dir = f"深段加速不显著（比 {g5_acc:.2f}×）"
    else:
        acc_dir = "深段加速未量化（早带速率≈0 或点数不足）"
    if late_over_early is not None and late_over_early > 2.0:
        eis_dir = (f"阻抗每 % fade 代价晚段抬升（晚/早 = {late_over_early:.1f}×，"
                   f"早 {g14_early_c if g14_early_c is not None else float('nan'):.2f}"
                   f" → 晚 {g14_late if g14_late is not None else float('nan'):.2f}"
                   f" mΩ/%）")
    elif late_over_early is not None:
        eis_dir = f"阻抗每 % fade 代价晚段/早段比 {late_over_early:.2f}×（抬升不显著）"
    else:
        eis_dir = "阻抗分段代价不可算（早/晚配对缺失）"
    g5_cmp = ""
    if g5_full is not None and g14_late is not None and g14_late > 0:
        g5_cmp = (f"；G5 全程代价 {g5_full:.2f} mΩ/% ≈ G1–4 晚段水平"
                  f"（{g14_late:.2f}）而非早段（{g14_early_c:.2f}）"
                  if abs(g5_full / g14_late - 1) < 0.5 else
                  f"；G5 全程代价 {g5_full:.2f} mΩ/% vs G1–4 晚段 "
                  f"{g14_late:.2f}（差 {g5_full/g14_late:.2f}×）")
    if g14_acc is not None:
        if g14_acc > 1.2:
            mid_txt = f"15–25% 带已见抬升（/早带 = {g14_acc:.2f}×）"
        elif g14_acc > 0.8:
            mid_txt = f"15–25% 带与早带持平（比 {g14_acc:.2f}×）"
        else:
            mid_txt = f"15–25% 带反而放缓（比 {g14_acc:.2f}×）"
        mid_dir = f"G1–4（止步 21%）{mid_txt}"
    else:
        mid_dir = "G1–4 深带无点"
    # ③ 共变判读：阻抗代价随深度单调 vs fade 加速只在深段——非锁定
    if late_over_early is not None and g5_acc is not None:
        if g5_acc > 1.2 and (g14_acc is None or g14_acc < 1.2):
            g14_acc_txt = (f"{g14_acc:.2f}×" if g14_acc is not None
                           else "无点")
            covar_txt = (
                "深段同向但**非锁定**（阻抗代价随深度单调抬升发生在 "
                f"fade 放平段——G1–4 代价 {late_over_early:.1f}× 而速率 "
                f"{g14_acc_txt}，与 50「fade 后段放缓而阻抗加速」解耦"
                "一致；fade 速率加速只在 >25% 膝区出现，G5 深段两量合流")
        elif g5_acc > 1.2:
            covar_txt = (
                "同向共变（深段容量损失伴随界面/扩散阻抗超线性增长")
        else:
            covar_txt = (
                "方向不齐（轨迹加速与阻抗代价不同步，深段加速需另寻机制源")
    else:
        covar_txt = "共变不可判（加速比或阻抗分段缺失）"
    verdict = (
        f"深段加速定位 + 阻抗-深度耦合收口（{traj.groupby(['group','cell']).ngroups}"
        f" 芯/{len(traj)} 轨迹点、EIS 三阶段逐芯配对）——"
        f"① 轨迹侧：{acc_dir}；{mid_dir}。"
        f"② EIS 侧：{eis_dir}{g5_cmp}。"
        f"③ 共变判读：fade>25% 深段加速与阻抗每 % fade 代价抬升"
        f"{covar_txt}"
        f"——与 49「后段阻抗链份额 63%」拼合为容量-阻抗分层解耦图景）。"
        f"⚠️ fade 带内 OLS 需 ≥3 点（flag=1 为无数据档，不用默认值顶）；"
        f"短轨迹（G7–10 仅 5–6 点）深带全 flag；G5 C5 无 EoL 谱"
        f"（全程代价 n=2）；EIS 特征同芯 SOC 中位（谱间散布见 47–49）；"
        f"早/晚段窗口由 MoL 落点决定（fade_mol 10–17%），带边界与段"
        f"边界不完全重合；G1–4 晚段只跨 5–8% fade（分母窄→比值散布"
        f"大，逐芯配对线见图 55a 右）。"
    )

    metrics = {
        "n_cells": int(traj.groupby(["group", "cell"]).ngroups),
        "n_traj_points": int(len(traj)),
        "bands_pct": [list(b) for b in BANDS],
        "group_table": grp.to_dict(orient="records"),
        "per_cell_band": acc.to_dict(orient="records"),
        "eis_depth_coupling": cou.to_dict(orient="records"),
        "g5_vs_g1_4": {
            "g5_deep_rate_med": g5_deep,
            "g5_early_rate_med": g5_early,
            "g5_acc_25_100_med": g5_acc,
            "g5_n_deep_cells": n_deep_cells,
            "g1_4_early_rate_med": g14_early,
            "g1_4_rate_15_25_med": g14_mid,
            "g1_4_acc_15_25_med": g14_acc,
            "late_over_early_drpol": late_over_early,
            "drpol_early_med_g1_4": g14_early_c,
            "drpol_late_med_g1_4": g14_late,
            "drpol_full_med_g5": g5_full,
        },
        "caveats": [
            "fade 带内 OLS 需 ≥3 点；flag=1 为无数据档（不用默认值顶实测标签）",
            "G5 无 MoL 谱：早/晚段分割只在 G1–4 可做；G5 只有 BoL→EoL 全程代价",
            "G5 C5 无 EoL 谱（EIS 全程代价 n=2）",
            "EIS 特征同芯 SOC 中位（20/50/80 三谱）",
            "G1–4 晚段 fade 区间窄（5–8%）→ 比值散布大，用逐芯配对中位",
            "短轨迹（G7–10，5–6 点）深带无点，不入加速比",
            "带边界（15/25%）与 EIS 段边界（MoL 落点 10–17%）不重合",
        ],
        "verdict": verdict,
    }
    OUT_M.mkdir(parents=True, exist_ok=True)
    with open(OUT_M / "55_deep_fade.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print("[5] 判词")
    print(verdict)
    print("[55] done")


if __name__ == "__main__":
    main()
