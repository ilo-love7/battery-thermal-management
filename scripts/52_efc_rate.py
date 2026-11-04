#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""52 — EFC/时间轴归一化：fade 与 ΔR_pol 的同里程速率（v1.14）

问题（50 遗留）：组 5 fade 61.4% vs 组 6 4.0% 是同窗口「谁死谁活」，
不是同里程 fade 速率。本脚本把 27 芯全量轨迹分解为：
  ① 里程差：EFC 总量（放电 Ah / q_bol）与日历天数（块时长求和）；
  ② 速率差：fade/100 EFC（割线 + OLS）与 fade/100 天——回答
     「61% vs 4% 是里程多还是速率快」；
  ③ ΔR_pol/100 EFC（§72 的 per-mileage 版）——跨工况可比口径。
双估计器对拍：∫|I|dt vs 分步 Amphr 极差和（坑：Amphr 步内累积）。

产物：results/cache/oxford_tp_blocks.csv、results/metrics/52_efc_rate.json、
      results/figures/52a_efc_rate.png
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal.eis_features import (  # noqa: E402
    block_throughput_stats, parse_block_member,
)
from bmthermal.efc_rate import build_trajectories, group_rates  # noqa: E402
from bmthermal.guide_map import is_rpt, load_guide_rows  # noqa: E402
from bmthermal.plotting import save as save_fig, setup_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OXFORD = Path("D:/项目/02_数据集/01_国际公开集/Oxford_PathDependent")
OUT_M = ROOT / "results" / "metrics"
OUT_F = ROOT / "results" / "figures"
CACHE = ROOT / "results" / "cache" / "_tp_blocks.csv"

GUIDE_FILES = [
    (OXFORD / "Guide_to_Datafiles.xlsx", 1, 4),
    (OXFORD / "Part2" / "Guide_to_Datafiles_2.xlsx", 1, 6),
    (OXFORD / "Part3" / "Guide_to_Datafiles_3.xlsx", 7, 10),
]

ZIP_FILES = {
    1: [OXFORD / "Group_1.zip", OXFORD / "Part2" / "Group_1.zip"],
    2: [OXFORD / "Group_2.zip", OXFORD / "Part2" / "Group_2.zip"],
    3: [OXFORD / "Group_3.zip", OXFORD / "Part2" / "Group_3.zip"],
    4: [OXFORD / "Group_4.zip", OXFORD / "Part2" / "Group_4.zip"],
    5: [OXFORD / "Part2" / "Group_5.zip"],
    6: [OXFORD / "Part2" / "Group_6.zip"],
    7: [OXFORD / "Part3" / "Group_7.zip"],
    8: [OXFORD / "Part3" / "Group_8.zip"],
    9: [OXFORD / "Part3" / "Group_9.zip"],
    10: [OXFORD / "Part3" / "Group_10.zip"],
}

GROUP_META = {
    1: {"kind": "combined", "c_rate": 0.5}, 2: {"kind": "combined", "c_rate": 0.25},
    3: {"kind": "combined", "c_rate": 0.5}, 4: {"kind": "combined", "c_rate": 0.25},
    5: {"kind": "cycle_only", "c_rate": 0.5}, 6: {"kind": "calendar_only", "c_rate": None},
    7: {"kind": "combined_cccv", "c_rate": 0.5}, 8: {"kind": "combined_cccv", "c_rate": 0.5},
    9: {"kind": "combined_cccv", "c_rate": 0.5}, 10: {"kind": "combined_cccv", "c_rate": 0.5},
}
CYCLED = {1, 2, 3, 4, 5, 7, 8, 9, 10}

Q_MIN = 0.3          # 容量点门槛（Ah，滤碎段）
TP_REL_TOL = 0.05    # 双估计器 5% 相对差容忍


# ---------------------------------------------------------------------------
# 1. 全块吞吐收集（含 Profile；缓存 _tp_blocks.csv）
# ---------------------------------------------------------------------------

def collect_tp(guide: pd.DataFrame) -> pd.DataFrame:
    if CACHE.exists():
        out = pd.read_csv(CACHE)
        print(f"[2] 块缓存命中 {len(out)} 行")
        return out

    frows = []
    for g in sorted(guide.group.unique()):
        sub_g = guide[guide.group == g]
        zips = [p for p in ZIP_FILES.get(g, []) if p.exists()]
        if not zips:
            print(f"    ! 组 {g} 无 zip，跳过")
            continue
        member_idx: dict[tuple[int, int], tuple[str, zipfile.ZipFile]] = {}
        handles = []
        for p in zips:
            z = zipfile.ZipFile(p)
            handles.append(z)
            for name in z.namelist():
                meta = parse_block_member(name)
                if meta and meta["tpg"] == g:
                    member_idx[(meta["cell"], meta["step"])] = (name, z)
        try:
            for _, gr in sub_g.sort_values("step").iterrows():
                hit = member_idx.get((int(gr.cell), int(gr.step)))
                if hit is None:
                    print(f"    ! 缺块 G{g} C{gr.cell} step {gr.step}")
                    continue
                name, z = hit
                try:
                    st = block_throughput_stats(z.read(name))
                except Exception as exc:  # noqa: BLE001
                    print(f"    ! 解析失败 G{g} C{gr.cell} step "
                          f"{gr.step}: {exc}")
                    continue
                frows.append({
                    "group": g, "cell": int(gr.cell), "step": int(gr.step),
                    "label": gr.label, "guide": gr.guide, "member": name,
                    **st,
                })
        finally:
            for z in handles:
                z.close()
        n = sum(1 for r in frows if r["group"] == g)
        print(f"    组 {g}: {n} 块解析")
    out = pd.DataFrame(frows)
    if len(out):
        out.to_csv(CACHE, index=False)
    return out



# ---------------------------------------------------------------------------
# 3. ΔR_pol/100 EFC（49 的 EIS 特征 × 52 的 EFC 轨迹）
# ---------------------------------------------------------------------------

def drpol_per_efc(tp: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    cache = ROOT / "results" / "cache" / "eis_features_eol.csv"
    if not cache.exists():
        return pd.DataFrame()
    eis = pd.read_csv(cache)
    if not {"group", "cell", "soc", "stage", "r_pol"}.issubset(eis.columns):
        return pd.DataFrame()
    rows = []
    for (g, cell, soc), sub in eis.groupby(["group", "cell", "soc"]):
        st = dict(zip(sub.stage, sub.r_pol))
        if "BoL" in st and "EoL" in st:
            d_rp = float(st["EoL"] - st["BoL"]) * 1e3      # mΩ
        elif "BoL" in st and "MoL" in st:
            d_rp = float(st["MoL"] - st["BoL"]) * 1e3
            st["partial"] = True
        else:
            continue
        rr = rates[(rates.group == g) & (rates.cell == cell)]
        if rr.empty:
            continue
        efc_tot = float(rr.efc_total.iloc[0])
        if efc_tot <= 0:
            continue
        rows.append({
            "group": int(g), "cell": int(cell), "soc": int(soc),
            "d_r_pol_mohm": d_rp,
            "efc_total": efc_tot,
            "drpol_per_100efc": d_rp / efc_tot * 100,
            "fade_eol_pct": float(rr.fade_eol_pct.iloc[0]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. 组统计 + 判词
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_M.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)
    setup_style()

    # ---- 1. Guide + 全块吞吐 ---------------------------------------
    print("[1] guide 映射")
    guide = load_guide_rows(GUIDE_FILES)
    print(f"    {len(guide)} 行（RPT {int(guide.label.map(is_rpt).sum())}）")

    print("[2] 全块吞吐解析（含 Profile）")
    tp = collect_tp(guide)
    print(f"    共 {len(tp)} 块")

    # 双估计器对拍
    valid = tp[(tp.ah_tp > 1e-6) & (tp.n_rows > 3)]
    rel = (valid.ah_tp - valid.ah_amphr).abs() / valid.ah_tp.replace(0, np.nan)
    dual = {
        "n": int(len(valid)),
        "rel_diff_med": float(np.nanmedian(rel)),
        "rel_diff_p95": float(np.nanquantile(rel.dropna(), 0.95))
        if rel.notna().any() else None,
        "frac_over_5pct": float((rel > TP_REL_TOL).mean()) if len(valid) else None,
    }
    print(f"[2b] 双估计器: med {dual['rel_diff_med']:.3f} "
          f"p95 {dual['rel_diff_p95']:.3f} >5% 占 {dual['frac_over_5pct']:.0%}")

    # ---- 2. 轨迹与速率 ---------------------------------------------
    print("[3] 逐芯 EFC/日历轨迹")
    traj, rates = build_trajectories(tp)
    traj.to_csv(ROOT / "results" / "cache" / "oxford_efc_traj.csv", index=False)
    print(f"    容量点 {len(traj)} → 速率表 {len(rates)} 芯")

    greq = group_rates(rates, GROUP_META)
    dr = drpol_per_efc(tp, rates)
    dr_grp = {}
    if len(dr):
        for g, sub in dr.groupby("group"):
            dr_grp[f"g{int(g)}"] = {
                "n": int(len(sub)),
                "drpol_per_100efc_med": float(np.nanmedian(sub.drpol_per_100efc)),
                "d_r_pol_med_mohm": float(np.nanmedian(sub.d_r_pol_mohm)),
            }

    # ---- 3. 图 ------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    colors = {1: "#2980b9", 2: "#3498db", 3: "#1abc9c", 4: "#16a085",
              5: "#c0392b", 6: "#8e44ad", 7: "#d35400", 8: "#e67e22",
              9: "#7f8c8d", 10: "#95a5a6"}
    for g in sorted(traj.group.unique()):
        sub = traj[traj.group == g]
        for cell, sc in sub.groupby("cell"):
            axes[0, 0].plot(sc.efc, sc.fade_pct, color=colors.get(g, "k"),
                            alpha=0.6, lw=1.2)
            axes[0, 1].plot(sc.days, sc.fade_pct, color=colors.get(g, "k"),
                            alpha=0.6, lw=1.2)
    axes[0, 0].set_xlabel("EFC (discharge Ah / q_bol)")
    axes[0, 0].set_ylabel("fade [%]")
    axes[0, 0].set_title("fade vs EFC (per cell)")
    axes[0, 1].set_xlabel("calendar days (block-time sum)")
    axes[0, 1].set_ylabel("fade [%]")
    axes[0, 1].set_title("fade vs days (per cell)")

    # 速率条形：cycled 组 fade/100 EFC（割线中位）
    cyc_g = [g for g in sorted(greq) if greq[g]["kind"] != "calendar_only"
             and greq[g].get("rate_efc_secant_med") is not None]
    xs = np.arange(len(cyc_g))
    vals = [greq[k]["rate_efc_secant_med"] or 0 for k in cyc_g]
    labels = [f"{k}\n({greq[k]['n']})" for k in cyc_g]
    axes[1, 0].bar(xs, vals, 0.6, color=[colors.get(int(k[1:]), "k") for k in cyc_g])
    axes[1, 0].set_xticks(xs, labels, fontsize=8)
    axes[1, 0].set_ylabel("fade / 100 EFC  [%]")
    axes[1, 0].set_title("same-mileage fade rate (secant)")

    if len(dr):
        dr_keys = sorted(dr_grp)
        xs2 = np.arange(len(dr_keys))
        vals2 = [dr_grp[k]["drpol_per_100efc_med"] for k in dr_keys]
        axes[1, 1].bar(xs2, vals2, 0.6,
                       color=[colors.get(int(k[1:]), "k") for k in dr_keys])
        axes[1, 1].set_xticks(xs2, [f"{k}\n({dr_grp[k]['n']})" for k in dr_keys],
                              fontsize=8)
    axes[1, 1].set_ylabel("ΔR_pol / 100 EFC  [mΩ]")
    axes[1, 1].set_title("impedance growth per mileage (BoL→EoL)")
    fig.tight_layout()
    save_fig(fig, OUT_F / "52a_efc_rate.png")

    # ---- 4. 判词（全部从计算值生成） --------------------------------
    g5 = greq.get("g5") or {}
    g6 = greq.get("g6") or {}
    g14 = {
        "n": int(rates[rates.group.isin([1, 2, 3, 4])].shape[0]),
        "efc": float(np.nanmedian(rates[rates.group.isin([1, 2, 3, 4])].efc_total)),
        "sec": float(np.nanmedian(rates[rates.group.isin([1, 2, 3, 4])].rate_efc_secant)),
        "day": float(np.nanmedian(rates[rates.group.isin([1, 2, 3, 4])].rate_day_secant)),
    }
    efc5 = g5.get("efc_total_med") or float("nan")
    efc14 = g14["efc"]
    sec5 = g5.get("rate_efc_secant_med") or float("nan")
    sec14 = g14["sec"]
    day5 = g5.get("rate_day_secant_med") or float("nan")
    day6 = g6.get("rate_day_secant_med") or float("nan")
    mile_ratio = efc5 / efc14 if efc14 else float("nan")
    rate_ratio = sec5 / sec14 if sec14 else float("nan")

    verdict = (
        f"EFC/时间轴归一化收口（{len(rates)} 芯、{len(tp)} 块全量解析）——"
        f"「61% vs 4%」分解为里程×速率："
        f"组 5 连续循环 EFC 中位 {efc5:.0f} vs 组 1–4 组合 {efc14:.0f}"
        f"（里程比 {mile_ratio:.1f}×）；"
        f"同里程 fade 速率（fade/100 EFC 割线中位）组 5 {sec5:.2f}% vs "
        f"组 1–4 {sec14:.2f}%（速率比 {rate_ratio:.2f}）——"
        + (
            "速率比≈1 则 61% 主要是里程差（跑得多死得快），"
            "组 5 的单位里程退化并不更快；"
            if abs(rate_ratio - 1) < 0.25 else
            ("组 5 单位里程退化也更快（速率×里程双杀）；"
             if rate_ratio > 1.25 else
             "组 5 单位里程退化反而更慢（单纯里程驱动）；")
        )
        + f"组 6 日历 fade/100 天 {day6:.3f}% vs 组 5 {day5:.2f}%"
        f"（组 6 EFC 中位 {(g6.get('efc_total_med') or float('nan')):.2f}"
        f"≈0——日历老化是时间驱动非通量驱动）。"
        + (
            f"ΔR_pol/100 EFC：" +
            "，".join(
                f"组 {k[1:]} {dr_grp[k]['drpol_per_100efc_med']:.1f} mΩ"
                for k in sorted(dr_grp)
            ) + "——" +
            (
                "跨工况量级接近=阻抗增长与里程近似成比例"
                if (max(v["drpol_per_100efc_med"] for v in dr_grp.values())
                    / max(min(v["drpol_per_100efc_med"]
                              for v in dr_grp.values()), 1e-9) < 2)
                else "跨工况差异显著=阻抗机制配比不同（49 三通道）"
            ) + "。"
            if dr_grp else "ΔR_pol/EFC 待 EIS 缓存。"
        )
        + f"双估计器（∫|I|dt vs 分步 Amphr）中位相对差 "
        f"{dual['rel_diff_med']:.1%}、>5% 占 {dual['frac_over_5pct']:.0%}"
        f"（Amphr 步内累积口径核对）。"
        f"⚠️ EFC=放电 Ah/q_bol 口径；合并块（Profile+多 RPT）容量点"
        f"取块内最大 CC 放电段（≈首个 RPT，块粒度近似）；"
        f"组 6 n=1 个案；EoL 标签块计吞吐不计容量点（与 50 同口径）。"
    )

    metrics = {
        "n_cells": int(len(rates)),
        "n_blocks": int(len(tp)),
        "n_q_points": int(len(traj)),
        "dual_estimator": dual,
        "group_rates": greq,
        "g1_4_pooled": g14,
        "drpol_per_efc": dr_grp,
        "caveats": [
            "EFC=放电 Ah/q_bol；块级累积+容量点对齐段结束点",
            "合并块（Profile+RPT）q 点=块内最大 CC 段（块粒度近似）",
            "组 6 n=1 个案；EoL 标签块计吞吐不计容量点",
            "TestTime 分钟制；Amphr 步内累积（双估计器口径）",
        ],
        "verdict": verdict,
    }
    (OUT_M / "52_efc_rate.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[4] 组速率: {json.dumps({k: greq[k] for k in list(greq)[:3]}, ensure_ascii=False)[:200]}")
    print(f"[5] ΔR/100EFC: {dr_grp}")
    print(f"[6] {verdict[:200]}")
    print(f"[7] → {OUT_M / '52_efc_rate.json'}")


if __name__ == "__main__":
    main()
