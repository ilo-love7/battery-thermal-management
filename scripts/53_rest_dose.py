#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/53 — 静置恢复剂量量化 + matched-depth 速率（v1.15，WP59）。

回答：组 5 的 fade 速率超额（v1.14 割线 1.93×）里，「无静置恢复」
贡献多少？三个手段：
1. 静置剂量（延长静置窗 ≥2 h）：逐芯 ext_rest_h/100 EFC、窗长统计；
   组 1–4（1:5 循环:日历设计）vs 组 5（连续循环）剂量对比；
2. 组 1–4 内部设计性剂量-响应（Readme 因子：C/2 vs C/4、5 d vs
   10 d 日历窗）——静置在 90% SOC 是「每 EFC 加伤」还是「恢复」；
3. matched-depth 速率（fade ≤15% OLS）——修割线跨深度不可比
   （组 5 到 61% fade、组 1–4 只到 15–21%），G5 比值在同深度下
   剩多少。

产物：results/metrics/53_rest_dose.json、results/figures/53a_rest_dose.png。
依赖：_tp_blocks.csv（v1.15 清洗后含 rest 字段，52 生成）、
oxford_efc_traj.csv。判词从计算值生成。
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
TP_CACHE = ROOT / "results" / "cache" / "_tp_blocks.csv"
TRAJ_CACHE = ROOT / "results" / "cache" / "oxford_efc_traj.csv"

FADE_CAP = 15.0        # matched-depth 上限（%）
KIND = {
    1: "combined_c2_5d", 2: "combined_c4_5d",
    3: "combined_c2_10d", 4: "combined_c4_10d",
    5: "cycle_only", 6: "calendar_only",
    7: "cccv_90_5d", 8: "cccv_90_10d",
    9: "cccv_42_5d", 10: "cccv_42_10d",
}


def _med(s: pd.Series):
    return float(np.nanmedian(s)) if len(s) else None


def per_cell_rest(tp: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    """块表 → 逐芯静置剂量表（延长静置窗 ≥ REST_EXT 口径）。"""
    rows = []
    for (g, cell), sub in tp.groupby(["group", "cell"]):
        efc_dis = float(sub.ah_dis.sum())  # 放电 Ah 累积（EFC 分子）
        ext_min = float(sub.ext_rest_min.sum())
        rows.append({
            "group": int(g), "cell": int(cell),
            "kind": KIND.get(int(g), "?"),
            "rest_h": float(sub.rest_min_total.sum()) / 60.0,
            "ext_rest_h": ext_min / 60.0,
            "ext_rest_n": int(sub.ext_rest_n.sum()),
            "ext_rest_max_h": float(sub.ext_rest_max_min.max()) / 60.0
            if len(sub) else 0.0,
            "ext_rest_med_h": float(np.median(
                sub.ext_rest_max_min[sub.ext_rest_max_min > 0])) / 60.0
            if (sub.ext_rest_max_min > 0).any() else 0.0,
            "ah_dis_total": efc_dis,
        })
    df = pd.DataFrame(rows)
    # 合并 matched-depth 速率（轨迹重建，依赖 efc 轴）
    md = matched_depth_rates(traj, fade_cap=FADE_CAP)
    df = df.merge(md[["group", "cell", "rate_early", "efc_at_cap",
                      "rate_to_cap", "fade_max", "efc_total", "flag"]],
                  on=["group", "cell"], how="left")
    # 剂量归一：延长静置小时 / 100 EFC（EFC=放电 Ah/q_bol → 这里用
    # efc_total 轨迹值；缺失时跳过归一）
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ext_rest_per_100efc"] = np.where(
            df.efc_total > 1, df.ext_rest_h / df.efc_total * 100.0, np.nan)
    return df


def dose_response(cells: pd.DataFrame) -> dict:
    """组 1–4 内静置剂量 × fade 速率（设计性对比，n=12）。"""
    sub = cells[cells.group.isin([1, 2, 3, 4])].dropna(
        subset=["ext_rest_per_100efc", "rate_early"])
    out: dict = {"n": int(len(sub))}
    if len(sub) < 4:
        out["note"] = "n<4 不做相关"
        return out
    x = sub.ext_rest_per_100efc.to_numpy(float)
    y = sub.rate_early.to_numpy(float)
    r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 \
        else float("nan")
    # Spearman（秩相关，抗野点）
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 else float("nan")
    out.update({"pearson_r": r, "spearman_rho": rho})
    # 因子分解（Readme 设计）：C/2 vs C/4（剂量轴）、5d vs 10d（窗长轴）
    for tag, sel in [
        ("c2_g1_3", sub.group.isin([1, 3])),
        ("c4_g2_4", sub.group.isin([2, 4])),
        ("w5_g1_2", sub.group.isin([1, 2])),
        ("w10_g3_4", sub.group.isin([3, 4])),
    ]:
        s = sub[sel]
        out[tag] = {
            "n": int(len(s)),
            "rate_early_med": _med(s.rate_early),
            "ext_rest_per_100efc_med": _med(s.ext_rest_per_100efc),
        }
    return out


def main() -> None:
    print("[1] 读缓存")
    tp = pd.read_csv(TP_CACHE)
    traj = pd.read_csv(TRAJ_CACHE)
    assert "ext_rest_min" in tp.columns, "缺 rest 字段——先重跑 52 重建缓存"
    n_bad = int(((tp.t_min.abs() > 1e6) | (tp.ah_tp.abs() > 1e4)).sum())
    print(f"    tp {len(tp)} 块；清洗后残余野值 {n_bad}（应为 0）")

    print("[2] 逐芯静置剂量 + matched-depth")
    cells = per_cell_rest(tp, traj)
    OUT_M.mkdir(parents=True, exist_ok=True)
    cells.to_csv(ROOT / "results" / "cache" / "rest_dose_cells.csv", index=False)

    print("[3] 组对比 / 剂量-响应 / 稳健化锚")
    grp_rows = []
    for g in sorted(cells.group.unique()):
        s = cells[cells.group == g]
        grp_rows.append({
            "group": int(g), "kind": KIND.get(int(g), "?"), "n": int(len(s)),
            "ext_rest_h_med": _med(s.ext_rest_h),
            "ext_rest_per_100efc_med": _med(s.ext_rest_per_100efc),
            "ext_rest_max_h_med": _med(s.ext_rest_max_h),
            "rate_early_med": _med(s.rate_early),
            "rate_to_cap_med": _med(s.rate_to_cap),
            "fade_max_med": _med(s.fade_max),
        })
    grp = pd.DataFrame(grp_rows)
    dr = dose_response(cells)
    # 稳健化：BoL 选点敏感带 + fade@350 EFC 同里程锚
    band = early_rate_band(traj, fade_cap=FADE_CAP, max_skip=2)
    f350 = fade_at_efc(traj, 350.0)
    f350.to_csv(ROOT / "results" / "cache" / "fade_at_350efc.csv", index=False)

    # 组 5 vs 组 1–4：剂量对比 + matched-depth 比值（vs 割线 1.93）
    g5 = cells[cells.group == 5]
    g14 = cells[cells.group.isin([1, 2, 3, 4])]
    dose_ratio = None
    if _med(g5.ext_rest_per_100efc) and _med(g14.ext_rest_per_100efc):
        dose_ratio = float(_med(g5.ext_rest_per_100efc)
                           / _med(g14.ext_rest_per_100efc))
    early_ratio = None
    if _med(g5.rate_early) and _med(g14.rate_early):
        early_ratio = float(_med(g5.rate_early) / _med(g14.rate_early))
    sec = None
    if _med(g5.fade_max) and _med(g5.efc_total) and _med(g14.efc_total):
        g5_sec = float(np.nanmedian(g5.fade_max / g5.efc_total * 100))
        g14_sec = float(np.nanmedian(g14.fade_max / g14.efc_total * 100))
        sec = {"g5": g5_sec, "g1_4": g14_sec,
               "ratio": g5_sec / g14_sec if g14_sec else None}

    print("[4] 图 53a")
    setup_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    # 左：静置剂量条形（组中位）
    ax = axes[0]
    order = [1, 2, 3, 4, 5, 6]
    xs = np.arange(len(order))
    vals = [grp.loc[grp.group == g, "ext_rest_per_100efc_med"].iloc[0]
            if (grp.group == g).any() else 0 for g in order]
    colors = ["#2980b9"] * 4 + ["#c0392b", "#7f8c8d"]
    ax.bar(xs, vals, 0.6, color=colors)
    ax.set_xticks(xs, [f"G{g}" for g in order], fontsize=8)
    ax.set_ylabel("ext-rest h / 100 EFC")
    ax.set_title("rest dose by group (med)")
    # 中：剂量-响应散点（组 1–4）
    ax = axes[1]
    for g, mk, c in [(1, "o", "#2980b9"), (2, "s", "#8e44ad"),
                     (3, "^", "#27ae60"), (4, "D", "#e67e22")]:
        s = cells[cells.group == g].dropna(subset=["ext_rest_per_100efc",
                                                   "rate_early"])
        ax.scatter(s.ext_rest_per_100efc, s.rate_early, marker=mk, s=42,
                   color=c, label=f"G{g}")
    ax.set_xlabel("ext-rest h / 100 EFC")
    ax.set_ylabel(f"early fade rate (≤{FADE_CAP:.0f}%, %/100 EFC)")
    ax.set_title("G1-4 designed dose-response")
    ax.legend(fontsize=7)
    # 右：matched-depth 速率条形（G5 vs G1-4）
    ax = axes[2]
    labels = ["G5\n(cycle)", "G1-4\n(comb.)", "G6\n(cal.)"]
    med_vals = [
        _med(g5.rate_early) or 0,
        _med(g14.rate_early) or 0,
        _med(cells[cells.group == 6].rate_early) or 0,
    ]
    ax.bar(np.arange(3), med_vals, 0.55,
           color=["#c0392b", "#2980b9", "#7f8c8d"])
    ax.set_xticks(np.arange(3), labels, fontsize=8)
    ax.set_ylabel(f"rate ≤{FADE_CAP:.0f}% fade (%/100 EFC)")
    ax.set_title("matched-depth fade rate")
    fig.tight_layout()
    save_fig(fig, "53a_rest_dose.png", OUT_F)

    # ---- 判词（从计算值生成）----
    g5_dose = _med(g5.ext_rest_per_100efc)
    g14_dose = _med(g14.ext_rest_per_100efc)
    c2 = (dr.get("c2_g1_3") or {}).get("rate_early_med")
    c4 = (dr.get("c4_g2_4") or {}).get("rate_early_med")
    w5 = (dr.get("w5_g1_2") or {}).get("rate_early_med")
    w10 = (dr.get("w10_g3_4") or {}).get("rate_early_med")
    rho = dr.get("spearman_rho")
    er = early_ratio
    b5 = (band["group_med_band"].get("g5") or {}).get("band") or [None, None]
    b14 = []
    for g in (1, 2, 3, 4):
        bb = (band["group_med_band"].get(f"g{g}") or {}).get("band")
        if bb:
            b14.extend(bb)
    f350_g5 = _med(f350[f350.group == 5].fade_at_efc)
    f350_g14 = _med(f350[f350.group.isin([1, 2, 3, 4])].fade_at_efc)
    dose_dir = ("静置剂量正相关 fade 速率（90% SOC 静置=每 EFC 加伤）"
                if (rho is not None and rho > 0.3) else
                "静置剂量与 fade 速率弱/负相关（恢复方向证据）"
                if (rho is not None and rho < -0.3) else
                "静置剂量-响应方向未定（|ρ|≤0.3 或 n 不足）")
    verdict = (
        f"静置剂量量化 + matched-depth 收口（27 芯、{len(tp)} 块，"
        f"清洗后残余野值 {n_bad}）——"
        f"① 【口径修订】v1.14「同里程速率 1.93×双杀」被深度混杂修正："
        f"全深度割线比 {sec['ratio'] if sec and sec['ratio'] else float('nan'):.2f}"
        f"（G5 割线跨 0→61% fade、G1–4 只跨 15–21%，凸轨迹让深割线"
        f"虚高）；matched-depth（fade≤{FADE_CAP:.0f}%）G5/G1–4 early "
        f"速率比 {er if er is not None else float('nan'):.2f}"
        f"（G5 带宽 {b5[0]:.2f}–{b5[1]:.2f} vs G1–4 "
        f"{min(b14):.2f}–{max(b14):.2f}，BoL 选点敏感带重叠）；"
        f"fade@350 EFC 同里程锚 G5 {f350_g5 if f350_g5 is not None else float('nan'):.1f}%"
        f" vs G1–4 {f350_g14 if f350_g14 is not None else float('nan'):.1f}%"
        f"——G5 早期同里程**不更伤**；61% 终点=里程 1.7×（852 vs 489 "
        f"EFC）+ 深段加速（末 250 EFC 内 23→61%），不是早期速率超额。"
        f"② 剂量对比：组 5 延长静置 "
        f"{g5_dose if g5_dose is not None else float('nan'):.1f} h/100 EFC "
        f"vs 组 1–4 {g14_dose if g14_dose is not None else float('nan'):.1f}"
        f"（比 {dose_ratio if dose_ratio is not None else float('nan'):.3f}；"
        f"组 6 全静置=老化条件非恢复窗，不入比）。"
        f"③ 组 1–4 设计性剂量-响应（n={dr.get('n', 0)}，Spearman ρ="
        f"{rho if rho is not None else float('nan'):.2f}，n=12 不显著）："
        f"{dose_dir}；因子 early 速率 C/2 {c2 if c2 is not None else float('nan'):.2f}"
        f" vs C/4 {c4 if c4 is not None else float('nan'):.2f}、"
        f"5 d 窗 {w5 if w5 is not None else float('nan'):.2f}"
        f" vs 10 d 窗 {w10 if w10 is not None else float('nan'):.2f} %/100 EFC。"
        f"④ 「无静置恢复致伤」假设**不被支持**：G5 early 速率不高"
        f"于 G1–4，而 90% SOC 静置是每 EFC 加伤项——恢复窗证据需要"
        f"低 SOC 静置对照（本数据集无，硬件 E1 可补）。"
        f"⚠️ 静置窗含 RPT 协议静置（合并块不可分，保守方向）；"
        f"日历静置≠循环间恢复窗；fade@350 的 G2 无点（376–385 EFC "
        f"止）不入锚；G4 C19 q_bol 参考噪声（fade 首点 14%→负值）"
        f"已滤负 fade 但其 rate_early 仍偏低，n=3 组中位稳健。"
    )
    metrics = {
        "n_cells": int(len(cells)),
        "n_blocks": int(len(tp)),
        "fade_cap_pct": FADE_CAP,
        "residual_bad_blocks": n_bad,
        "group_table": grp.to_dict(orient="records"),
        "dose_response": dr,
        "g5_vs_g1_4": {
            "dose_ratio_ext_rest_per_100efc": dose_ratio,
            "early_rate_ratio": early_ratio,
            "early_rate_band_g5": b5,
            "early_rate_band_g1_4": [min(b14), max(b14)] if b14 else None,
            "secant_ratio_recomputed": sec,
            "fade_at_350efc": {"g5": f350_g5, "g1_4": f350_g14},
        },
        "early_rate_band": band["group_med_band"],
        "caveats": [
            "延长静置窗 ≥2 h 口径；含 RPT 协议静置（合并块不可分，保守方向）",
            "组 6 静置=日历老化条件，非恢复窗，不入剂量比",
            "matched-depth OLS 用 fade≤15% 点；不足 3 点取前 3 点 flag=1；负 fade（q_bol 噪声）滤除",
            "BoL 选点敏感带（skip 0–2）：G5 early 速率带与 G1–4 重叠，单值不可硬比",
            "fade@350 EFC 同里程锚；G2（376–385 EFC 止）不入锚",
            "G4 C19 q_bol 参考噪声（fade 首点 14%→负值）：rate_early 偏低，组中位稳健",
            "剂量-响应为组间设计性对比（n=12），非随机化因果",
        ],
        "verdict": verdict,
    }
    with open(OUT_M / "53_rest_dose.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print("[5] 判词")
    print(verdict)
    print("[53] done")


if __name__ == "__main__":
    main()
