#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""scripts/56 — 深段 V(Q) 两模式分解轨迹（v1.16，WP62）。

回答 v1.15 收尾的机制侧第二问：G5 深段加速（fade>25%）时，无损
模式配比（shift=LLI 代理 / shape=LAM 代理，几何归属口径见
degrade_modes）**随深度怎么变**——shape 份额深段抬升（LAM 加速）
还是 shift 一以贯之（LLI 主导）？

1. G5 三芯（唯一样深段）：逐块 RPT V(q)（合并块内 C/25 细采样段，
   block_rpt_curves）对 BoL 参考做 mode_split——share_shape vs
   fade_pct 轨迹，浅段（≤15%）vs 深段（>25%）份额对比；
2. G1 浅段对照（3 芯，fade≤21%）：同口径 share_shape 轨迹基线
   ——深段行为是否超出浅段趋势外推；
3. 判词：深段加速的模式侧归属（shape 份额抬升=类 LAM 加速 / 平
   =LLI 累积以外另有机制）。

⚠️ 口径：两模式是曲线几何归属非拆解位点（degrade_modes 文档警示，
与 v1.10 同源）；合并块内多段 RPT 同块（同工况窗）不同深度，段间
独立性弱于块间（散布给带不给点）；q_bol=BoL 块最大段（48 的坑十
三纪律）；边界解/低质量段（ModeSplit.low_quality）剔除后聚合。
判词从计算值生成。

产物：results/metrics/56_deep_modes.json、results/cache/deep_modes_cells.csv、
results/figures/56a_deep_modes.png。
依赖：Group_1.zip / Part2/Group_5.zip（块解析）、eis_features.block_rpt_curves、
degrade_modes.mode_split_trajectory。
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal.degrade_modes import mode_split_trajectory  # noqa: E402
from bmthermal.eis_features import (  # noqa: E402
    block_rpt_curves, load_maccor_block,
)
from bmthermal.plotting import save as save_fig, setup_style  # noqa: E402

OUT_M = ROOT / "results" / "metrics"
OUT_F = ROOT / "results" / "figures"
CACHE = ROOT / "results" / "cache" / "deep_modes_cells.csv"

OXFORD = Path("D:/项目/02_数据集/01_国际公开集/Oxford_PathDependent")

# 芯 → zip 路径（成员名按 parse_block_member 枚举，不硬编码步号——
# G1 成员名空格不规则：TPG1 - Cell 9 / TPG1.8-Cell9 混排）
CELL_ZIPS = {
    (5, 5): OXFORD / "Part2" / "Group_5.zip",
    (5, 6): OXFORD / "Part2" / "Group_5.zip",
    (5, 16): OXFORD / "Part2" / "Group_5.zip",
    (1, 9): OXFORD / "Group_1.zip",
    (1, 15): OXFORD / "Group_1.zip",
    (1, 20): OXFORD / "Group_1.zip",
}


def _med(s: pd.Series):
    return float(np.nanmedian(s)) if len(s) and s.notna().any() else None


def collect_curves(zip_path: Path, group: int, cell: int) -> list[dict]:
    """逐块提取 RPT V(q) 段（带元数据）。块缺失跳过（如实）。"""
    from bmthermal.eis_features import parse_block_member
    zf = zipfile.ZipFile(zip_path)
    members = []
    for name in zf.namelist():
        if not name.endswith(".mat"):
            continue
        meta = parse_block_member(name)
        if meta and meta["cell"] == cell:
            members.append((meta["step"], name))
    members.sort()
    curves = []
    for step, member in members:
        try:
            raw = zf.read(member)
        except KeyError:
            continue
        df = load_maccor_block(raw)
        segs = block_rpt_curves(df)
        for k, seg in enumerate(segs):
            curves.append({
                "group": group, "cell": cell, "step": step, "seg": k,
                "member": member.split("/")[-1],
                "q": seg["q"], "v": seg["v"], "q_end": seg["q_end"],
                "n_pts": seg["n_pts"],
            })
    zf.close()
    return curves


def main() -> None:
    print("[1] 提取 V(q) 段（G5 3 芯 × 合并块多段 + G1 3 芯对照）")
    all_rows = []
    cell_summaries = []
    for (group, cell), zip_path in CELL_ZIPS.items():
        curves = collect_curves(zip_path, group, cell)
        if len(curves) < 3:
            cell_summaries.append({
                "group": group, "cell": cell, "n_curves": len(curves),
                "note": "段数不足 3",
            })
            continue
        q_bol = max(c["q_end"] for c in curves)
        for c in curves:
            c["fade_pct"] = float((1 - c["q_end"] / q_bol) * 100)
        # BoL 参考 = q_end 最大段
        ref_idx = int(np.argmax([c["q_end"] for c in curves]))
        tr = mode_split_trajectory(curves, ref_idx=ref_idx)
        # 筛：低质量剔除后聚合（保留全表落盘）
        good = tr[~tr.low_quality].copy()
        if good.empty:
            good = tr.copy()
        shallow = good[good.fade_pct <= 15.0]
        deep = good[good.fade_pct > 25.0]
        mid = good[(good.fade_pct > 15.0) & (good.fade_pct <= 25.0)]
        cell_summaries.append({
            "group": group, "cell": cell, "n_curves": int(len(tr)),
            "n_good": int(len(good)),
            "q_bol": float(q_bol),
            "fade_max": float(good.fade_pct.max()),
            "share_shape_shallow": _med(shallow.share_shape),
            "share_shape_mid": _med(mid.share_shape),
            "share_shape_deep": _med(deep.share_shape),
            "n_deep": int(len(deep)),
            "gamma_deep_med": _med(deep.gamma),
            "gamma_shallow_med": _med(shallow.gamma),
        })
        for _, r in tr.iterrows():
            all_rows.append({
                "group": group, "cell": cell,
                "step": int(r.step), "seg": int(r.seg),
                "fade_pct": float(r.fade_pct), "q_end": float(r.q_end),
                "share_shape": float(r.share_shape),
                "q_shift": float(r.q_shift), "q_shape": float(r.q_shape),
                "gamma": float(r.gamma), "r2": float(r.r2),
                "qsum_err": float(r.qsum_err),
                "low_quality": bool(r.low_quality),
                "boundary": str(r.boundary),
                })
    cells_df = pd.DataFrame(all_rows)
    cells_df.to_csv(CACHE, index=False)
    summ = pd.DataFrame(cell_summaries)
    print(f"    段表 {len(cells_df)} 行；芯汇总 {len(summ)} 行")

    print("[2] 浅/深段份额对比")
    g5s = summ[(summ.group == 5) & summ.share_shape_deep.notna()]
    g1s = summ[summ.group == 1]
    deep_med = _med(g5s.share_shape_deep) if len(g5s) else None
    shallow_med = _med(g5s.share_shape_shallow) if len(g5s) else None
    mid_med = _med(g5s.share_shape_mid) if len(g5s) else None
    g1_shallow = _med(g1s.share_shape_shallow) if len(g1s) else None
    g1_all = _med(cells_df[cells_df.group == 1].share_shape) \
        if (cells_df.group == 1).any() else None
    delta = None
    if deep_med is not None and shallow_med is not None:
        delta = float(deep_med - shallow_med)

    print("[3] 图 56a")
    setup_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    ax = axes[0]
    for (g, cell), sub in cells_df.groupby(["group", "cell"]):
        c = "#c0392b" if g == 5 else "#2980b9"
        good = sub[~sub.low_quality]
        ax.plot(good.fade_pct, good.share_shape, marker="o", ms=3.0,
                lw=1.0, alpha=0.7, color=c)
    ax.plot([], [], color="#c0392b", label="G5 (deep)")
    ax.plot([], [], color="#2980b9", label="G1 (shallow ctrl)")
    ax.axvline(25.0, color="#7f8c8d", lw=0.9, ls="--")
    ax.set_xlabel("fade (%)")
    ax.set_ylabel("share_shape (LAM proxy)")
    ax.set_title("mode share vs depth")
    ax.legend(fontsize=7)
    ax = axes[1]
    labels = ["G5 shallow\n(≤15%)", "G5 mid\n(15–25%)", "G5 deep\n(>25%)",
              "G1 all"]
    vals = [shallow_med or 0, mid_med or 0, deep_med or 0, g1_all or 0]
    colors = ["#e67e22", "#e67e22", "#c0392b", "#2980b9"]
    ax.bar(np.arange(4), vals, 0.55, color=colors)
    ax.set_xticks(np.arange(4), labels, fontsize=7.5)
    ax.set_ylabel("share_shape (med)")
    ax.set_title("depth-binned mode share")
    ax = axes[2]
    for (g, cell), sub in cells_df.groupby(["group", "cell"]):
        c = "#c0392b" if g == 5 else "#2980b9"
        good = sub[~sub.low_quality]
        ax.scatter(good.q_shift, good.q_shape, s=18, alpha=0.65, color=c)
    lim = float(np.nanmax(cells_df.q_shift.max())) * 1.1 if len(cells_df) \
        else 1.0
    ax.plot([0, lim], [0, lim], color="#7f8c8d", lw=0.8, ls=":")
    ax.set_xlabel("q_shift (LLI proxy, Ah)")
    ax.set_ylabel("q_shape (LAM proxy, Ah)")
    ax.set_title("shift vs shape loss")
    fig.tight_layout()
    save_fig(fig, "56a_deep_modes.png", OUT_F)

    # ---- 判词（从计算值生成）----
    if delta is not None:
        if delta > 0.05:
            mode_dir = (f"shape 份额深段抬升（浅 {shallow_med:.2f} → 深 "
                        f"{deep_med:.2f}，Δ={delta:+.2f}）——类 LAM 加速")
        elif delta < -0.05:
            mode_dir = (f"shape 份额深段回落（浅 {shallow_med:.2f} → 深 "
                        f"{deep_med:.2f}，Δ={delta:+.2f}）——深段更偏 shift/LLI")
        else:
            mode_dir = (f"shape 份额深段基本不变（浅 {shallow_med:.2f} vs "
                        f"深 {deep_med:.2f}，Δ={delta:+.2f}）——模式配比稳定")
    else:
        mode_dir = "深/浅段份额不可比（一侧无有效段）"
    g1_txt = (f"G1 浅段对照 share_shape 中位 {g1_shallow:.2f}"
              if g1_shallow is not None else "G1 对照不可算")
    if deep_med is not None and g1_shallow is not None and delta is not None:
        if deep_med > g1_shallow + 0.05 and delta > 0.05:
            beyond = "且超出浅段对照水平（深段行为不是浅趋势平移）"
        elif deep_med > g1_shallow + 0.05:
            beyond = "水平高于浅段对照但组内趋势平缓"
        else:
            beyond = "未超出浅段对照水平（深段可由浅趋势外推）"
    else:
        beyond = "对照不可比"
    n_deep_cells = int(len(g5s))
    # 55 的深段加速比（同轮产出，拼合引用——数字从计算值来）
    m55_path = OUT_M / "55_deep_fade.json"
    g5_acc = None
    if m55_path.exists():
        with open(m55_path, encoding="utf-8") as f:
            g5_acc = json.load(f)["g5_vs_g1_4"]["g5_acc_25_100_med"]
    acc_txt = f"{g5_acc:.1f}×" if g5_acc is not None else "未量化"
    if (delta or 0) > 0.05:
        tail_txt = ("有 shape 抬升证据——LAM/活性材料损失加速与阻抗"
                    "尾段主导（49 组 5 尾段 88%）")
    else:
        tail_txt = ("无（或弱）shape 抬升证据——LLI 累积仍主导，深段"
                    "加速或来自损失-阻抗正反馈而非模式切换")
    verdict = (
        f"深段 V(Q) 两模式分解收口（G5 {n_deep_cells} 芯含 >25% 深段、"
        f"G1 3 芯浅段对照；段表 {len(cells_df)} 行，低质量剔除 "
        f"{int(cells_df.low_quality.sum())} 段）——"
        f"① 模式侧：{mode_dir}；{beyond}。"
        f"② {g1_txt}；G5 深段 gamma 中位 "
        f"{_med(g5s.gamma_deep_med) if len(g5s) else float('nan'):.4f}"
        f" vs 浅段 "
        f"{_med(g5s.gamma_shallow_med) if len(g5s) else float('nan'):.4f}。"
        f"③ 与 55 拼合：深段 fade 加速（{acc_txt}）的模式侧"
        f"{tail_txt}。"
        f"⚠️ 两模式=曲线几何归属非拆解位点（Dubarry/Birkl 文献惯例映射）；"
        f"合并块内多段同工况窗（段间独立性弱，份额给中位不给点值）；"
        f"share_shape 定义在 EOD（q_shape/(q_shift+q_shape)）；"
        f"边界解（boundary≠interior）与 low_quality 段剔除后聚合；"
        f"q_bol 取 BoL 块最大段（48 坑十三）；G1 块名 TPG1.4/8/… 不连续"
        f"（Guide 步号），缺失块跳过。"
    )

    metrics = {
        "n_segments": int(len(cells_df)),
        "n_low_quality": int(cells_df.low_quality.sum()) if len(cells_df) else 0,
        "cell_summary": summ.to_dict(orient="records"),
        "g5_shallow_share_med": shallow_med,
        "g5_mid_share_med": mid_med,
        "g5_deep_share_med": deep_med,
        "g5_delta_deep_minus_shallow": delta,
        "g1_shallow_share_med": g1_shallow,
        "n_g5_deep_cells": n_deep_cells,
        "caveats": [
            "两模式=曲线几何归属非拆解位点（LLI/LAM 文献惯例映射）",
            "合并块内多段 RPT 同工况窗，段间独立性弱于块间",
            "share_shape 定义在 EOD；边界解/low_quality 剔除后聚合",
            "q_bol=BoL 块最大段；负 fade 段（q_bol 噪声）不入浅深分箱",
            "G1 块步号不连续（Guide 步号），缺失块跳过",
        ],
        "verdict": verdict,
    }
    OUT_M.mkdir(parents=True, exist_ok=True)
    with open(OUT_M / "56_deep_modes.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print("[4] 判词")
    print(verdict)
    print("[56] done")


if __name__ == "__main__":
    main()
