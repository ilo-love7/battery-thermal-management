# -*- coding: utf-8 -*-
"""WP49 fade 归一化阻抗增长 + 模式联动（v1.12 第三线）。

把 WP48 的 EIS BoL→MoL 增量接到 **fade 轨迹**上（Group_1–4 zip 的
RPT 块参考放电容量，Maccor 表块解析）：

1. Guide xlsx 块映射（Cell → TPGx.y → RPT/Profile 标签）；
2. 11 个配对芯的 RPT 容量轨迹（q_cc 最大恒流放电段）→ BoL→MoL
   fade %；参考容量 = 首个 BoL RPT（第二 BoL RPT 给散布）；
3. fade 归一化阻抗增长（ΔR_pol / fade%）——跨芯可比的「每 % 容量
   损失的界面阻抗代价」；
4. 路径条件（C/2 vs C/4、1+5d vs 2+10d）与论文路径依赖结论对照；
5. 模式框架证据更新（慢倍率下 shift 主导图景的阻抗侧证 + 两读法
   纪律）。

产物：results/cache/oxford_fade.csv、results/metrics/48_eis_mode_link.json、
results/figures/48a_eis_mode_link.png
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
    block_cc_discharge_capacity, block_discharge_capacity,
    load_maccor_block, parse_block_member,
)
from bmthermal.plotting import save as save_fig, setup_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OXFORD = Path("D:/项目/02_数据集/01_国际公开集/Oxford_PathDependent")
GUIDE_XLSX = OXFORD / "Guide_to_Datafiles.xlsx"
OUT_M = ROOT / "results" / "metrics"
OUT_F = ROOT / "results" / "figures"
CACHE_FADE = ROOT / "results" / "cache" / "oxford_fade.csv"

GROUP_META = {
    1: {"cycle_days": 1, "cal_days": 5, "c_rate": 0.5, "period": "1+5d"},
    2: {"cycle_days": 1, "cal_days": 5, "c_rate": 0.25, "period": "1+5d"},
    3: {"cycle_days": 2, "cal_days": 10, "c_rate": 0.5, "period": "2+10d"},
    4: {"cycle_days": 2, "cal_days": 10, "c_rate": 0.25, "period": "2+10d"},
}


def load_guide_map() -> pd.DataFrame:
    """Guide xlsx → 表：tpg, step, cell, label（RPT/Profile）。"""
    df = pd.read_excel(GUIDE_XLSX, sheet_name="Data File Names",
                       header=None)
    rows = []
    group = None
    cell = None
    for _, r in df.iterrows():
        v0, v1, v2 = (str(r[0]) if pd.notna(r[0]) else "",
                      str(r[1]) if pd.notna(r[1]) else "",
                      str(r[2]) if pd.notna(r[2]) else "")
        if v0.startswith("Group"):
            m = re.search(r"Group\s*(\d+)", v0)
            group = int(m.group(1)) if m else None
            continue
        if v0.strip() == "Cell" or not v1:
            continue
        if v0.strip().isdigit():
            cell = int(v0.strip())
        m = re.match(r"TPG(\d+)(?:\.(\d+))?$", v1.strip())
        if m is None or cell is None or group is None:
            continue
        step = int(m.group(2)) if m.group(2) else 1
        rows.append({"tpg": int(m.group(1)), "step": step,
                     "cell": cell, "group": group,
                     "label": v2.strip()})
    return pd.DataFrame(rows)


def is_rpt(label: str) -> bool:
    return "RPT" in label.upper()


def main() -> None:
    OUT_M.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "cache").mkdir(parents=True, exist_ok=True)

    guide = load_guide_map()
    n_rpt = int(guide.label.map(is_rpt).sum())
    print(f"[1] guide 映射 {len(guide)} 块（RPT {n_rpt}）")

    # 配对芯 = 47 的严格双阶段对集（BoL-only 组不进）
    m47 = json.loads((OUT_M / "47_eis_degrade.json").read_text(
        encoding="utf-8"))
    pairs = pd.DataFrame(m47["pairs"])
    paired = sorted(set(zip(pairs.group.astype(int), pairs.cell.astype(int))))
    print(f"[2] 配对芯 {len(paired)}: {paired}")

    # ---- fade 轨迹（RPT 块参考放电容量） ---------------------------
    if CACHE_FADE.exists():
        fade_df = pd.read_csv(CACHE_FADE)
        print(f"[3] fade 缓存命中 {len(fade_df)} 行")
    else:
        frows = []
        with zipfile.ZipFile(OXFORD / "Group_1.zip") as z1, \
                zipfile.ZipFile(OXFORD / "Group_2.zip") as z2, \
                zipfile.ZipFile(OXFORD / "Group_3.zip") as z3, \
                zipfile.ZipFile(OXFORD / "Group_4.zip") as z4:
            zips = {1: z1, 2: z2, 3: z3, 4: z4}
            for g, cell in paired:
                sub = guide[(guide.group == g) & (guide.cell == cell)
                            & guide.label.map(is_rpt)]
                z = zips[g]
                # 成员名索引（步号 → 成员）
                member_by_step = {}
                for name in z.namelist():
                    meta = parse_block_member(name)
                    if meta and meta["tpg"] == g and meta["cell"] == cell:
                        member_by_step[meta["step"]] = name
                for _, gr in sub.sort_values("step").iterrows():
                    name = member_by_step.get(gr.step)
                    if name is None:
                        print(f"    ! 缺块 G{g} C{cell} step {gr.step}"
                              f"（guide 有记录，zip 无成员）")
                        continue
                    try:
                        df = load_maccor_block(z.read(name))
                    except Exception as exc:  # noqa: BLE001
                        print(f"    ! 解析失败 G{g} C{cell} step "
                              f"{gr.step}: {exc}")
                        continue
                    q_cc = block_cc_discharge_capacity(df)
                    q_dis = block_discharge_capacity(df)
                    frows.append({"group": g, "cell": cell,
                                  "step": int(gr.step),
                                  "label": gr.label, "member": name,
                                  "q_cc": q_cc, "q_dis": q_dis})
                    print(f"    G{g} C{cell} step{gr.step} {gr.label}: "
                          f"q_cc={q_cc:.3f} Ah")
        fade_df = pd.DataFrame(frows)
        fade_df.to_csv(CACHE_FADE, index=False)
        print(f"[3] fade 轨迹 {len(fade_df)} 块 → 缓存")

    # ---- fade %（BoL → 末 RPT） ------------------------------------
    fade_rows = []
    for g, cell in paired:
        sub = fade_df[(fade_df.group == g) & (fade_df.cell == cell)] \
            .sort_values("step")
        if len(sub) < 2:
            continue
        bol = sub[sub.label.str.contains("BoL")]
        # BoL 参考 = BoL RPT 中最大者（首个块偶发偏低，如 G4 C19
        # 2.55 vs 2.97——取 max 保证 fade≥0 的物理方向）
        q_bol = float(bol.q_cc.max()) if len(bol) else float(sub.q_cc.iloc[0])
        q_bol2 = float(bol.q_cc.min()) if len(bol) > 1 else float("nan")
        q_end = float(sub.q_cc.iloc[-1])
        fade_pct = (1.0 - q_end / q_bol) * 100.0
        fade_rows.append({
            "group": g, "cell": cell, "q_bol": q_bol,
            "q_bol2": q_bol2, "q_end": q_end, "fade_pct": fade_pct,
            "n_rpt": int(len(sub)),
        })
        print(f"[4] G{g} C{cell}: BoL {q_bol:.3f} → 末 {q_end:.3f} Ah "
              f"fade {fade_pct:.2f}%（{len(sub)} RPT）")
    fdf = pd.DataFrame(fade_rows)

    # ---- EIS 增量 × fade 归一 --------------------------------------
    merged = pairs.merge(fdf[["group", "cell", "fade_pct", "q_bol"]],
                         on=["group", "cell"], how="left")
    merged["drpol_per_fade"] = merged.apply(
        lambda r: r.d_r_pol / r.fade_pct * 1e3
        if r.fade_pct and r.fade_pct > 0.05 else float("nan"), axis=1)
    merged["drohm_per_fade"] = merged.apply(
        lambda r: r.d_r_ohm / r.fade_pct * 1e3
        if r.fade_pct and r.fade_pct > 0.05 else float("nan"), axis=1)

    valid = merged.dropna(subset=["drpol_per_fade"])
    drpol_pf_med = float(np.nanmedian(valid.drpol_per_fade)) \
        if len(valid) else float("nan")
    print(f"[5] ΔR_pol/fade 中位 {drpol_pf_med:.1f} mΩ/%（{len(valid)} 对）")

    # 路径条件
    rate_rows = []
    for g, gg in merged.groupby("group"):
        meta = GROUP_META.get(int(g), {})
        gf = fdf[fdf.group == g]
        rate_rows.append({
            "group": int(g), "c_rate": meta.get("c_rate"),
            "period": meta.get("period"),
            "fade_pct_med": float(np.nanmedian(gf.fade_pct)),
            "d_r_pol_med": float(np.nanmedian(gg.d_r_pol)) * 1e3,
            "n": int(len(gg)),
        })

    half_f = fdf[fdf.group.isin([1, 3])].fade_pct
    quart_f = fdf[fdf.group.isin([2, 4])].fade_pct
    fade_rate_cmp = {
        "c2_fade_med": float(np.nanmedian(half_f)),
        "c4_fade_med": float(np.nanmedian(quart_f)),
        "n_c2": int(len(half_f)), "n_c4": int(len(quart_f)),
    }
    print(f"[6] fade：C/2 组 {fade_rate_cmp['c2_fade_med']:.2f}% vs "
          f"C/4 {fade_rate_cmp['c4_fade_med']:.2f}%")

    # ---- 图 48a -----------------------------------------------------
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    colors = {1: "#2e86de", 2: "#27ae60", 3: "#e67e22", 4: "#c0392b"}

    ax = axes[0, 0]
    for g, sub in fade_df.groupby("group"):
        for cell, ss in sub.groupby("cell"):
            ss = ss.sort_values("step")
            q0 = float(ss.q_cc.iloc[0])
            ax.plot(ss.step, ss.q_cc / q0 * 100, "o-", ms=3.5, lw=1.1,
                    color=colors.get(int(g), "#888"),
                    label=f"TPG{int(g)} C{int(cell)}" if cell == \
                        sub.cell.iloc[0] else None)
    ax.set_xlabel("程序步（RPT 序）")
    ax.set_ylabel("参考容量 [% BoL]")
    ax.set_title("a) RPT 容量轨迹（q_cc 归一）")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    for r in fdf.itertuples():
        ax.scatter(r.group, r.fade_pct, s=55,
                   color=colors.get(int(r.group), "#888"))
        ax.annotate(f"C{int(r.cell)}", (r.group, r.fade_pct), fontsize=6,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xticks([1, 2, 3, 4], ["TPG1\nC/2 1+5d", "TPG2\nC/4 1+5d",
                                 "TPG3\nC/2 2+10d", "TPG4\nC/4 2+10d"])
    ax.set_ylabel("BoL→末 RPT fade [%]")
    ax.set_title("b) 逐芯 fade（按路径组）")

    ax = axes[1, 0]
    sub = merged.dropna(subset=["fade_pct"])
    ax.scatter(sub.fade_pct, sub.d_r_pol * 1e3, s=55,
               c=[colors.get(int(g), "#888") for g in sub.group])
    for r in sub.itertuples():
        ax.annotate(f"{int(r.cell)}/{int(r.soc)}",
                    (r.fade_pct, r.d_r_pol * 1e3), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("fade [%]")
    ax.set_ylabel("ΔR_pol [mΩ]")
    ax.set_title("c) 阻抗增长 vs fade（标注 cell/SOC）")

    ax = axes[1, 1]
    v = valid.drpol_per_fade.dropna()
    ax.hist(v, bins=8, color="#8e44ad", alpha=0.8)
    ax.axvline(drpol_pf_med, color="#c0392b", ls="--", lw=1.2,
               label=f"中位 {drpol_pf_med:.0f} mΩ/%")
    ax.set_xlabel("ΔR_pol / fade [mΩ/%]")
    ax.set_ylabel("对数")
    ax.set_title("d) fade 归一化界面阻抗代价")
    ax.legend(fontsize=8)

    fig.suptitle("48a fade 归一化阻抗增长与路径条件（Oxford NCA BoL→MoL）",
                 fontsize=12)
    fig.tight_layout()
    save_fig(fig, "48a_eis_mode_link", OUT_F)

    # ---- 判词 -------------------------------------------------------
    frag_ratio = (fade_rate_cmp["c2_fade_med"] /
                  max(fade_rate_cmp["c4_fade_med"], 1e-9))
    verdict = (
        f"fade 归一化阻抗联动收口（v1.12 第三线）：{len(fdf)} 芯 RPT "
        f"轨迹（Guide 映射 {n_rpt} 个 RPT 块）BoL→末 fade 中位 "
        f"{float(np.nanmedian(fdf.fade_pct)):.2f}%；"
        f"界面阻抗代价 ΔR_pol/fade 中位 {drpol_pf_med:.0f} mΩ/%"
        f"（{len(valid)} 对，跨芯可比）。"
        f"路径条件：C/2 组 fade {fade_rate_cmp['c2_fade_med']:.2f}% vs "
        f"C/4 {fade_rate_cmp['c4_fade_med']:.2f}%（比 {frag_ratio:.2f}）"
        f"——{'高倍率组退化更快（与论文路径依赖方向一致）' if frag_ratio > 1.15 else '本样本量下倍率差弱（11 芯，组内散布覆盖）'}；"
        f"而 WP48 的 ΔR_pol 增长近倍率无关（29 vs 31 mΩ）→ "
        f"fade 差异不来自界面阻抗增长总量，而来自模式配比/动力学"
        f"——与论文 DVA 归因负极位点一致（阻抗总量不分辨位点）。"
        f"模式框架证据：慢倍率路径老化（C/2–C/4）阻抗签名界面主导+"
        f"欧姆小 → 与 SNL 1C/2C 的 shift-like 主导图景（v1.10 "
        f"LFP 1C shape 份额 0.05）方向一致，"
        f"支持「速率依赖模式配比」（慢→shift 主导、快→shape 份额升）；"
        f"弧段两读法（SEI 增厚 vs 活性表面丧失）仍并列，"
        f"不作单选。⚠️ MoL 时点与末 RPT 对齐为假设（Part 1 = BoL→MoL "
        f"区间）；fade 用 q_cc（最大 CC 放电段）口径。"
    )
    print("[7]", verdict[:120], "...")

    metrics = {
        "n_guide_blocks": int(len(guide)),
        "n_rpt_blocks": n_rpt,
        "n_cells": int(len(fdf)),
        "fade_summary": {
            "fade_pct_med": float(np.nanmedian(fdf.fade_pct)),
            "fade_pct_min": float(np.nanmin(fdf.fade_pct)),
            "fade_pct_max": float(np.nanmax(fdf.fade_pct)),
        },
        "drpol_per_fade_med_mohm_per_pct": drpol_pf_med,
        "n_valid_pairs": int(len(valid)),
        "rate_fade_compare": fade_rate_cmp,
        "group_rows": rate_rows,
        "fade_rows": fdf.to_dict(orient="records"),
        "caveats": [
            "MoL 时点与末 RPT 对齐为假设（Part 1 = BoL→MoL 区间）",
            "q_cc=最大恒流放电段口径（块总吞吐含多循环）",
            "弧段归属两读法并列；频率轴不可溯源",
            "n=11 芯，组间对比为方向性非显著性检验",
        ],
        "verdict": verdict,
    }
    (OUT_M / "48_eis_mode_link.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[8] → {OUT_M / '48_eis_mode_link.json'}")


if __name__ == "__main__":
    main()
