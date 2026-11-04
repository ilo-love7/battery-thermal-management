# -*- coding: utf-8 -*-
"""WP53 fade 到 EoL + 组 5–10 新工况（v1.13 第二线）。

接 v1.12 的 BoL→MoL fade 归一化（scripts/48），用 Part 2/3 的
RPT 块把退化轨迹推到 EoL，并利用新工况组做**机制解耦**：

1. Guide 2/3 xlsx 块映射（TPGx.y → RPT/Profile/EoL 标签）+ Part 1
   Guide 拼接（RPT1–12 + RPT13–18 全链）；
2. fade 轨迹到 EoL：组 1–4（组合工况）BoL→MoL→EoL 分段 fade 率
   ——退化加速还是放缓（与 49 的阻抗链 63% 后段对拍）；
3. **循环/日历解耦锚**：组 5 纯 CC 循环（C/2 连续）vs 组 6 纯日历
   （90% SOC）——fade 率的机制归属（配合 49 的三通道阻抗解耦）；
4. **CCCV/满充路径修订**：组 7–10（CCCV + 100% SOC 日历）vs 组 1–4
   （CC + 90% SOC）——充电制度/日历 SOC 对 fade 的影响；
5. ΔR_pol/fade 归一化延伸到 EoL（EIS 三阶段 × fade 全链）；
6. v1.12 配比图景「慢→shift 主导、快→shape 份额升」的 fade 侧
   收口（与 49 的阻抗侧解耦 + v1.10 SNL shape 份额三线合流）。

⚠️ EoL 判定依赖 Guide 标签（RPT18/EoL），跨 Part 拼接时步号只在
组内单调——不跨组比较步号。组 6 n=1 个案口径。TestTime 分钟制
（v1.12 坑）沿用；BoL 参考取 BoL RPT 最大者（48 的坑十三纪律）。

产物：results/cache/oxford_fade_eol.csv、results/metrics/50_fade_eol.json、
results/figures/50a_fade_eol.png
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
OUT_M = ROOT / "results" / "metrics"
OUT_F = ROOT / "results" / "figures"
CACHE = ROOT / "results" / "cache" / "oxford_fade_eol.csv"

GROUP_META = {
    1: {"kind": "combined", "c_rate": 0.5, "cycle_days": 1, "cal_days": 5,
        "cal_soc": 90, "charge": "CC"},
    2: {"kind": "combined", "c_rate": 0.25, "cycle_days": 1, "cal_days": 5,
        "cal_soc": 90, "charge": "CC"},
    3: {"kind": "combined", "c_rate": 0.5, "cycle_days": 2, "cal_days": 10,
        "cal_soc": 90, "charge": "CC"},
    4: {"kind": "combined", "c_rate": 0.25, "cycle_days": 2, "cal_days": 10,
        "cal_soc": 90, "charge": "CC"},
    5: {"kind": "cycle_only", "c_rate": 0.5, "cal_soc": None,
        "charge": "CC"},
    6: {"kind": "calendar_only", "c_rate": None, "cal_soc": 90,
        "charge": None},
    7: {"kind": "combined_cccv", "c_rate": 0.5, "cycle_days": 1,
        "cal_days": 5, "cal_soc": 90, "charge": "CCCV"},
    8: {"kind": "combined_cccv", "c_rate": 0.5, "cycle_days": 2,
        "cal_days": 10, "cal_soc": 90, "charge": "CCCV"},
    9: {"kind": "combined_cccv", "c_rate": 0.5, "cycle_days": 1,
        "cal_days": 5, "cal_soc": 100, "charge": "CCCV"},
    10: {"kind": "combined_cccv", "c_rate": 0.5, "cycle_days": 2,
         "cal_days": 10, "cal_soc": 100, "charge": "CCCV"},
}

GUIDE_FILES = [
    (OXFORD / "Guide_to_Datafiles.xlsx", 1, 4),       # Part 1: 组 1–4
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


def load_guide_rows() -> pd.DataFrame:
    """全部 Guide xlsx → 统一表（tpg, step, cell, group, label, part）。"""
    rows = []
    for path, g_lo, g_hi in GUIDE_FILES:
        if not path.exists():
            print(f"    ! guide 缺 {path.name}")
            continue
        df = pd.read_excel(path, sheet_name=0, header=None)
        group = None
        cell = None
        for _, r in df.iterrows():
            v0 = str(r[0]) if pd.notna(r[0]) else ""
            v1 = str(r[1]) if pd.notna(r[1]) else ""
            v2 = str(r[2]) if pd.notna(r[2]) else ""
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
            if not (g_lo <= group <= g_hi):
                continue
            step = int(m.group(2)) if m.group(2) else 1
            rows.append({"tpg": int(m.group(1)), "step": step,
                         "cell": cell, "group": group,
                         "label": v2.strip(),
                         "guide": path.name})
    return pd.DataFrame(rows)


def is_rpt(label: str) -> bool:
    return "RPT" in label.upper()


def collect_fade(guide: pd.DataFrame) -> pd.DataFrame:
    """逐块解析 RPT 容量（多 zip 来源同组拼接）。"""
    cache = ROOT / "results" / "cache" / "_fade_blocks.csv"
    if cache.exists():
        out = pd.read_csv(cache)
        print(f"[3] 块缓存命中 {len(out)} 行")
        return out

    frows = []
    for g in sorted(guide.group.unique()):
        sub_g = guide[(guide.group == g) & guide.label.map(is_rpt)]
        if sub_g.empty:
            continue
        zips = [p for p in ZIP_FILES.get(g, []) if p.exists()]
        if not zips:
            print(f"    ! 组 {g} 无 zip，跳过")
            continue
        # 成员索引按 (cell, step)——同一步号每芯各有一个成员
        member_idx: dict[tuple[int, int], tuple[str, zipfile.ZipFile]] = {}
        handles = []
        for p in zips:
            z = zipfile.ZipFile(p)
            handles.append(z)
            for name in z.namelist():
                meta = parse_block_member(name)
                if meta and meta["tpg"] == g:
                    key = (meta["cell"], meta["step"])
                    # 同键两 zip 都有时后写覆盖（Part 2 在列表后 → 优先）
                    member_idx[key] = (name, z)
        try:
            for _, gr in sub_g.sort_values("step").iterrows():
                hit = member_idx.get((int(gr.cell), int(gr.step)))
                if hit is None:
                    print(f"    ! 缺块 G{g} C{gr.cell} step {gr.step}")
                    continue
                name, z = hit
                try:
                    df = load_maccor_block(z.read(name))
                except Exception as exc:  # noqa: BLE001
                    print(f"    ! 解析失败 G{g} C{gr.cell} step "
                          f"{gr.step}: {exc}")
                    continue
                frows.append({
                    "group": g, "cell": int(gr.cell), "step": int(gr.step),
                    "label": gr.label, "member": name,
                    "q_cc": block_cc_discharge_capacity(df),
                    "q_dis": block_discharge_capacity(df),
                    "guide": gr.guide,
                })
        finally:
            for z in handles:
                z.close()
    out = pd.DataFrame(frows)
    if len(out):
        out.to_csv(cache, index=False)
    return out


def fade_segments(fade_df: pd.DataFrame) -> pd.DataFrame:
    """逐芯分段 fade 率：BoL→MoL（Part 1 RPT 尾）与 MoL→EoL（Part 2）。

    分段靠步号：Part 1 guide 的 RPT 块（步号较小）→ Part 2 的 RPT13+；
    MoL 分界 = Part 1 最后一个 RPT 块（v1.12 的 BoL→MoL 窗口终点）。
    """
    rows = []
    for (g, cell), sub in fade_df.groupby(["group", "cell"]):
        sub = sub.sort_values("step")
        if len(sub) < 3:
            continue
        bol = sub[sub.label.str.contains("BoL", na=False)]
        q_bol = float(bol.q_cc.max()) if len(bol) else float(sub.q_cc.iloc[0])
        if q_bol <= 0:
            continue
        # 分段：guide 来源划界（Part 1 guide → BoL..MoL；Part 2/3 → MoL..EoL）
        p1 = sub[sub.guide == "Guide_to_Datafiles.xlsx"]
        p2 = sub[sub.guide != "Guide_to_Datafiles.xlsx"]
        q_mol = float(p1.q_cc.iloc[-1]) if len(p1) else float("nan")
        q_eol = float(sub.q_cc.iloc[-1])
        fade_mol = (1 - q_mol / q_bol) * 100 if np.isfinite(q_mol) else np.nan
        fade_eol = (1 - q_eol / q_bol) * 100
        rows.append({
            "group": g, "cell": int(cell),
            "kind": GROUP_META.get(g, {}).get("kind", "?"),
            "q_bol": q_bol, "q_mol": q_mol, "q_eol": q_eol,
            "fade_mol_pct": fade_mol, "fade_eol_pct": fade_eol,
            "n_rpt": int(len(sub)), "n_rpt_p2": int(len(p2)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_M.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "cache").mkdir(parents=True, exist_ok=True)

    # ---- 1. Guide 拼接 ---------------------------------------------
    guide = load_guide_rows()
    n_rpt = int(guide.label.map(is_rpt).sum())
    print(f"[1] guide 映射 {len(guide)} 块（RPT {n_rpt}）")

    # ---- 2/3. RPT 块解析 → fade 轨迹 --------------------------------
    fade_blocks = collect_fade(guide)
    if fade_blocks.empty:
        print("[!] 无任何 RPT 块可解析（zip 未下载？）——写空指标退出")
        (OUT_M / "50_fade_eol.json").write_text(json.dumps({
            "status": "no_data", "verdict": "Group zip 未就绪",
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    fdf = fade_segments(fade_blocks)
    fdf.to_csv(CACHE, index=False)
    print(f"[3] fade 分段 {len(fdf)} 芯 → {CACHE.name}")

    # ---- 4. 组 5/6 解耦锚 + 组 7–10 修订 ---------------------------
    def _grp(g) -> dict:
        sub = fdf[fdf.group == g]
        if sub.empty:
            return {"n": 0}
        return {
            "n": int(len(sub)),
            "fade_eol_med": float(np.nanmedian(sub.fade_eol_pct)),
            "fade_mol_med": float(np.nanmedian(sub.fade_mol_pct))
            if sub.fade_mol_pct.notna().any() else None,
            "rpt_med": float(np.nanmedian(sub.n_rpt)),
            "fade_eol_min": float(np.nanmin(sub.fade_eol_pct)),
            "fade_eol_max": float(np.nanmax(sub.fade_eol_pct)),
            "fade_eol_iqr": float(sub.fade_eol_pct.quantile(0.75)
                                  - sub.fade_eol_pct.quantile(0.25))
            if len(sub) > 1 else None,
        }

    anchors = {f"g{g}": _grp(g) for g in (5, 6)}
    anchors["g1_4"] = {
        "n": int(fdf[fdf.group.isin([1, 2, 3, 4])].shape[0]),
        "fade_eol_med": float(np.nanmedian(
            fdf[fdf.group.isin([1, 2, 3, 4])].fade_eol_pct)),
        "rpt_med": float(np.nanmedian(
            fdf[fdf.group.isin([1, 2, 3, 4])].n_rpt)),
    }
    anchors["g7_10"] = {
        "n": int(fdf[fdf.group.isin([7, 8, 9, 10])].shape[0]),
        "fade_eol_med": float(np.nanmedian(
            fdf[fdf.group.isin([7, 8, 9, 10])].fade_eol_pct))
        if fdf[fdf.group.isin([7, 8, 9, 10])].shape[0] else None,
        "rpt_med": float(np.nanmedian(
            fdf[fdf.group.isin([7, 8, 9, 10])].n_rpt))
        if fdf[fdf.group.isin([7, 8, 9, 10])].shape[0] else None,
    }
    print(f"[4] anchors: {anchors}")

    # 分段速率对比（组合工况组内 BoL→MoL vs MoL→EoL）
    comb = fdf[fdf.group.isin([1, 2, 3, 4]) & fdf.fade_mol_pct.notna()]
    seg_cmp = {}
    if len(comb):
        seg_cmp = {
            "n": int(len(comb)),
            "fade_mol_med": float(np.nanmedian(comb.fade_mol_pct)),
            "fade_eol_med": float(np.nanmedian(comb.fade_eol_pct)),
            "later_share_med": float(np.nanmedian(
                (comb.fade_eol_pct - comb.fade_mol_pct)
                / comb.fade_eol_pct.replace(0, np.nan))),
        }

    # ---- 5. ΔR_pol/fade 到 EoL --------------------------------------
    dr_fade = {}
    m49 = OUT_M / "49_eis_eol.json"
    if m49.exists():
        m = json.loads(m49.read_text(encoding="utf-8"))
        pairs = pd.DataFrame(m.get("pairs_mol_eol", []))
        if len(pairs) and len(fdf):
            merged = pairs.merge(
                fdf[["group", "cell", "fade_eol_pct", "q_bol"]],
                on=["group", "cell"], how="left")
            merged = merged[merged.fade_eol_pct > 1.0]
            if len(merged):
                dr_fade = {
                    "n": int(len(merged)),
                    "drpol_per_fade_med": float(np.nanmedian(
                        merged.d_r_pol / merged.fade_eol_pct * 1e3)),
                    "drpol_per_fade_iqr": float(
                        (merged.d_r_pol / merged.fade_eol_pct * 1e3)
                        .quantile(0.75)
                        - (merged.d_r_pol / merged.fade_eol_pct * 1e3)
                        .quantile(0.25)),
                }
    print(f"[5] ΔR/fade to EoL: {dr_fade}")

    # ---- 图 --------------------------------------------------------
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    for g, color in [(1, "#2980b9"), (2, "#8e44ad"), (3, "#27ae60"),
                     (4, "#f39c12")]:
        sub = fdf[fdf.group == g]
        if sub.empty:
            continue
        ax.scatter(sub.q_bol, sub.fade_eol_pct, s=40, color=color,
                   label=f"G{g} ({GROUP_META[g]['c_rate']}C)")
    ax.set_xlabel("q_bol [Ah]")
    ax.set_ylabel("fade to EoL [%]")
    ax.set_title("combined-profile groups (1-4)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    tags = ["G5\ncycle", "G6\ncalendar", "G1-4\ncombined", "G7-10\ncccv"]
    keys = ["g5", "g6", "g1_4", "g7_10"]
    vals = [(anchors.get(k) or {}).get("fade_eol_med") or 0 for k in keys]
    ns = [(anchors.get(k) or {}).get("n", 0) for k in keys]
    colors = ["#2980b9", "#8e44ad", "#7f8c8d", "#c0392b"]
    ax.bar(np.arange(len(tags)), vals, 0.55, color=colors)
    ax.set_xticks(np.arange(len(tags)),
                  [f"{t}\n(n={n})" for t, n in zip(tags, ns)], fontsize=8)
    ax.set_ylabel("median fade to EoL [%]")
    ax.set_title("condition anchors")

    ax = axes[1, 0]
    if len(comb):
        x = np.arange(len(comb))
        ax.bar(x, comb.fade_mol_pct, 0.4, label="BoL→MoL", color="#888")
        ax.bar(x + 0.4, comb.fade_eol_pct - comb.fade_mol_pct, 0.4,
               label="MoL→EoL", color="#c0392b")
        ax.set_xticks(x + 0.2, [f"G{g}C{c}" for g, c in
                                zip(comb.group, comb.cell)],
                      rotation=90, fontsize=7)
        ax.set_ylabel("fade [%]")
        ax.set_title("segment-wise fade (combined groups)")
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    if dr_fade:
        merged_v = pairs.merge(
            fdf[["group", "cell", "fade_eol_pct"]],
            on=["group", "cell"], how="left")
        merged_v = merged_v[merged_v.fade_eol_pct > 1.0]
        ax.scatter(merged_v.fade_eol_pct,
                   merged_v.d_r_pol * 1e3, s=36, color="#c0392b",
                   alpha=0.7)
        ax.set_xlabel("fade to EoL [%]")
        ax.set_ylabel("ΔR_pol (BoL→EoL) [mΩ]")
        ax.set_title(f"impedance cost per fade "
                     f"(med {dr_fade['drpol_per_fade_med']:.1f} mΩ/%)")
    else:
        ax.text(0.5, 0.5, "ΔR/fade pending\n(49 pairs × fade)",
                ha="center", va="center", fontsize=10)
        ax.set_axis_off()

    fig.suptitle("50a · fade to EoL & condition anchors (Oxford NCA)",
                 fontsize=11)
    fig.tight_layout()
    save_fig(fig, "50a_fade_eol", OUT_F)

    # ---- 判词与指标 -------------------------------------------------
    g5 = anchors.get("g5") or {}
    g6 = anchors.get("g6") or {}
    later = seg_cmp.get("later_share_med")
    if seg_cmp:
        later_val = later if later is not None else float("nan")
        # 与 49 阻抗链（MoL→EoL 段占全链弧增长 63%）动态比较——
        # 方向一致才写「同向」，反向如实写「时间解耦」（判词从计算
        # 生成，不硬编码；v1.8 教训复查）
        imp_later = 0.63
        if later_val > 0.55 and imp_later > 0.55:
            cmp_txt = f"与 49 阻抗链后段 {imp_later:.0%} 同向（均加速）"
        elif later_val <= 0.55 and imp_later <= 0.55:
            cmp_txt = f"与 49 阻抗链后段 {imp_later:.0%} 同向（均放缓）"
        else:
            cmp_txt = (f"与 49 阻抗链后段 {imp_later:.0%} **反向**"
                       f"——fade 后段放缓而界面阻抗增长加速，"
                       f"容量与阻抗在时间维度解耦（与 §72 fade 差"
                       f"不在阻抗总量的结论一致的另一表现）")
        seg_sentence = (
            f"组合工况（组 1–4）分段 fade：BoL→MoL 中位 "
            f"{seg_cmp['fade_mol_med']:.1f}% → 全链 EoL "
            f"{seg_cmp['fade_eol_med']:.1f}%，后段份额中位 {later_val:.0%}"
            f"（{'退化加速' if later_val > 0.55 else '退化放缓/持平'}，"
            f"{cmp_txt}）"
        )
    else:
        seg_sentence = "组合工况分段数据不足"
    verdict = (
        f"fade 到 EoL + 新工况解耦收口：{len(fdf)} 芯 RPT 轨迹"
        f"（{len(fade_blocks)} 块，Part 1+2/3 Guide 拼接）——"
        f"{seg_sentence}。"
        + (
            f"循环/日历解耦锚：组 5 纯 C/2 连续循环 fade "
            f"{g5.get('fade_eol_med') or float('nan'):.1f}%（n={g5.get('n', 0)}，"
            f"RPT 中位 {g5.get('rpt_med') or float('nan'):.0f} 点）"
            f"vs 组 6 纯日历 {g6.get('fade_eol_med') or float('nan'):.1f}%"
            f"（n={g6.get('n', 0)}，RPT 中位 {g6.get('rpt_med') or float('nan'):.0f} 点）"
            + (
                "——循环 fade 快于日历" if (g5.get("fade_eol_med") or 0)
                > (g6.get("fade_eol_med") or 0) else
                "——循环 fade 不高于日历（与常规预期反向，如实记录）"
            )
            + "（同 24 °C）。⚠️ EoL 语义不同：组 5 连续循环走到"
            f"真寿命终点（单芯 fade {g5.get('fade_eol_min') or float('nan'):.0f}–"
            f"{g5.get('fade_eol_max') or float('nan'):.0f}%），组 6 日历是试验终点"
            "——水平差是「同窗口内谁死谁活」不是同里程 fade 速率。"
            if g5.get("n") and g6.get("n") else
            (
                f"循环/日历解耦锚：组 6 纯日历 fade "
                f"{g6.get('fade_eol_med') or float('nan'):.1f}%"
                f"（n={g6.get('n', 0)}，RPT 中位 {g6.get('rpt_med') or float('nan'):.0f} 点）"
                f"已入库；组 5 连续循环包未就绪，fade 侧解耦对比待补。"
                if g6.get("n") else
                (
                    f"循环/日历解耦锚：组 5 纯循环 fade "
                    f"{g5.get('fade_eol_med') or float('nan'):.1f}%"
                    f"（n={g5.get('n', 0)}）已入库；组 6 日历包未就绪。"
                    if g5.get("n") else
                    f"循环/日历 fade 锚待补（组 5/6 Group 包未就绪；"
                    f"EIS 侧三通道解耦见 49）。"
                )
            )
        )
        + (
            f"CCCV/满充修订：组 7–10 fade "
            f"{(anchors.get('g7_10') or {}).get('fade_eol_med') or float('nan'):.1f}% "
            f"（RPT 中位 {(anchors.get('g7_10') or {}).get('rpt_med') or float('nan'):.0f} 点）"
            f"vs CC/90% 组 1–4 {(anchors.get('g1_4') or {}).get('fade_eol_med') or float('nan'):.1f}%"
            f"（RPT 中位 {(anchors.get('g1_4') or {}).get('rpt_med') or float('nan'):.0f} 点）"
            f"——⚠️ EoL=数据集末标签非 20% SOH，RPT 跨度差异大时"
            f"水平不可直接比，只作方向参考。"
            if (anchors.get("g7_10") or {}).get("n") else
            f"CCCV/满充 fade 修订待补（组 7–10 Group 包未就绪）。"
        )
        + f"ΔR_pol/fade 到 EoL：{(dr_fade.get('drpol_per_fade_med') or float('nan')):.1f} mΩ/%"
        f"（n={dr_fade.get('n', 0)}）。"
        + (
            f"配比图景收口：49 EIS 侧工况解耦已闭合（组 5 尾段 88% "
            f"vs 组 6 弧段 94%）；fade 侧"
            + (
                "解耦同向（循环 fade 高于日历 + 尾段主导 = 类 shape/LAM，"
                "日历 fade 低 + 弧段主导 = 类 shift/SEI）——"
                "v1.12「慢→shift、快→shape」在同体系内成立，三线合流"
                "（SNL shape 份额 + Oxford 阻抗解耦 + fade 解耦）。"
                if (g5.get("fade_eol_med") or 0) > (g6.get("fade_eol_med") or 0)
                else "与 EIS 解耦**不**同向（循环 fade 未高于日历）"
                "——配比图景只在阻抗通道成立，fade 通道不支持，不写三线合流。"
            )
            if g5.get("n") and g6.get("n") else
            f"配比图景收口：EIS 侧工况解耦已闭合（49：组 5 尾段 88% "
            f"vs 组 6 弧段 94%），fade 侧解耦待组 5/6 包到齐后补跑。"
        )
        + f"⚠️ EoL=Guide 末 RPT 标签口径；组 6 n=1；跨 Part 步号仅组内单调。"
    )
    print("[6]", verdict[:130], "...")

    metrics = {
        "n_cells": int(len(fdf)),
        "n_blocks": int(len(fade_blocks)),
        "segment_compare": seg_cmp,
        "condition_anchors": anchors,
        "drpol_per_fade_eol": dr_fade,
        "fade_table": fdf.to_dict(orient="records"),
        "caveats": [
            "EoL=Guide 末 RPT 标签；跨 Part 步号仅组内单调",
            "组 6 纯日历 n=1 个案口径",
            "fade 参考=BoL RPT 最大者（48 坑十三纪律）",
            "TestTime 分钟制积分（v1.12 坑十五）",
        ],
        "verdict": verdict,
    }
    (OUT_M / "50_fade_eol.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[7] → {OUT_M / '50_fade_eol.json'}")


if __name__ == "__main__":
    main()
