# -*- coding: utf-8 -*-
"""WP48 EIS BoL→MoL 阻抗退化模式（v1.12 第二线）。

Oxford PathDependent EIS.zip（117 谱，NCA/石墨 18650，Raj & Howey 2020）
BoL→MoL 配对差分的**频率无关几何分解**：

1. 全量特征提取（117 谱 Nyquist 几何）；
2. BoL↔MoL 配对（11 芯 × 3 SOC）→ ΔR_ohm / ΔR_pol / 弧段 / 尾段
   分区归属——阻抗增长住在欧姆段、弧段还是扩散尾；
3. SOC 依赖（20/50/80）与组条件（TPG1–4：C/2 vs C/4 倍率、
   1+5d vs 2+10d 周期，循环:日历 = 1:5）；
4. 与模式框架对拍：SEI/LLI（shift-like）vs LAM（shape-like）的
   阻抗签名归属 + 论文 DVA「高倍率负极退化」结论对照。

⚠️ 频率轴不可溯源（文件无频率、指南/论文实验条件不可得）——全部
口径为索引空间几何，谱间可比、绝对 Hz 不可。

产物：results/cache/eis_features.csv、results/metrics/47_eis_degrade.json、
results/figures/47a_eis_degrade.png
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

from bmthermal.eis_features import (  # noqa: E402
    load_eis_spectrum, nyquist_features, pair_deltas, parse_eis_member,
)
from bmthermal.plotting import save as save_fig, setup_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

EIS_ZIP = Path("D:/项目/02_数据集/01_国际公开集/Oxford_PathDependent/EIS.zip")
OUT_M = ROOT / "results" / "metrics"
OUT_F = ROOT / "results" / "figures"
CACHE = ROOT / "results" / "cache" / "eis_features.csv"

# 组条件（ORA 数据集说明：18650 NCA/石墨，循环:日历=1:5，90% SOC 日历）
GROUP_META = {
    1: {"cycle_days": 1, "cal_days": 5, "c_rate": 0.5},
    2: {"cycle_days": 1, "cal_days": 5, "c_rate": 0.25},
    3: {"cycle_days": 2, "cal_days": 10, "c_rate": 0.5},
    4: {"cycle_days": 2, "cal_days": 10, "c_rate": 0.25},
}


def main() -> None:
    OUT_M.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "cache").mkdir(parents=True, exist_ok=True)

    # ---- 1. 全量特征提取 --------------------------------------------
    rows = []
    spectra = {}
    with zipfile.ZipFile(EIS_ZIP) as z:
        members = sorted(n for n in z.namelist() if n.endswith(".mat"))
        for name in members:
            meta = parse_eis_member(name)
            if meta is None:
                continue
            z_re, z_im = load_eis_spectrum(z.read(name))
            f = nyquist_features(z_re, z_im)
            row = {**meta, **f.to_dict()}
            rows.append(row)
            spectra[(meta["group"], meta["cell"], meta["soc"], meta["stage"])] = \
                (z_re, z_im)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    print(f"[1] {len(df)} 谱特征 → {CACHE.relative_to(ROOT)}")
    qcount = df.quality.value_counts().to_dict()
    print(f"    质量旗标: {qcount}")

    # ---- 2. BoL↔MoL 配对差分 ---------------------------------------
    pairs = []
    keys = set((r.group, r.cell, r.soc) for r in df.itertuples()
               if r.stage == "BoL")
    for g, c, soc in sorted(keys):
        sub_b = df[(df.group == g) & (df.cell == c) & (df.soc == soc)
                   & (df.stage == "BoL")]
        sub_m = df[(df.group == g) & (df.cell == c) & (df.soc == soc)
                   & (df.stage == "MoL")]
        if len(sub_b) != 1 or len(sub_m) != 1:
            continue
        fb = nyquist_features(*spectra[(g, c, soc, "BoL")])
        fm = nyquist_features(*spectra[(g, c, soc, "MoL")])
        p = pair_deltas(fb, fm, group=g, cell=c, soc=soc)
        d = {"group": g, "cell": c, "soc": soc, **p.deltas}
        d["share_arc"] = p.fade_share_rpol
        d["q_bol"] = fb.quality
        d["q_mol"] = fm.quality
        pairs.append(d)
    pdf = pd.DataFrame(pairs)
    print(f"[2] {len(pdf)} 对（{pdf.groupby(['group','cell']).ngroups} 芯）")

    # ---- 3. SOC / 组条件聚合 ---------------------------------------
    def _med(x):
        return float(np.nanmedian(x))

    soc_stats = []
    for soc, g in pdf.groupby("soc"):
        soc_stats.append({
            "soc": int(soc), "n_pairs": int(len(g)),
            "d_r_ohm_med": _med(g.d_r_ohm),
            "d_r_pol_med": _med(g.d_r_pol),
            "d_arc_dre_med": _med(g.d_arc_dre),
            "d_tail_slope_med": _med(g.d_tail_slope),
            "share_arc_med": _med(g.share_arc.dropna()),
        })
        print(f"    SOC{soc}: ΔR_ohm {soc_stats[-1]['d_r_ohm_med']*1e3:+.2f} mΩ "
              f"ΔR_pol {soc_stats[-1]['d_r_pol_med']*1e3:+.2f} mΩ "
              f"弧段占比 {soc_stats[-1]['share_arc_med']:.2f}")

    group_stats = []
    for g_, gg in pdf.groupby("group"):
        meta = GROUP_META.get(int(g_), {})
        group_stats.append({
            "group": int(g_), "n_pairs": int(len(gg)),
            "c_rate": meta.get("c_rate"),
            "cycle_days": meta.get("cycle_days"),
            "d_r_ohm_med": _med(gg.d_r_ohm),
            "d_r_pol_med": _med(gg.d_r_pol),
            "d_arc_dre_med": _med(gg.d_arc_dre),
            "share_arc_med": _med(gg.share_arc.dropna()),
        })

    # C/2 vs C/4 对比
    half = pdf[pdf.group.isin([1, 3])]
    quart = pdf[pdf.group.isin([2, 4])]
    rate_cmp = {
        "c2_d_r_ohm_med": _med(half.d_r_ohm),
        "c4_d_r_ohm_med": _med(quart.d_r_ohm),
        "c2_d_r_pol_med": _med(half.d_r_pol),
        "c4_d_r_pol_med": _med(quart.d_r_pol),
        "c2_share_arc_med": _med(half.share_arc.dropna()),
        "c4_share_arc_med": _med(quart.share_arc.dropna()),
        "n_c2": int(len(half)), "n_c4": int(len(quart)),
    }
    print(f"[3] C/2 组 ΔR_pol {rate_cmp['c2_d_r_pol_med']*1e3:.2f} mΩ vs "
          f"C/4 {rate_cmp['c4_d_r_pol_med']*1e3:.2f} mΩ")

    # ---- 4. 模式框架对拍 -------------------------------------------
    # 阻抗签名归属（物理修正，v1.12）：35 点谱上 R_SEI 与 R_ct 合弧，
    # 弧段增长 = SEI 增厚（shift/LLI 侧）∪ 活性表面丧失（LAM/shape 侧）
    # ——几何分辨率内**不可分**，两读法并列（与 v1.10/v1.11 同条纪律）。
    # 可判的是「界面 vs 欧姆」的分区归属。
    total_pos = float(pdf.d_r_ohm.clip(lower=0).sum()
                      + pdf.d_r_pol.clip(lower=0).sum() + 1e-12)
    share_ohm = float(pdf.d_r_ohm.clip(lower=0).sum() / total_pos)
    share_arc = float(pdf.d_arc_dre.clip(lower=0).sum()
                      / (float(pdf.d_r_pol.sum()) + 1e-12))
    print(f"[4] 增长分区（正部合计）：ΔR_ohm 占比 {share_ohm:.2f}、"
          f"弧段实部占 ΔR_pol {share_arc:.2f}")

    # ---- 图 47a -----------------------------------------------------
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0, 0]
    colors = {20: "#2e86de", 50: "#27ae60", 80: "#e67e22"}
    for soc, c in colors.items():
        sub = df[(df.soc == soc) & (df.stage == "BoL")]
        for r in sub.itertuples():
            z_re, z_im = spectra[(r.group, r.cell, r.soc, "BoL")]
            ax.plot(z_re * 1e3, z_im * 1e3, color=c, alpha=0.35, lw=0.8)
        sub_m = df[(df.soc == soc) & (df.stage == "MoL")]
        for r in sub_m.itertuples():
            z_re, z_im = spectra[(r.group, r.cell, r.soc, "MoL")]
            ax.plot(z_re * 1e3, z_im * 1e3, color=c, alpha=0.9, lw=1.2,
                    ls="--")
    ax.axhline(0, color="#999", lw=0.5)
    ax.set_xlabel("Z_re [mΩ]")
    ax.set_ylabel("Z_im [mΩ]")
    ax.set_title("a) Nyquist：BoL（实）→ MoL（虚）三 SOC")
    ax.text(0.98, 0.02, "频率轴不可溯源→索引空间几何", transform=ax.transAxes,
            fontsize=7, ha="right", color="#888")

    ax = axes[0, 1]
    w = 0.25
    socs = [20, 50, 80]
    x = np.arange(len(socs))
    for k, (field, label, col) in enumerate([
            ("d_r_ohm", "ΔR_ohm", "#2e86de"),
            ("d_r_pol", "ΔR_pol", "#c0392b"),
            ("d_arc_dre", "Δ弧段", "#8e44ad")]):
        vals = [next(s[f"{field}_med"] for s in soc_stats if s["soc"] == so)
                * 1e3 for so in socs]
        ax.bar(x + (k - 1) * w, vals, w, label=label, color=col)
    ax.set_xticks(x, [f"SOC{so}" for so in socs])
    ax.set_ylabel("BoL→MoL 增量 [mΩ]")
    ax.set_title("b) 增量分区 vs SOC（中位）")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for g_, gg in pdf.groupby("group"):
        meta = GROUP_META.get(int(g_), {})
        lbl = (f"TPG{g_}（C/{int(1/meta['c_rate'])}，"
               f"{meta['cycle_days']}+{meta['cal_days']}d）")
        ax.scatter(gg.d_r_ohm * 1e3, gg.d_r_pol * 1e3, s=42,
                   label=lbl, alpha=0.85)
        for r in gg.itertuples():
            ax.annotate(f"{int(r.cell)}", (r.d_r_ohm * 1e3, r.d_r_pol * 1e3),
                        fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="#999", lw=0.5)
    ax.axvline(0, color="#999", lw=0.5)
    ax.set_xlabel("ΔR_ohm [mΩ]")
    ax.set_ylabel("ΔR_pol [mΩ]")
    ax.set_title("c) 逐对增量（按组）")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.bar([0, 1], [share_ohm, share_arc], 0.5,
           color=["#2e86de", "#c0392b"])
    ax.set_xticks([0, 1], ["ΔR_ohm /\n(ΔR_ohm+ΔR_pol)", "Δ弧段 /\nΔR_pol"])
    ax.set_ylabel("正部合计占比")
    ax.set_title("d) 阻抗增长分区（全对合计）")
    for i, v in enumerate([share_ohm, share_arc]):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)

    fig.suptitle("47a EIS BoL→MoL 阻抗退化模式（Oxford NCA，频率无关几何）",
                 fontsize=12)
    fig.tight_layout()
    save_fig(fig, "47a_eis_degrade", OUT_F)

    # ---- 指标 -------------------------------------------------------
    d_ohm_med = float(np.nanmedian(pdf.d_r_ohm))
    d_pol_med = float(np.nanmedian(pdf.d_r_pol))
    rate_gap = abs(rate_cmp["c2_d_r_pol_med"] - rate_cmp["c4_d_r_pol_med"])
    verdict = (
        f"EIS 阻抗退化模式收口（开放项②阻抗侧）：Oxford NCA 18650 "
        f"{len(pdf)} 个 BoL→MoL 对（11 芯 × 3 SOC）频率无关几何分解——"
        f"ΔR_ohm 中位 {d_ohm_med*1e3:+.1f} mΩ、ΔR_pol {d_pol_med*1e3:+.1f} mΩ，"
        f"增长分区：欧姆段仅占正部合计 {share_ohm:.0%}、弧段实部占 ΔR_pol "
        f"{share_arc:.0%}——**界面主导**（欧姆/电解液通道基本不动）。"
        f"弧段归属两读法并列（35 点谱 R_SEI∥R_ct 合弧，几何不可分）："
        f"读法 a「SEI 增厚」（shift/LLI 侧，慢倍率+日历占比高=预期主项）；"
        f"读法 b「活性表面丧失」（LAM/shape 侧）不可排除。"
        f"论文 DVA「路径依赖=负极退化」与两读法均相容（负极 SEI/LLI "
        f"与负极 LAM 都归因负极）——阻抗侧不裁决两读法，"
        f"与 v1.11 post-mortem 位点开放项衔接。"
        f"组对比：C/2 组 ΔR_pol {rate_cmp['c2_d_r_pol_med']*1e3:.1f} mΩ vs "
        f"C/4 {rate_cmp['c4_d_r_pol_med']*1e3:.1f} mΩ"
        f"（{rate_cmp['n_c2']}/{rate_cmp['n_c4']} 对，差 {rate_gap*1e3:.1f} mΩ）"
        f"——倍率差{'可见' if rate_gap > 0.003 else '不显著（路径老化阻抗增长近倍率无关）'}。"
        f"⚠️ 频率轴不可溯源，仅索引空间差分有效；弧段归属、"
        f"LAM-PE/NE 位点均在分辨率外。"
    )
    print("[5]", verdict[:120], "...")

    metrics = {
        "n_spectra": int(len(df)),
        "n_pairs": int(len(pdf)),
        "quality_flags": {str(k): int(v) for k, v in qcount.items()},
        "soc_stats": soc_stats,
        "group_stats": group_stats,
        "rate_compare": rate_cmp,
        "partition": {
            "share_ohm": share_ohm,
            "share_arc_of_rpol": share_arc,
            "d_r_ohm_med": d_ohm_med,
            "d_r_pol_med": d_pol_med,
        },
        "pairs": pdf.drop(columns=["q_bol", "q_mol"]).to_dict(orient="records"),
        "caveats": [
            "频率轴不可溯源（文件无频率、指南/论文实验条件不可得）——仅索引空间几何",
            "弧段归属（SEI 增厚 vs 活性表面丧失）几何分辨率内不可分",
            "LCO 半电池几何验证（v1.11）与本 NCA 化学不同体系",
        ],
        "verdict": verdict,
    }
    (OUT_M / "47_eis_degrade.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[6] → {OUT_M / '47_eis_degrade.json'}")


if __name__ == "__main__":
    main()
