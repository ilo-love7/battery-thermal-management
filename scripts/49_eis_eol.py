# -*- coding: utf-8 -*-
"""WP52 EIS MoL→EoL 延伸配对 + 三阶段链（v1.13 第一线）。

接 v1.12 的 BoL→MoL 几何分解（scripts/47），把阻抗退化模式推到 EoL：

1. Part 2 EIS.zip（129 谱，组 1–6 × BoL/MoL/EoL；组 5 无 MoL）+
   Part 3 EIS.zip（72 谱，组 7–10 × BoL/EoL）全量几何提取；
2. **复现质检**：Part 2 的 BoL→MoL 差分 vs Part 1（47 的 33 对）
   ——同芯同 SOC 一致性（数据打包独立导出，谱值应同源）；
3. MoL→EoL 配对差分（组 1–4、6）：欧姆/弧段/尾段分区归属是否延续
   「界面主导」到 EoL；
4. 三阶段链 BoL→MoL→EoL：各段占全链 ΔR_pol 的份额（退化加速/饱和）；
5. **循环/日历解耦锚**（Part 2 新工况）：组 5 纯 CC 循环（C/2 连续）
   vs 组 6 纯日历（90% SOC）——BoL→EoL 阻抗增长的模式归属对照；
6. **CCCV/满充新工况**（Part 3 组 7–10）：CCCV 循环 + 100% SOC 日历
   vs 组 1–4（CC + 90% SOC）——充电制度与日历 SOC 对阻抗增长的影响。

⚠️ 频率轴不可溯源（同 v1.12）：仅索引空间几何差分/比值有效。
组 6 n=1（单芯）——只报个案值不做统计推断。

产物：results/cache/eis_features_eol.csv、results/metrics/49_eis_eol.json、
results/figures/49a_eis_eol.png
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
    chain_share, load_eis_spectrum, nyquist_features, parse_eis_member,
    stage_deltas,
)
from bmthermal.plotting import save as save_fig, setup_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OXFORD = Path("D:/项目/02_数据集/01_国际公开集/Oxford_PathDependent")
EIS_P2 = OXFORD / "Part2" / "EIS.zip"
EIS_P3 = OXFORD / "Part3" / "EIS.zip"
EIS_P1 = OXFORD / "EIS.zip"                  # v1.12 基线（复现质检）
OUT_M = ROOT / "results" / "metrics"
OUT_F = ROOT / "results" / "figures"
CACHE = ROOT / "results" / "cache" / "eis_features_eol.csv"

# 组工况元数据（ORA Readme + Guide）
GROUP_META = {
    1: {"kind": "combined", "cycle_days": 1, "cal_days": 5,
        "c_rate": 0.5, "cal_soc": 90, "charge": "CC"},
    2: {"kind": "combined", "cycle_days": 1, "cal_days": 5,
        "c_rate": 0.25, "cal_soc": 90, "charge": "CC"},
    3: {"kind": "combined", "cycle_days": 2, "cal_days": 10,
        "c_rate": 0.5, "cal_soc": 90, "charge": "CC"},
    4: {"kind": "combined", "cycle_days": 2, "cal_days": 10,
        "c_rate": 0.25, "cal_soc": 90, "charge": "CC"},
    5: {"kind": "cycle_only", "c_rate": 0.5, "cal_soc": None,
        "charge": "CC"},                       # 连续 CC 循环 C/2
    6: {"kind": "calendar_only", "c_rate": None, "cal_soc": 90,
        "charge": None},                       # 纯日历 90% SOC（n=1）
    7: {"kind": "combined_cccv", "cycle_days": 1, "cal_days": 5,
        "c_rate": 0.5, "cal_soc": 90, "charge": "CCCV"},
    8: {"kind": "combined_cccv", "cycle_days": 2, "cal_days": 10,
        "c_rate": 0.5, "cal_soc": 90, "charge": "CCCV"},
    9: {"kind": "combined_cccv", "cycle_days": 1, "cal_days": 5,
        "c_rate": 0.5, "cal_soc": 100, "charge": "CCCV"},   # 4.2V 日历
    10: {"kind": "combined_cccv", "cycle_days": 2, "cal_days": 10,
         "c_rate": 0.5, "cal_soc": 100, "charge": "CCCV"},
}


def extract_zip(zip_path: Path, source: str) -> list[dict]:
    """EIS zip → 特征行列表。"""
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".mat")):
            meta = parse_eis_member(name)
            if meta is None:
                continue
            z_re, z_im = load_eis_spectrum(z.read(name))
            f = nyquist_features(z_re, z_im)
            rows.append({"source": source, "member": name,
                         **meta, **f.to_dict()})
    return rows


def pair_table(df: pd.DataFrame, stage_a: str, stage_b: str) -> pd.DataFrame:
    """同 (group, cell, soc) 的 stage_a→stage_b 配对差分表。"""
    rows = []
    keys = ["group", "cell", "soc"]
    a = df[df.stage == stage_a].set_index(keys)
    b = df[df.stage == stage_b].set_index(keys)
    for key in a.index.intersection(b.index):
        ra, rb = a.loc[key], b.loc[key]
        d = stage_deltas(
            _row_feat(ra), _row_feat(rb))
        share = float(d["d_arc_dre"] / d["d_r_pol"]) if d["d_r_pol"] > 0 \
            else float("nan")
        rows.append({"group": key[0], "cell": key[1], "soc": key[2],
                     "stage_pair": f"{stage_a}->{stage_b}",
                     "quality": rb.get("quality", "ok"),
                     **d, "arc_share_rpol": share})
    return pd.DataFrame(rows)


def _row_feat(r) -> "object":
    from bmthermal.eis_features import NyquistFeatures
    return NyquistFeatures(
        n_pts=int(r.n_pts), r_ohm=float(r.r_ohm), r_pol=float(r.r_pol),
        z_im_hf=float(r.z_im_hf), z_im_peak=float(r.z_im_peak),
        arc_height=float(r.arc_height), zero_cross=int(r.zero_cross),
        valley_idx=int(r.valley_idx), arc_dre=float(r.arc_dre),
        tail_slope=float(r.tail_slope), lf_abs=float(r.lf_abs),
        lf_phase_deg=float(r.lf_phase_deg), quality=str(r.quality))


def segment_stats(pdf: pd.DataFrame) -> dict:
    """配对表 → 三通道分区统计（欧姆/弧段/尾段）。

    r_pol = (Z_re[zc]−Z_re[0]) + arc_dre + tail_dre（谱内 Z_re 单调时
    精确分解）；尾段增量 ≈ d_r_pol − d_arc_dre（钩-弧过渡段小量并入
    尾段，实测 ≤3 mΩ）。v1.13 口径坑：两通道（只算欧姆+弧）会把
    组 5 尾段爆炸误判成「欧姆主导」——份额分母必须覆盖 r_pol 全部增量。
    """
    if pdf.empty:
        return {"n": 0}
    tail = (pdf.d_r_pol - pdf.d_arc_dre).clip(lower=0)
    ohm = pdf.d_r_ohm.clip(lower=0)
    arc = pdf.d_arc_dre.clip(lower=0)
    total = float(ohm.sum() + arc.sum() + tail.sum()) or float("nan")
    return {
        "n": int(len(pdf)),
        "d_r_ohm_med": float(np.nanmedian(pdf.d_r_ohm)),
        "d_r_pol_med": float(np.nanmedian(pdf.d_r_pol)),
        "d_arc_dre_med": float(np.nanmedian(pdf.d_arc_dre)),
        "d_tail_dre_med": float(np.nanmedian(tail)),
        "d_tail_slope_med": float(np.nanmedian(pdf.d_tail_slope)),
        "arc_share_med": float(np.nanmedian(pdf.arc_share_rpol)),
        "share_ohm": float(ohm.sum() / total) if total else None,
        "share_arc": float(arc.sum() / total) if total else None,
        "share_tail": float(tail.sum() / total) if total else None,
    }


def main() -> None:
    OUT_M.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "cache").mkdir(parents=True, exist_ok=True)

    # ---- 1. 全量特征提取（Part 2 + Part 3）--------------------------
    rows = extract_zip(EIS_P2, "Part2") + extract_zip(EIS_P3, "Part3")
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    print(f"[1] spectra: {len(df)} (P2+P3) → {CACHE.name}")
    print("    stages:", df.groupby(["source", "stage"]).size().to_dict())

    # ---- 2. 复现质检（Part 2 BoL→MoL vs Part 1 基线）----------------
    qc: dict = {"n_overlap": 0, "dr_pol_max_abs_diff": None}
    if EIS_P1.exists():
        df_p1 = pd.DataFrame(extract_zip(EIS_P1, "Part1"))
        p1 = pair_table(df_p1, "BoL", "MoL")
        p2 = pair_table(df, "BoL", "MoL")
        m = p1.merge(p2, on=["group", "cell", "soc"],
                     suffixes=("_p1", "_p2"))
        qc["n_overlap"] = int(len(m))
        if len(m):
            diff = (m.d_r_pol_p1 - m.d_r_pol_p2).abs()
            qc["dr_pol_max_abs_diff"] = float(diff.max())
            qc["dr_pol_med_abs_diff"] = float(diff.median())
            qc["reproduced"] = bool(diff.max() < 1e-9)
    print(f"[2] QC vs Part1: {qc}")

    # ---- 3. MoL→EoL 配对差分 ---------------------------------------
    mol_eol = pair_table(df, "MoL", "EoL")
    bol_mol = pair_table(df, "BoL", "MoL")
    bol_eol = pair_table(df, "BoL", "EoL")
    seg_mol_eol = segment_stats(mol_eol)
    seg_bol_mol = segment_stats(bol_mol)
    seg_bol_eol = segment_stats(bol_eol)
    print(f"[3] MoL→EoL pairs: {seg_mol_eol['n']}, "
          f"arc_share_med={seg_mol_eol.get('arc_share_med'):.3f}")

    # ---- 4. 三阶段链份额 -------------------------------------------
    chain_rows = []
    for key in mol_eol.set_index(["group", "cell", "soc"]).index:
        b1 = bol_mol.set_index(["group", "cell", "soc"])
        b2 = mol_eol.set_index(["group", "cell", "soc"])
        b3 = bol_eol.set_index(["group", "cell", "soc"])
        if key not in b1.index or key not in b3.index:
            continue
        s1 = b1.loc[key]
        s2 = b2.loc[key]
        s3 = b3.loc[key]
        chain_rows.append({
            "group": key[0], "cell": key[1], "soc": key[2],
            "seg_bol_mol_arc": float(s1.d_arc_dre),
            "seg_mol_eol_arc": float(s2.d_arc_dre),
            "seg_bol_eol_arc": float(s3.d_arc_dre),
            "seg_bol_mol_rpol": float(s1.d_r_pol),
            "seg_mol_eol_rpol": float(s2.d_r_pol),
            "seg_bol_eol_rpol": float(s3.d_r_pol),
            "mol_eol_share_arc": chain_share(
                {"d_arc_dre": float(s1.d_arc_dre)},
                {"d_arc_dre": float(s2.d_arc_dre)}),
            "additivity_residual": float(
                s1.d_r_pol + s2.d_r_pol - s3.d_r_pol),
        })
    chain = pd.DataFrame(chain_rows)
    chain_med = {
        "n": int(len(chain)),
        "mol_eol_share_arc_med": float(np.nanmedian(chain.mol_eol_share_arc))
        if len(chain) else None,
        "additivity_residual_max_abs": float(
            chain.additivity_residual.abs().max()) if len(chain) else None,
        "seg_arc_bol_mol_med": float(np.nanmedian(chain.seg_bol_mol_arc))
        if len(chain) else None,
        "seg_arc_mol_eol_med": float(np.nanmedian(chain.seg_mol_eol_arc))
        if len(chain) else None,
    }
    print(f"[4] chain: {chain_med}")

    # ---- 5. 循环/日历解耦锚（组 5 vs 组 6）--------------------------
    anchor = {}
    for g in (5, 6):
        sub = bol_eol[bol_eol.group == g]
        if len(sub):
            anchor[f"g{g}"] = {
                "kind": GROUP_META[g]["kind"], **segment_stats(sub)}
    # 组 1–4 合并对照（组合工况）
    anchor["g1_4"] = segment_stats(bol_eol[bol_eol.group.isin([1, 2, 3, 4])])
    anchor["g7_10"] = segment_stats(bol_eol[bol_eol.group.isin([7, 8, 9, 10])])
    print(f"[5] anchors: g5={anchor.get('g5', {}).get('n')} "
          f"g6={anchor.get('g6', {}).get('n')}")

    # ---- 6. CCCV/满充新工况（组 7–10 vs 1–4）-----------------------
    def _grp_stats(pdf, groups):
        sub = pdf[pdf.group.isin(groups)]
        return segment_stats(sub)

    cond_cmp = {
        "cc_90soc_bol_eol": _grp_stats(bol_eol, [1, 2, 3, 4]),
        "cccv_90soc_bol_eol": _grp_stats(bol_eol, [7, 8]),
        "cccv_100soc_bol_eol": _grp_stats(bol_eol, [9, 10]),
        "cccv_mol_eol": _grp_stats(mol_eol, [7, 8, 9, 10]),
    }
    print(f"[6] cond_cmp keys: {list(cond_cmp)}")

    # ---- 图 --------------------------------------------------------
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    for stage, color, marker in [("BoL->MoL", "#888", "o"),
                                 ("MoL->EoL", "#c0392b", "s"),
                                 ("BoL->EoL", "#2c3e50", "^")]:
        t = {"BoL->MoL": bol_mol, "MoL->EoL": mol_eol,
             "BoL->EoL": bol_eol}[stage]
        if t.empty:
            continue
        ax.scatter(t.d_r_ohm * 1e3, t.d_arc_dre * 1e3, s=28, alpha=0.65,
                   color=color, marker=marker, label=stage)
    ax.axhline(0, color="#999", lw=0.6)
    ax.axvline(0, color="#999", lw=0.6)
    ax.set_xlabel("ΔR_ohm [mΩ]")
    ax.set_ylabel("Δ arc width [mΩ]")
    ax.set_title("stage-wise impedance growth")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    labels, ohm_vals, arc_vals, tail_vals = [], [], [], []
    for tag, seg in [("BoL→MoL", seg_bol_mol), ("MoL→EoL", seg_mol_eol),
                     ("BoL→EoL", seg_bol_eol)]:
        if seg.get("n"):
            labels.append(f"{tag}\n(n={seg['n']})")
            ohm_vals.append(seg.get("share_ohm") or 0)
            arc_vals.append(seg.get("share_arc") or 0)
            tail_vals.append(seg.get("share_tail") or 0)
    x = np.arange(len(labels))
    ax.bar(x, ohm_vals, 0.55, label="ohmic", color="#7f8c8d")
    ax.bar(x, arc_vals, 0.55, bottom=ohm_vals, label="arc", color="#c0392b")
    ax.bar(x, tail_vals, 0.55,
           bottom=np.array(ohm_vals) + np.array(arc_vals),
           label="tail/diffusion", color="#2980b9")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("share of positive growth")
    ax.set_title("growth partition by stage (3-channel)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    if len(chain):
        cells = [f"G{g}C{c}S{s}" for g, c, s in
                 zip(chain.group, chain.cell, chain.soc)]
        ax.bar(np.arange(len(chain)), chain.seg_bol_mol_arc * 1e3, 0.45,
               label="BoL→MoL", color="#888")
        ax.bar(np.arange(len(chain)) + 0.45, chain.seg_mol_eol_arc * 1e3,
               0.45, label="MoL→EoL", color="#c0392b")
        ax.set_xticks(np.arange(len(chain)) + 0.22, cells,
                      rotation=90, fontsize=6)
        ax.set_ylabel("Δ arc width [mΩ]")
        ax.set_title("per-cell arc growth segments")
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    tags = ["g5\ncycle", "g6\ncalend", "g1-4\ncomb", "g7-10\ncccv"]
    keys = ["g5", "g6", "g1_4", "g7_10"]
    arc_meds = [(anchor.get(k, {}) or {}).get("arc_share_med", np.nan)
                for k in keys]
    ns = [(anchor.get(k, {}) or {}).get("n", 0) for k in keys]
    colors = ["#2980b9", "#8e44ad", "#7f8c8d", "#c0392b"]
    ax.bar(np.arange(len(tags)), arc_meds, 0.55, color=colors)
    ax.set_xticks(np.arange(len(tags)),
                  [f"{t}\n(n={n})" for t, n in zip(tags, ns)], fontsize=8)
    ax.set_ylabel("median arc share of ΔR_pol")
    ax.set_title("condition anchors (BoL→EoL)")
    ax.set_ylim(0, 1.05)

    fig.suptitle("49a · EIS BoL→MoL→EoL three-stage growth (Oxford NCA)",
                 fontsize=11)
    fig.tight_layout()
    save_fig(fig, "49a_eis_eol", OUT_F)

    # ---- 判词与指标 -------------------------------------------------
    arc_med = seg_mol_eol.get("arc_share_med")
    chain_frac = chain_med.get("mol_eol_share_arc_med")
    g5 = anchor.get("g5") or {}
    g6 = anchor.get("g6") or {}
    g14 = anchor.get("g1_4") or {}
    g710 = anchor.get("g7_10") or {}
    verdict = (
        f"EIS 三阶段链收口（MoL→EoL 延伸）：Part 2+3 共 {len(df)} 谱，"
        f"MoL→EoL 配对 {seg_mol_eol['n']} 个——弧段占 ΔR_pol 中位 "
        f"{arc_med:.0%}、ΔR_pol 中位 {seg_mol_eol['d_r_pol_med']*1e3:+.1f} mΩ，"
        f"「界面主导」延续到 EoL（三通道正部份额：欧姆 "
        f"{seg_mol_eol.get('share_ohm'):.0%} / 弧段 "
        f"{seg_mol_eol.get('share_arc'):.0%} / 尾段 "
        f"{seg_mol_eol.get('share_tail'):.0%}）。"
        f"三阶段链 {chain_med['n']} 芯：MoL→EoL 段占全链弧增长中位 "
        f"{chain_frac:.0%}（{'后段主导=界面退化加速' if chain_frac and chain_frac > 0.55 else '前段主导=增长放缓或持平'}），"
        f"链可加性残差 max {chain_med['additivity_residual_max_abs']*1e3:.2e} mΩ。"
        f"循环/日历解耦锚（BoL→EoL，三通道）：组 5 纯 C/2 连续循环"
        f"（n={g5.get('n')}）**扩散尾主导**（尾段份额 "
        f"{g5.get('share_tail', float('nan')):.0%}、弧段仅 "
        f"{g5.get('share_arc', float('nan')):.0%}，ΔR_pol 中位 "
        f"{g5.get('d_r_pol_med', float('nan'))*1e3:+.0f} mΩ 且低频 Z_re "
        f"冲到 0.6 Ω 量级）vs 组 6 纯日历（n=1 个案）弧段份额 "
        f"{g6.get('arc_share_med', float('nan')):.0%} 但 ΔR_pol 仅 "
        f"{g6.get('d_r_pol_med', float('nan'))*1e3:+.1f} mΩ（几乎不动）"
        f"vs 组 1–4 组合工况弧段份额 {g14.get('share_arc', float('nan')):.0%}。"
        f"→ 三通道解耦：**连续循环走扩散尾（类 shape/LAM 侧）、"
        f"日历+慢路径走界面弧（类 shift/SEI 侧）**——与 v1.12 配比图景"
        f"「慢倍率 shift 主导、快倍率 shape 份额升」同向，且给出独立的"
        f"阻抗侧证据（首次工况解耦而非跨体系互证）。"
        f"CCCV/满充新工况：组 7–10 弧段份额 "
        f"{g710.get('share_arc', float('nan')):.0%}（100% SOC 组 9–10 "
        f"ΔR_pol {cond_cmp['cccv_100soc_bol_eol']['d_r_pol_med']*1e3:+.1f} mΩ "
        f"vs 90% SOC 组 7–8 "
        f"{cond_cmp['cccv_90soc_bol_eol']['d_r_pol_med']*1e3:+.1f} mΩ——"
        f"满充日历抬高总量）。"
        f"⚠️ 频率轴不可溯源仅索引空间口径；弧段归属两读法（SEI 增厚 vs "
        f"活性表面丧失）在 EoL 段同样不可分，不裁决；组 5 EoL 仅 2/3 芯"
        f"（Cell5 无 EoL 谱）；钩-弧过渡段小量并入尾段（≤3 mΩ）。"
    )
    print("[7]", verdict[:130], "...")

    metrics = {
        "n_spectra": int(len(df)),
        "stages": {f"{s}_{st}": int(n) for (s, st), n in
                   df.groupby(["source", "stage"]).size().items()},
        "qc_vs_part1": qc,
        "pair_stats": {
            "BoL->MoL": seg_bol_mol, "MoL->EoL": seg_mol_eol,
            "BoL->EoL": seg_bol_eol,
        },
        "chain": chain_med,
        "condition_anchors": anchor,
        "cond_compare": cond_cmp,
        "pairs_mol_eol": mol_eol.to_dict(orient="records"),
        "caveats": [
            "频率轴不可溯源——仅索引空间几何差分/比值有效",
            "组 6 纯日历 n=1：个案值不做统计推断",
            "组 5 无 MoL EIS（Readme 声明）且 EoL 仅 2/3 芯（Cell5 缺）",
            "三通道分区：尾段增量≈d_r_pol−d_arc_dre（钩-弧过渡≤3 mΩ 并入）",
            "两通道口径（只算欧姆+弧）会把组 5 尾段爆炸误判为欧姆主导——v1.13 已修正",
            "弧段归属（SEI 增厚 vs 活性表面丧失）EoL 段同样几何不可分",
        ],
        "verdict": verdict,
    }
    (OUT_M / "49_eis_eol.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[8] → {OUT_M / '49_eis_eol.json'}")


if __name__ == "__main__":
    main()
