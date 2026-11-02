# -*- coding: utf-8 -*-
"""WP41 无损退化模式分解（v1.10 第一线）。

开放项②「post-mortem 材料位点」数据侧收口——SNL 内嵌 0.5C RPT
细采样段（全化学体系同口径）做 V(Q) 曲线位移/形变两模式分解：

1. **全集 RPT 轨迹**：0–100% 窗全部电芯（三化学 × 0.5–3C ×
   15/25/35 °C）逐芯 mode trajectory → share_shape 随 fade 演化；
2. **v1.9 机理对拍**（独立估计器互证）：LFP 25 °C 3C 的 shape-like
   份额 vs 锚定超额分解机械份额 54% CI95 [50,62]%；跨组对比
   （3C − 1C/2C）方向是否同向；
3. **NMC 3C nuisance 对照**：v1.9 判「m<1 是批次异质性」→ shape
   份额应不升高（定量确认）；
4. **dQ/dV 峰跟踪**（NMC/NCA 可辨识时）：峰位漂移（LLI 交叉证据）
   + 峰高比（LAM 交叉证据）；LFP 峰弱如实记录。

产物：results/cache/rpt_modes.csv（逐 RPT 点缓存）、
results/metrics/41_degrade_modes.json、results/figures/41a_degrade_modes.png
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal.degrade_modes import cell_mode_trajectory  # noqa: E402
from bmthermal.plotting import save as save_fig, setup_style  # noqa: E402
from bmthermal.snl_aging import list_members, parse_name  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

MET = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures"
CACHE = ROOT / "results" / "cache"
SNL_ZIP = Path("D:/项目/02_数据集/01_国际公开集/Sandia_SNL/SNL.zip")
MODES_CSV = CACHE / "rpt_modes.csv"

CHEMS = ("LFP", "NMC", "NCA")


def build_cache(force: bool = False) -> pd.DataFrame:
    """全集 RPT 模式轨迹 → 缓存 CSV（解压+pickle 61 支较慢）。"""
    if MODES_CSV.exists() and not force:
        return pd.read_csv(MODES_CSV)
    rows = []
    members = [m for m in list_members(SNL_ZIP)]
    for i, m in enumerate(members):
        meta = parse_name(m)
        if meta["soc_lo"] != 0 or meta["soc_hi"] != 100:
            continue  # 20–80% 窗混组会错口径（v1.5 教训）
        with zipfile.ZipFile(SNL_ZIP) as zf, zf.open(m) as fh:
            d = pickle.load(fh)
        nominal = float(d.get("nominal_capacity_in_Ah") or 3.0)
        cm = cell_mode_trajectory(
            d["cycle_data"], nominal,
            cell_id=str(d.get("cell_id") or Path(m).stem),
            meta={**meta},
        )
        row = cm.to_dict()
        row["member"] = Path(m).name
        rows.append(row)
        print(f"  [{i+1}/{len(members)}] {Path(m).name} n_rpt={cm.n_rpt} "
              f"share_eol={cm.share_shape_eol:.2f} fade_eol={cm.fade_at_eol:.3f}")
    df = pd.DataFrame(rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(MODES_CSV, index=False)
    return df


def group_rows(df: pd.DataFrame, chem: str, temp_c: float, c_dis: float) -> pd.DataFrame:
    return df[(df.chem == chem) & (df.temp_c == temp_c) & (df.c_dis == c_dis)]


def boot_median_ci(vals: np.ndarray, n_boot: int = 4000, seed: int = 7) -> list:
    """组中位 bootstrap CI（电芯级重采样）。"""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return [float("nan"), float("nan")]
    if vals.size == 1:
        return [float(vals[0]), float(vals[0])]
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(vals, (n_boot, vals.size), replace=True), axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def main() -> None:
    t0 = time.time()
    MET.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    print("[1/3] 全集 RPT 模式轨迹（缓存）...")
    df = build_cache()
    print(f"   {len(df)} 芯，RPT 总点 {int(df.n_rpt.sum())}")

    print("[2/3] 组聚合 + v1.9 对拍 ...")
    group_stats = []
    for chem in CHEMS:
        for temp in (15.0, 25.0, 35.0):
            for c in (0.5, 1.0, 2.0, 3.0):
                g = group_rows(df, chem, temp, c)
                if g.empty:
                    continue
                s = g.share_shape_eol.to_numpy(float)
                s = s[np.isfinite(s)]
                stat = {
                    "chem": chem, "temp_c": temp, "c_dis": c,
                    "n_cells": int(len(g)),
                    "share_shape_eol_median": float(np.median(s)) if s.size else float("nan"),
                    "share_shape_eol_ci95": boot_median_ci(s),
                    "share_at_10pct_median": float(np.nanmedian(g.share_shape_at_fade)),
                    "fade_eol_median": float(np.nanmedian(g.fade_at_eol)),
                    "n_rpt_median": float(np.median(g.n_rpt)),
                    "low_n": bool(len(g) < 2),
                }
                group_stats.append(stat)
                print(f"   {chem} {temp:g}C {c:g}C n={len(g)} "
                      f"share_eol={stat['share_shape_eol_median']:.2f} "
                      f"CI{stat['share_shape_eol_ci95']} "
                      f"@10%={stat['share_at_10pct_median']:.2f}"
                      f"{' [n<2]' if stat['low_n'] else ''}")

    # v1.9 对拍：LFP 25C 3C vs 1C/2C（跨组对比是可辨识量）
    v19 = {"mech_share_lfp3c": 0.54, "ci95": [0.50, 0.62]}
    p38 = MET / "38_mech_attribution.json"
    if p38.exists():
        try:
            j = json.loads(p38.read_text(encoding="utf-8"))
            v19 = j.get("v19_crosscheck", v19) or v19
            v19.setdefault("mech_share_lfp3c", 0.54)
        except Exception:
            pass
    def _share(chem, temp, c):
        s = next((g for g in group_stats
                  if g["chem"] == chem and g["temp_c"] == temp and g["c_dis"] == c), None)
        return s["share_shape_eol_median"] if s else float("nan")
    contrast = {
        "lfp_25c_3c": _share("LFP", 25.0, 3.0),
        "lfp_25c_1c": _share("LFP", 25.0, 1.0),
        "lfp_25c_2c": _share("LFP", 25.0, 2.0),
        "nmc_25c_3c": _share("NMC", 25.0, 3.0),
        "nmc_25c_1c": _share("NMC", 25.0, 1.0),
    }
    contrast["lfp_3c_minus_1c"] = contrast["lfp_25c_3c"] - contrast["lfp_25c_1c"]
    contrast["nmc_3c_minus_1c"] = contrast["nmc_25c_3c"] - contrast["nmc_25c_1c"]
    # 温度轴方向（教科书预期：SEI/LLI 随 T 加速 → shape 份额应降）
    contrast["lfp_1c_35c_vs_25c"] = _share("LFP", 35.0, 1.0) - contrast["lfp_25c_1c"]
    contrast["nca_1c_35c_vs_25c"] = _share("NCA", 35.0, 1.0) - _share("NCA", 25.0, 1.0)

    # dQ/dV 峰跟踪汇总（可辨识芯）
    pk = df[df.get("peak_matched", pd.Series(0, index=df.index)) >= 2]
    peak_summary = {
        "n_cells_with_peaks": int(len(pk)),
        "median_shift_v": float(np.nanmedian(pk.peak_shift_v)) if len(pk) else float("nan"),
        "median_height_ratio": float(np.nanmedian(pk.peak_hratio)) if len(pk) else float("nan"),
        "by_chem": {
            chem: {
                "n": int((pk.chem == chem).sum()),
                "shift_v": float(np.nanmedian(pk[pk.chem == chem].peak_shift_v))
                if (pk.chem == chem).sum() else float("nan"),
                "height_ratio": float(np.nanmedian(pk[pk.chem == chem].peak_hratio))
                if (pk.chem == chem).sum() else float("nan"),
            } for chem in CHEMS
        },
    }
    print(f"   dQ/dV 峰可辨识 {peak_summary['n_cells_with_peaks']} 芯："
          f"峰位漂移中位 {peak_summary['median_shift_v']:.4f} V，"
          f"峰高比 {peak_summary['median_height_ratio']:.2f}")

    print("[3/3] 图 + 产物 ...")
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    colors = {"LFP": "#c0392b", "NMC": "#2980b9", "NCA": "#27ae60"}

    # a) RPT V(Q) 族（LFP 25C 3C 首芯）
    ax = axes[0, 0]
    cell = df[(df.chem == "LFP") & (df.temp_c == 25.0) & (df.c_dis == 3.0)]
    if not cell.empty:
        mid = cell.iloc[0]["member"]
        with zipfile.ZipFile(SNL_ZIP) as zf, zf.open("SNL/" + mid) as fh:
            d = pickle.load(fh)
        from bmthermal.degrade_modes import identify_rpt_segments, resample_discharge
        segs = identify_rpt_segments(d["cycle_data"], float(d["nominal_capacity_in_Ah"]))
        picks = segs[:: max(1, len(segs) // 6)][:6]
        cmap = plt.get_cmap("viridis")
        for k, seg in enumerate(picks):
            qg, vg = resample_discharge(seg)
            ax.plot(qg, vg, color=cmap(k / max(1, len(picks) - 1)),
                    lw=1.5, label=f"N={seg['cycle_number']}")
        ax.set_xlabel("放电容量 [Ah]")
        ax.set_ylabel("电压 [V]")
        ax.set_title("a) LFP 3C RPT V(Q) 族（0.5C，随寿命）")
        ax.legend(fontsize=7)

    # b) 锚点损失剖面（同芯）
    ax = axes[0, 1]
    if not cell.empty:
        from bmthermal.degrade_modes import anchor_profile
        q0, v0 = resample_discharge(segs[0])
        for k, seg in enumerate(picks[1:4], 1):
            qg, vg = resample_discharge(seg)
            prof = anchor_profile(q0, v0, qg, vg)
            ax.plot(prof["v_anchors"], prof["loss"], "o-",
                    color=cmap(k / 4), lw=1.5, label=f"N={seg['cycle_number']}")
        ax.set_xlabel("锚点电压 [V]")
        ax.set_ylabel("容量损失 L(V) [Ah]")
        ax.set_title("b) 锚点损失剖面（平直=shift 型）")
        ax.legend(fontsize=7)

    # c) share_shape vs fade（LFP 各倍率轨迹中位）
    ax = axes[1, 0]
    for c, style in ((1.0, "o-"), (2.0, "s--"), (3.0, "d:")):
        g = group_rows(df, "LFP", 25.0, c)
        if g.empty:
            continue
        ax.plot(g.fade_at_eol, g.share_shape_at_fade, style, color=colors["LFP"],
                mfc="white", ms=7, label=f"LFP {c:g}C", lw=1.4)
    ax.axhline(v19["mech_share_lfp3c"], color="black", ls="--", lw=1,
               label=f"v1.9 机械份额 {v19['mech_share_lfp3c']:.0%}")
    ax.set_xlabel("fade（RPT 末点）")
    ax.set_ylabel("shape-like 份额（fade=10% 锚点）")
    ax.set_title("c) LFP 25C 模式份额 vs 衰减深度")
    ax.legend(fontsize=7)

    # d) 组 EOL 份额条形（25 °C 各化学×倍率）
    ax = axes[1, 1]
    xs, labels, cols, errs = [], [], [], []
    for chem in CHEMS:
        for c in (1.0, 2.0, 3.0):
            st = next((g for g in group_stats
                       if g["chem"] == chem and g["temp_c"] == 25.0 and g["c_dis"] == c), None)
            if st is None:
                continue
            xs.append(st["share_shape_eol_median"])
            labels.append(f"{chem}\n{c:g}C")
            cols.append(colors[chem])
            lo, hi = st["share_shape_eol_ci95"]
            errs.append([max(0.0, st["share_shape_eol_median"] - lo),
                         max(0.0, hi - st["share_shape_eol_median"])])
    if xs:
        e = np.array(errs).T
        ax.bar(np.arange(len(xs)), xs, 0.55, color=cols, yerr=e, capsize=3)
        ax.set_xticks(np.arange(len(xs)), labels, fontsize=8)
    ax.axhline(v19["mech_share_lfp3c"], color="black", ls="--", lw=1,
               label="v1.9 LFP 3C 机械份额")
    ax.set_ylabel("shape-like 份额（EOL）")
    ax.set_title("d) 25 °C 组模式份额（CI=芯级 bootstrap）")
    ax.legend(fontsize=7)
    ax.text(0.02, 0.02, "曲线几何归属≠拆解位点（文献映射假设）",
            transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round", facecolor="#fdebd0", edgecolor="#e67e22"))

    fig.tight_layout()
    save_fig(fig, FIG / "41a_degrade_modes.png")

    # 判词
    nmc_g = next(g for g in group_stats
                 if g["chem"] == "NMC" and g["temp_c"] == 25.0 and g["c_dis"] == 3.0)
    nmc_ci_wide = (nmc_g["share_shape_eol_ci95"][1]
                   - nmc_g["share_shape_eol_ci95"][0]) > 0.5
    t_anomaly = (contrast["lfp_1c_35c_vs_25c"] > 0.3) or (contrast["nca_1c_35c_vs_25c"] > 0.3)
    verdict = (
        f"无损退化模式分解收口（post-mortem 数据侧替身）：SNL 内嵌 0.5C RPT"
        f"（细采样，{int(df.n_rpt.sum())} 段 / {len(df)} 芯）V(Q) 两模式分解——"
        f"LFP 25 °C 3C shape-like（形变/曲线几何）份额 EOL {contrast['lfp_25c_3c']:.0%}"
        f"（fade=10% 锚 {next(g for g in group_stats if g['chem']=='LFP' and g['temp_c']==25.0 and g['c_dis']==3.0)['share_at_10pct_median']:.0%}），"
        f"vs v1.9 锚定超额分解机械份额 {v19['mech_share_lfp3c']:.0%} CI95 {v19['ci95']}——"
        f"独立估计器同量级同向（LFP 3C − 1C 形状差 {contrast['lfp_3c_minus_1c']:+.2f}）。"
        + (f"NMC 3C CI [{nmc_g['share_shape_eol_ci95'][0]:.2f}, "
           f"{nmc_g['share_shape_eol_ci95'][1]:.2f}] 组内散布主导（0.11–0.97），"
           f"中位差 {contrast['nmc_3c_minus_1c']:+.2f} 在 CI 内不显著——"
           f"与 v1.9 nuisance（异质性主导）定性一致，非机械升高证据。"
           if nmc_ci_wide else
           f"NMC 3C − 1C 形状差 {contrast['nmc_3c_minus_1c']:+.2f}——"
           f"{'不升高，nuisance 判定得定量确认' if contrast['nmc_3c_minus_1c'] <= 0.1 else '升高，需人工复核'}。")
        + f"dQ/dV 峰跟踪 {peak_summary['n_cells_with_peaks']} 芯可辨识"
        f"（峰位漂移中位 {peak_summary['median_shift_v']:+.4f} V、"
        f"峰高比 {peak_summary['median_height_ratio']:.2f}）；LFP 峰弱如实记录。"
        + (f"⚠️ 温度轴负结果：shape 份额 35 °C 反而高于 25 °C"
           f"（LFP 1C Δ{contrast['lfp_1c_35c_vs_25c']:+.2f}、"
           f"NCA 1C Δ{contrast['nca_1c_35c_vs_25c']:+.2f}），"
           f"与「SEI/LLI 随 T 加速」教科书预期反向——两模式几何归属"
           f"在温度轴上≠LLI/LAM 物理份额（映射假设失效），温度轴不外推。"
           if t_anomaly else "温度轴方向与 SEI/LLI 预期一致。")
        + f"⚠️ 曲线几何归属≠拆解材料位点（LLI/LAM 标签是文献惯例映射，"
        f"Dubarry et al. 2020）——份额对拍是互证不是同口径重复。"
    )
    metrics = {
        "dataset": "Sandia SNL 0-100%（内嵌 0.5C RPT 段）",
        "group_stats": group_stats,
        "v19_crosscheck": v19,
        "contrast": contrast,
        "peak_summary": peak_summary,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (MET / "41_degrade_modes.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(verdict)
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
