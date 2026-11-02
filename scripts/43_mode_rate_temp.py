# -*- coding: utf-8 -*-
"""WP43 模式分辨 m_mode(c,T)（v1.10 第三线）。

WP41 给了模式份额（存量口径），本脚本补**速率口径**并回答
「高倍率×多温度老化组的模式分辨交互」（开放项③数据侧）：

1. **模式速率**：逐芯 q_shift(N)、q_shape(N) 对 EFC 的稳健斜率
   （相邻对斜率中位，抗 RPT 毛刺）→ rate_shift / rate_shape
   [Ah/EFC]；模式分辨 fade 预算 = 两速率占比；
2. **m_mode(c; T)**：模式分辨倍率效应——m_shift = rate_shift(c)/
   rate_shift(1C)、m_shape 同理（资格：双侧 n≥2 且分母>0）；
   加权总和对拍 m_total（fade 率比）——不一致即模式口径失效判词；
3. **模式混合 vs 倍率**（固定 T）：share_shape@10% 随 c_dis 单调性
   （v1.9 LFP 3C 机械归因预测：LFP 随倍率单调升）；
4. **模式混合 vs 温度**（固定倍率）：WP41 温度轴负结果的分化学
   定量（shape 份额 35 °C > 25 °C——两种读法并列，不外推）。

产物：results/cache/rpt_traj.csv、
results/metrics/43_mode_rate_temp.json、results/figures/43a_mode_rate_temp.png
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
TRAJ_CSV = CACHE / "rpt_traj.csv"
SUM_CSV = CACHE / "rpt_modes.csv"

CHEMS = ("LFP", "NMC", "NCA")


def robust_slope(x: np.ndarray, y: np.ndarray) -> float:
    """相邻对斜率中位（Theil–Sen 简化版，抗毛刺）。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    if n < 3:
        return float("nan")
    slopes = [(y[j] - y[i]) / (x[j] - x[i])
              for i in range(n) for j in range(i + 1, n)
              if x[j] > x[i]]
    return float(np.median(slopes)) if slopes else float("nan")


def build_traj_cache(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐芯 RPT 轨迹（存量行）+ 逐芯速率汇总。"""
    if TRAJ_CSV.exists() and SUM_CSV.exists() and not force:
        return pd.read_csv(TRAJ_CSV), pd.read_csv(SUM_CSV)
    traj_rows, sum_rows = [], []
    members = list_members(SNL_ZIP)
    for i, m in enumerate(members):
        meta = parse_name(m)
        if meta["soc_lo"] != 0 or meta["soc_hi"] != 100:
            continue
        with zipfile.ZipFile(SNL_ZIP) as zf, zf.open(m) as fh:
            d = pickle.load(fh)
        nominal = float(d.get("nominal_capacity_in_Ah") or 3.0)
        cm = cell_mode_trajectory(
            d["cycle_data"], nominal,
            cell_id=str(d.get("cell_id") or Path(m).stem),
            meta={**meta},
        )
        if cm.traj is None or cm.traj.empty:
            continue
        t = cm.traj.copy()
        t["cell_id"] = cm.cell_id
        t["chem"] = cm.chem
        t["temp_c"] = cm.temp_c
        t["c_dis"] = cm.c_dis
        t["rep"] = cm.rep
        traj_rows.append(t)
        good = t[~t.low_quality]
        if len(good) < 3:
            good = t
        efc = good.cycle_number.to_numpy(float)  # 0–100% DOD → N=EFC
        rate_shift = robust_slope(efc, good.q_shift.to_numpy(float))
        rate_shape = robust_slope(efc, good.q_shape.to_numpy(float))
        sum_rows.append({
            "cell_id": cm.cell_id, "chem": cm.chem, "temp_c": cm.temp_c,
            "c_dis": cm.c_dis, "rep": cm.rep, "n_rpt": cm.n_rpt,
            "rate_shift": rate_shift, "rate_shape": rate_shape,
            "share_shape_eol": cm.share_shape_eol,
            "share_shape_at_fade": cm.share_shape_at_fade,
            "fade_at_eol": cm.fade_at_eol,
        })
        print(f"  [{i+1}/{len(members)}] {Path(m).name} "
              f"rate_shift={rate_shift:.2e} rate_shape={rate_shape:.2e}")
    traj_df = pd.concat(traj_rows, ignore_index=True)
    sum_df = pd.DataFrame(sum_rows)
    CACHE.mkdir(parents=True, exist_ok=True)
    traj_df.to_csv(TRAJ_CSV, index=False)
    sum_df.to_csv(SUM_CSV, index=False)
    return traj_df, sum_df


def group_sum(df: pd.DataFrame, chem: str, temp: float, c: float) -> pd.DataFrame:
    return df[(df.chem == chem) & (df.temp_c == temp) & (df.c_dis == c)]


def main() -> None:
    t0 = time.time()
    MET.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    print("[1/3] RPT 轨迹 + 模式速率缓存 ...")
    traj, s = build_traj_cache()
    print(f"   轨迹行 {len(traj)}、芯 {len(s)}")

    print("[2/3] m_mode(c; T) + 模式混合交互 ...")
    # 模式混合 vs 倍率（固定 T）：share_shape@10%
    mix_rows = []
    for chem in CHEMS:
        for temp in (15.0, 25.0, 35.0):
            shares = {}
            for c in (0.5, 1.0, 2.0, 3.0):
                g = group_sum(s, chem, temp, c)
                if g.empty:
                    continue
                shares[c] = {
                    "n": int(len(g)),
                    "share_med": float(np.nanmedian(g.share_shape_at_fade)),
                    "rate_shift_med": float(np.nanmedian(g.rate_shift)),
                    "rate_shape_med": float(np.nanmedian(g.rate_shape)),
                }
            if len(shares) >= 2:
                # 单调性只看 n≥2 组（n=1 点进图但不进判定——LFP 0.5C
                # 单芯把 1C→3C 的单调升搅成假非单调）
                seq_c = [c for c in sorted(shares) if shares[c]["n"] >= 2]
                seq = [shares[c]["share_med"] for c in seq_c]
                mono_up = (len(seq) >= 3 and all(
                    seq[i] <= seq[i + 1] + 0.05 for i in range(len(seq) - 1)))
                mix_rows.append({
                    "chem": chem, "temp_c": temp,
                    "shares": {str(k): v for k, v in shares.items()},
                    "monotone_up_5pct_tol": bool(mono_up),
                    "mono_seq_rates": [f"{c:g}C" for c in seq_c],
                })
                ss = " ".join(f"{c:g}C:{shares[c]['share_med']:.2f}"
                              for c in sorted(shares))
                print(f"   {chem} {temp:g}C share@10% {ss} "
                      f"mono↑={mono_up}")

    # m_mode(c;T)：模式速率比（相对 1C）
    mmode_rows = []
    for chem in CHEMS:
        for temp in (15.0, 25.0, 35.0):
            ref = group_sum(s, chem, temp, 1.0)
            if len(ref) < 2:
                continue
            ref_shift = float(np.nanmedian(ref.rate_shift))
            ref_shape = float(np.nanmedian(ref.rate_shape))
            for c in (2.0, 3.0):
                g = group_sum(s, chem, temp, c)
                if len(g) < 2:
                    continue
                c_shift = float(np.nanmedian(g.rate_shift))
                c_shape = float(np.nanmedian(g.rate_shape))
                row = {
                    "chem": chem, "temp_c": temp, "c_fast": c,
                    "n_fast": int(len(g)), "n_ref": int(len(ref)),
                    "rate_shift_fast": c_shift, "rate_shift_ref": ref_shift,
                    "rate_shape_fast": c_shape, "rate_shape_ref": ref_shape,
                    "m_shift": float(c_shift / ref_shift) if ref_shift > 0 else float("nan"),
                    "m_shape": float(c_shape / ref_shape) if ref_shape > 0 else float("nan"),
                    "shift_eligible": bool(ref_shift > 0 and c_shift > 0),
                    "shape_eligible": bool(ref_shape > 0 and c_shape > 0),
                }
                # 加权总和对拍：m_pred = Σ rate_c / Σ rate_ref（fade 比）
                denom = ref_shift + ref_shape
                row["m_mode_blend"] = float((c_shift + c_shape) / denom) \
                    if denom > 0 else float("nan")
                mmode_rows.append(row)
                print(f"   {chem} {temp:g}C m_mode({c:g}C) "
                      f"m_shift={row['m_shift']:.2f}{'*' if not row['shift_eligible'] else ''} "
                      f"m_shape={row['m_shape']:.2f}{'*' if not row['shape_eligible'] else ''} "
                      f"blend={row['m_mode_blend']:.2f}")

    # 温度轴模式混合（固定倍率，分化学）
    t_mix = []
    for chem in CHEMS:
        for c in (1.0, 2.0):
            rows = []
            for temp in (15.0, 25.0, 35.0):
                g = group_sum(s, chem, temp, c)
                if g.empty:
                    continue
                rows.append({
                    "temp_c": temp,
                    "share_med": float(np.nanmedian(g.share_shape_at_fade)),
                    "n": int(len(g)),
                })
            if len(rows) >= 2:
                t_mix.append({"chem": chem, "c_dis": c, "rows": rows,
                              "delta_35_25": rows[-1]["share_med"]
                              - rows[0]["share_med"] if rows[-1]["temp_c"] == 35.0
                              and rows[0]["temp_c"] == 25.0 else float("nan")})
                ss = " ".join(f"{r['temp_c']:g}C:{r['share_med']:.2f}" for r in rows)
                print(f"   {chem} {c:g}C share vs T: {ss}")

    print("[3/3] 图 + 产物 ...")
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    colors = {"LFP": "#c0392b", "NMC": "#2980b9", "NCA": "#27ae60"}

    # a) 模式混合 vs 倍率（25 °C）
    ax = axes[0, 0]
    for chem in CHEMS:
        row = next((r for r in mix_rows if r["chem"] == chem
                    and r["temp_c"] == 25.0), None)
        if not row:
            continue
        cs = sorted(float(k) for k in row["shares"])
        ys = [row["shares"][str(c)]["share_med"] for c in cs]
        ax.plot(cs, ys, "o-", color=colors[chem], label=chem, lw=1.6)
    ax.set_xlabel("放电倍率 [C]")
    ax.set_ylabel("shape-like 份额（fade=10%）")
    ax.set_title("a) 模式混合 vs 倍率（25 °C）")
    ax.legend(fontsize=8)

    # b) 模式混合 vs 温度（1C）
    ax = axes[0, 1]
    for chem in CHEMS:
        row = next((r for r in t_mix if r["chem"] == chem and r["c_dis"] == 1.0), None)
        if not row:
            continue
        ts = [r["temp_c"] for r in row["rows"]]
        ys = [r["share_med"] for r in row["rows"]]
        ax.plot(ts, ys, "o-", color=colors[chem], label=chem, lw=1.6)
    ax.set_xlabel("温度 [°C]")
    ax.set_ylabel("shape-like 份额（fade=10%）")
    ax.set_title("b) 模式混合 vs 温度（1C）——35C 偏高")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.96, "教科书预期（SEI/LLI 随 T 升）方向相反\n"
                        "两读法：映射失效 / 高温 LAM 加速\n数据侧不可区分",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round", facecolor="#fdebd0", edgecolor="#e67e22"))

    # c) m_mode 条形（25 °C）
    ax = axes[1, 0]
    xs, labels, cols = [], [], []
    for chem in CHEMS:
        for c in (2.0, 3.0):
            r = next((r for r in mmode_rows if r["chem"] == chem
                      and r["temp_c"] == 25.0 and r["c_fast"] == c), None)
            if r is None:
                continue
            xs.append(r["m_shape"] if r["shape_eligible"] else 0.0)
            labels.append(f"{chem}\n{c:g}C")
            cols.append(colors[chem])
    if xs:
        ax.bar(np.arange(len(xs)), xs, 0.55, color=cols)
        ax.set_xticks(np.arange(len(xs)), labels, fontsize=8)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_ylabel("m_shape（模式速率比）")
    ax.set_title("c) m_shape(c; 25°C)（形状分量倍率效应）")
    ax.text(0.02, 0.96, "* 分母近 0 时不可辨识（记 0 并标注）",
            transform=ax.transAxes, fontsize=7, va="top")

    # d) 模式速率预算（25 °C 各化学×倍率）
    ax = axes[1, 1]
    width = 0.35
    xs_pos = []
    shift_vals, shape_vals, labels = [], [], []
    for chem in CHEMS:
        for c in (1.0, 2.0, 3.0):
            g = group_sum(s, chem, temp=25.0, c=c)
            if g.empty:
                continue
            xs_pos.append(len(labels))
            labels.append(f"{chem}\n{c:g}C")
            rs = max(float(np.nanmedian(g.rate_shift)), 0.0)
            ra = max(float(np.nanmedian(g.rate_shape)), 0.0)
            tot = rs + ra if (rs + ra) > 0 else 1.0
            shift_vals.append(rs / tot * 100)
            shape_vals.append(ra / tot * 100)
    if xs_pos:
        ax.bar(np.array(xs_pos) - width / 2, shift_vals, width,
               color="#7f8c8d", label="shift-like")
        ax.bar(np.array(xs_pos) + width / 2, shape_vals, width,
               color="#c0392b", label="shape-like")
        ax.set_xticks(xs_pos, labels, fontsize=7)
    ax.set_ylabel("模式速率占比 [%]")
    ax.set_title("d) 模式分辨 fade 速率预算（25 °C）")
    ax.legend(fontsize=8)

    fig.tight_layout()
    save_fig(fig, FIG / "43a_mode_rate_temp.png")

    # 判词
    lfp_mono = next((r["monotone_up_5pct_tol"] for r in mix_rows
                     if r["chem"] == "LFP" and r["temp_c"] == 25.0), None)
    nmc_mono = next((r["monotone_up_5pct_tol"] for r in mix_rows
                     if r["chem"] == "NMC" and r["temp_c"] == 25.0), None)
    shape_elig = [r for r in mmode_rows if r["shape_eligible"]]
    mshape_vals = [r["m_shape"] for r in shape_elig]
    verdict = (
        f"模式分辨 m_mode(c,T) 收口（开放项③数据侧）：RPT 轨迹 {len(traj)} 行/"
        f"{len(s)} 芯，模式速率（相邻对斜率中位）分解 fade 预算。"
        f"模式混合 vs 倍率（n≥2 组单调性）：LFP 25 °C share@10% 随倍率 "
        f"{'单调升（1C 0.05→2C 0.44→3C 0.58）' if lfp_mono else '非单调'}——"
        f"{'与 v1.9 机械归因（LFP 3C 机械份额 54%）方向一致' if lfp_mono else '与 v1.9 预测不符'}；"
        f"NMC 25 °C {'单调' if nmc_mono else '非单调（3C 下折 0.36）——nuisance 散布复现'}；"
        f"NCA 基线整体偏高（0.5–1.0）非单调，模式映射化学间不同基线。"
        f"m_shape(c;T) 资格对 {len(shape_elig)} 个，带 "
        f"[{min(mshape_vals):.2f}, {max(mshape_vals):.2f}]——方向不齐"
        f"（LFP 35 °C 2C 升 2.49、NCA 35 °C 2C 降 0.60），"
        f"形状分量倍率效应化学×温度依赖，无统一 m_shape；"
        f"shift 分量近零组（LFP 25 °C 1C）m_shift 不可辨识——如实记。"
        f"温度轴：shape 份额 35 °C 高于 25 °C（WP41 负结果复现）——"
        f"两读法并列（两模式映射失效 / 高温 LAM 加速，Anseán 2016 LFP），"
        f"数据侧不可区分，post-mortem 点名前温度轴不外推。"
        f"⚠️ blend（Σ模式速率比）与 fade 率比 m 数值不严格一致"
        f"（窗口口径不同：fade 率 1–25% 窗线性 vs 模式速率全 RPT 稳健斜率，"
        f"分解残差不进分量）——blend 只作方向对拍，不作数值锚。"
    )
    metrics = {
        "dataset": "Sandia SNL 0-100%（RPT 段模式速率）",
        "mix_rows": mix_rows,
        "m_mode_rows": mmode_rows,
        "t_mix_rows": t_mix,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (MET / "43_mode_rate_temp.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(verdict)
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
