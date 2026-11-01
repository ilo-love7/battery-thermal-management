"""32 · 绝对寿命锚点：SNL fade_per_efc → 各体系 N_EOL 重标 + 形状检验。

背景（报告开放项 #3）
---------------------
v1.0–v1.6 的损伤模型只报相对寿命倍数（A 未标定，D_EOL 在比值里
消掉）。scripts/18 用 ANCHOR_EFC=2000（LFP 文献典型，**假设**）把
相对比换成绝对 EFC——那是脚手架，不是实测。本脚本用 SNL 61 节的
实测 fade_per_efc 按体系重标绝对尺度，并检验外推形状。

三问三答：
1. **各体系绝对 N_EOL 是多少？** 参考工况（25 °C、0.5C 充/1C 放、
   0–100%）线性口径：LFP ~8.0e3、NMC ~7.3e2、NCA ~5.6e2 EFC——
   旧锚 2000 对 LFP 偏悲观 4×，对 NMC/NCA 偏乐观 0.3×。
   「一个锚打天下」作废，体系差 14×。
2. **线性窗口外推可信吗？** 用 CS2 到达 EOL 的电芯审计三种口径：
   直接交叉 vs 线性窗口 vs 幂律 f=a·N^b。CS2 b 中位 0.76（亚线性）
   → 线性口径是 N_EOL 下界；幂律口径修正方向向上。
3. **绝对尺度怎么进模型？** A_chem = fade_ref/d_ref（每损伤秒衰减），
   场景 N_EOL = FADE_EOL/(A·d_s·m(c))。跨电芯格式迁移只保证量级。

产出：
  results/metrics/32_abs_life.json
  results/figures/32a_abs_life.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal import plotting  # noqa: E402
from bmthermal.abs_life import (  # noqa: E402
    ChemAnchor, audit_extrapolation, n_eol_linear, n_eol_power,
)
from bmthermal.aging import damage_integral  # noqa: E402
from bmthermal.heat import q_profile  # noqa: E402
from bmthermal.network import build_cell, simulate_two_node  # noqa: E402
from bmthermal.params import SCHEMES  # noqa: E402
from bmthermal.snl_aging import load_dataset  # noqa: E402

OUT = ROOT / "results"
MET = OUT / "metrics"
CACHE = OUT / "cache"
FIG = OUT / "figures"
for p in (MET, CACHE, FIG):
    p.mkdir(parents=True, exist_ok=True)

SNL_ZIP = Path("D:/项目/02_数据集/01_国际公开集/Sandia_SNL/SNL.zip")
SNL_CSV = CACHE / "snl_summary.csv"
REF_TEMP = 25.0
REF_C_DIS = 1.0
OLD_ANCHOR_EFC = 2000.0
FADE_EOL = 0.2


def ensure_snl_cache() -> pd.DataFrame:
    """缓存缺形状列则重建（v1.7 扩列）。"""
    if SNL_CSV.exists():
        df = pd.read_csv(SNL_CSV)
        if "b_fade" in df.columns:
            return df
        SNL_CSV.unlink()
    return load_dataset(SNL_ZIP, SNL_CSV)


def ref_damage_s() -> tuple[float, dict]:
    """参考工况（1C 自然，25 °C）每 EFC 损伤（等效 25 °C 秒）。"""
    cache: dict = {}
    t_arr, q_arr, info = q_profile(1.0, t_amb_c=25.0, cache=cache)
    scheme = SCHEMES["natural"]
    net = build_cell(h=scheme.h_mid,
                     pcm_extra_capacity_j_k=scheme.pcm_extra_capacity_j_k)
    out = simulate_two_node(t_arr, q_arr, 25.0, net)
    d_ref = damage_integral(t_arr, out["t_core_c"])
    return float(d_ref), {
        "d_ref_s": float(d_ref),
        "d_ref_h": float(d_ref) / 3600.0,
        "span_s": float(t_arr[-1] - t_arr[0]),
        "q_source": info.get("source", ""),
        "t_core_peak_c": float(np.max(out["t_core_c"])),
    }


def cs2_curves() -> list:
    """CS2 到达 EOL 的电芯曲线（逐循环缓存）。"""
    curves = []
    for f in sorted(CACHE.glob("cs2_*_per_cycle.parquet")):
        df = pd.read_parquet(f)
        if "cycle" not in df.columns or "fade" not in df.columns:
            continue
        n = df["cycle"].to_numpy(float)
        fade = df["fade"].to_numpy(float)
        ok = np.isfinite(n) & np.isfinite(fade)
        curves.append((f.stem.replace("cs2_", "").replace("_per_cycle", ""),
                       n[ok], fade[ok]))
    return curves


def main() -> None:
    t0 = time.time()

    print("[1/4] SNL 电芯与形状列 ...")
    df = ensure_snl_cache()
    df = df[df["fade_rate_per_efc"].notna()].copy()
    print(f"   {len(df)} 支有效电芯")

    d_ref, d_info = ref_damage_s()
    print(f"   参考工况 d_ref = {d_ref:.0f} s/EFC（{d_ref/3600:.2f} h 等效 25 °C）")

    # ---- 体系锚（25 °C, 1C, 0–100%）----
    print("[2/4] 体系锚点 ...")
    anchors: dict[str, ChemAnchor] = {}
    anchor_rows = []
    for chem in ("LFP", "NMC", "NCA"):
        g = df[(df.chem == chem) & (df.temp_c == REF_TEMP)
               & (df.c_dis == REF_C_DIS) & (df.soc_lo == 0) & (df.soc_hi == 100)]
        if g.empty:
            print(f"   {chem}: 无参考组，跳过")
            continue
        fade_ref = float(g["fade_rate_per_efc"].mean())
        fade_std = float(g["fade_rate_per_efc"].std(ddof=1)) if len(g) > 1 else 0.0
        b_med = float(np.nanmedian(g["b_fade"])) if g["b_fade"].notna().any() else float("nan")
        a_med = float(np.nanmedian(g["a_fade"])) if g["a_fade"].notna().any() else float("nan")
        cross_med = float(np.nanmedian(g["n_eol_cross"])) if g["n_eol_cross"].notna().any() else float("nan")
        anc = ChemAnchor(
            chem=chem, temp_c=REF_TEMP, c_dis=REF_C_DIS,
            fade_ref_per_efc=fade_ref, n_cells=int(len(g)),
            fade_std_per_efc=fade_std, d_ref_s=d_ref,
            b_fade=b_med, a_fade=a_med, n_eol_cross_median=cross_med,
            note="SNL 18650 实测；跨格式迁移只保证量级",
        )
        anchors[chem] = anc
        anchor_rows.append(anc.to_dict())
        print(f"   {chem}: fade={fade_ref:.3e}/EFC (n={len(g)}, std={fade_std:.1e}) "
              f"N_EOL_lin={anc.n_eol_ref:.0f}  N_EOL_pow={n_eol_power(a_med, b_med):.0f}  "
              f"b={b_med:.2f}")

    # ---- 形状审计（CS2 到达 EOL）----
    print("[3/4] 外推形状审计（CS2）...")
    audit = audit_extrapolation(cs2_curves())
    aud = audit.to_dict()
    med_l, med_p = aud["err_linear_median_pct"], aud["err_power_median_pct"]
    # 两种口径都偏乐观 → SNL 线性/幂律 N_EOL 是上界侧，给下修中心带
    bias = float(np.nanmedian([med_l, med_p]))
    if np.isfinite(bias) and bias > 5:
        corr = 1.0 / (1.0 + bias / 100.0)
        shape_verdict = (
            f"CS2 全寿命审计：线性窗口外推偏乐观 +{med_l:.0f}%、"
            f"幂律 +{med_p:.0f}%（参考=平滑实测交叉；单点毛刺已滤）——"
            f"SNL 的 N_EOL 按审计中位下修 ×{corr:.2f}（中心带），"
            f"线性/幂律原值保留为上界侧。"
        )
    elif np.isfinite(bias) and abs(bias) <= 10:
        corr = 1.0
        shape_verdict = (
            f"线性偏差中位 {med_l:+.0f}%、幂律 {med_p:+.0f}%——"
            "两种口径同阶，SNL N_EOL 直接用（带形状带）。"
        )
    else:
        corr = 1.0
        shape_verdict = (
            f"线性偏差中位 {med_l:+.0f}%、幂律 {med_p:+.0f}%——"
            "外推形状不确定度显著且方向不一，N_EOL 报双口径带并标注量级误差。"
        )
    aud["verdict"] = shape_verdict
    aud["snl_correction_factor"] = corr
    print("   " + shape_verdict)

    # SNL 侧形状分布（未到 EOL，只看 b）
    snl_b = df.loc[df["b_fade"].notna() & (df.soc_lo == 0), "b_fade"]
    shape_snl = {
        "b_median": float(snl_b.median()),
        "b_p25": float(snl_b.quantile(0.25)),
        "b_p75": float(snl_b.quantile(0.75)),
        "n_cells": int(snl_b.size),
        "share_b_lt_1": float((snl_b < 1.0).mean()),
    }

    # ---- 锚点替换对照 ----
    print("[4/4] 锚点对照与产物 ...")
    old_vs_new = []
    for chem, anc in anchors.items():
        n_pow = n_eol_power(anc.a_fade, anc.b_fade)
        old_vs_new.append({
            "chem": chem,
            "old_anchor_efc": OLD_ANCHOR_EFC,
            "new_n_eol_linear": anc.n_eol_ref,
            "new_n_eol_power": n_pow,
            "new_n_eol_corrected": float(np.nanmedian([anc.n_eol_ref, n_pow]) * corr),
            "ratio_old_over_new": OLD_ANCHOR_EFC / anc.n_eol_ref,
        })
        print(f"   {chem}: 旧锚 2000 → 新 {anc.n_eol_ref:.0f} EFC "
              f"（修正中心 {old_vs_new[-1]['new_n_eol_corrected']:.0f}，"
              f"旧/新 = {OLD_ANCHOR_EFC/anc.n_eol_ref:.2f}）")

    # 体系跨度
    n_vals = [a.n_eol_ref for a in anchors.values()]
    spread = float(max(n_vals) / min(n_vals)) if n_vals else float("nan")

    metrics = {
        "dataset": "Sandia SNL（Preger 2020）61 节 + CALCE CS2 形状审计",
        "reference_condition": {
            "temp_c": REF_TEMP, "c_dis": REF_C_DIS,
            "c_chg": 0.5, "soc_window": [0, 100],
            "damage_model_ref": d_info,
        },
        "chem_anchors": anchor_rows,
        "extrapolation_audit": aud,
        "snl_shape": shape_snl,
        "old_anchor_table": {
            "old_anchor_efc": OLD_ANCHOR_EFC,
            "old_anchor_note": "scripts/18 LFP 文献典型假设（非实测）",
            "rows": old_vs_new,
            "chem_spread_ratio": spread,
        },
        "verdict": (
            f"绝对寿命按体系重标：LFP {anchors['LFP'].n_eol_ref:.0f} / "
            f"NMC {anchors['NMC'].n_eol_ref:.0f} / NCA {anchors['NCA'].n_eol_ref:.0f} EFC"
            f"（线性口径上界侧，25 °C 1C），体系跨度 {spread:.0f}×——"
            f"旧单锚 2000 EFC 作废。{shape_verdict}"
            "跨电芯格式（SNL 18650 → 基准模型电芯）只保证量级；"
            "A_chem = fade_ref/d_ref 按体系入账，场景 N_EOL=FADE_EOL/(A·d_s·m(c))。"
        ),
        "elapsed_s": time.time() - t0,
    }
    (MET / "32_abs_life.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 图 32a ----
    plotting.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    chems = [a.chem for a in anchors.values()]
    n_lin = [a.n_eol_ref for a in anchors.values()]
    n_pow = [n_eol_power(a.a_fade, a.b_fade) for a in anchors.values()]
    x = np.arange(len(chems))
    w = 0.35
    ax.bar(x - w / 2, n_lin, w, label="线性窗口", color="#2980b9")
    ax.bar(x + w / 2, n_pow, w, label="幂律", color="#c0392b")
    ax.axhline(OLD_ANCHOR_EFC, color="k", ls="--", lw=1.2,
               label=f"旧锚 {OLD_ANCHOR_EFC:.0f} EFC")
    ax.set_yscale("log")
    ax.set_xticks(x, chems)
    ax.set_ylabel("N_EOL @25 °C 1C（EFC，对数轴）")
    ax.set_title("a) 各体系绝对寿命锚点")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    b_snl = df.loc[(df.soc_lo == 0) & df["b_fade"].notna(), ["chem", "b_fade"]]
    for i, chem in enumerate(chems):
        vals = b_snl.loc[b_snl.chem == chem, "b_fade"].dropna().values
        if vals.size:
            ax.scatter(np.full(vals.size, i) + np.random.uniform(-0.12, 0.12, vals.size),
                       vals, s=28, alpha=0.75, label=chem if i == 0 else None)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="线性 b=1")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="SEI b=0.5")
    ax.axhline(shape_snl["b_median"], color="#c0392b", lw=1.2,
               label=f"SNL 中位 {shape_snl['b_median']:.2f}")
    ax.set_xticks(range(len(chems)), chems)
    ax.set_ylabel("幂律指数 b")
    ax.set_title("b) SNL 衰减形状（1–25% 窗）")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    rows = aud.get("rows", [])
    if rows:
        nc = [r["n_cross"] for r in rows]
        ax.scatter(nc, [r["n_linear"] for r in rows], s=36, label="线性窗口",
                   color="#2980b9")
        ax.scatter(nc, [r["n_power"] for r in rows], s=36, marker="s",
                   label="幂律", color="#c0392b")
        lim = [0, max(nc) * 1.15]
        ax.plot(lim, lim, "k--", lw=1, label="理想（=直接交叉）")
        ax.set_xlabel("实测 N(f=20%)（EFC）")
        ax.set_ylabel("外推 N_EOL（EFC）")
        ax.set_title("c) 外推审计（CS2 到 EOL 电芯）")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "无到达 EOL 曲线", ha="center", va="center")
        ax.set_title("c) 外推审计")

    ax = axes[1, 1]
    labels = [r["chem"] for r in old_vs_new]
    ratios = [r["ratio_old_over_new"] for r in old_vs_new]
    colors = ["#c0392b" if r > 1 else "#27ae60" for r in ratios]
    ax.bar(labels, ratios, color=colors, width=0.5)
    ax.axhline(1.0, color="k", lw=1)
    ax.set_ylabel("旧锚 2000 / 新 N_EOL")
    ax.set_title("d) 旧假设锚的偏差（>1=旧锚偏悲观）")
    for i, r in enumerate(ratios):
        ax.text(i, r, f"{r:.2f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("32a · 绝对寿命锚点：SNL 实测重标与外推形状检验", fontsize=12)
    fig.tight_layout()
    plotting.save(fig, FIG / "32a_abs_life.png")
    plt.close(fig)

    print(f"done in {metrics['elapsed_s']:.0f}s -> 32_abs_life.json / 32a_abs_life.png")
    print("verdict:", metrics["verdict"][:200])


if __name__ == "__main__":
    main()
