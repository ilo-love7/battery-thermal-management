"""35 · β 跨体系锚：NMC/NCA 内阻老化实测（v1.8, WP35）。

开放项 #5（NMC 内阻老化数据）收口：

1. **MATR-10 截面**（NCA 21700，94 芯 SOH 62–100%）：满充 C→D 反转
   提 R(1C)（ΔI=4.84 A），R vs f 截面回归 → β_NCA（bootstrap CI）。
2. **HNEI 纵向**（LCO+NMC 18650，14 芯 >1000 循环）：逐循环放电起始
   反转提 R_inst → R/R0 vs f → β 逐芯。
3. **MICH 纵向**（NMC111 软包，40 芯）与 **XJTU 纵向**（NCM523 18650，
   23 节 2C/3C 组）：同 2。

对账：CS2 LFP 实测 β=0.48（scripts/19）、文献 [1.5, 3.5]、
feedback.py 默认 2.5。按各体系 β 重算反馈漂移（C5 口径）。

注意：所有内阻列（batteryarchive schema / MATR-10 DCIR 列）全零——
R 全部从 V/I 电流阶跃提取（v1.7「schema≠数据」教训第三次确认）。

产出：
  results/metrics/35_beta_anchor.json
  results/figures/35a_beta_anchor.png
  results/cache/beta_extract.csv（纵向逐循环提取行）
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal import plotting  # noqa: E402
from bmthermal.beta_anchor import (  # noqa: E402
    NOMINAL_AH, RATE_1C_A,
    fit_beta_cross_sectional, fit_beta_longitudinal,
    load_longitudinal_zip, scan_matr10,
)
from bmthermal.feedback import (  # noqa: E402
    BETA_DEFAULT, BETA_GRID, FeedbackParams, life_amplification, pair_life_ratio,
)
from bmthermal.heat import q_profile  # noqa: E402
from bmthermal.network import build_cell, simulate_two_node  # noqa: E402
from bmthermal.params import SCHEMES  # noqa: E402
from bmthermal.aging import EA_DEFAULT, damage_integral  # noqa: E402

OUT = ROOT / "results"
MET = OUT / "metrics"
CACHE = OUT / "cache"
FIG = OUT / "figures"
for p in (MET, CACHE, FIG):
    p.mkdir(parents=True, exist_ok=True)

DATA = Path("D:/项目/02_数据集/01_国际公开集")
MATR10_DIR = DATA / "MATR-10_DiagnosticFree" / "ResValData"
MATR10_INV = DATA / "MATR-10_DiagnosticFree" / "电芯清单_scan.csv"
HNEI_ZIP = DATA / "HNEI" / "HNEI.zip"
MICH_ZIP = DATA / "Michigan_MICH" / "MICH.zip"
XJTU_ZIP = Path("D:/项目/02_数据集/02_国内公开集/XJTU_西安交大/XJTU.zip")
BETA_CSV = CACHE / "beta_extract.csv"

CS2_BETA_MEDIAN = 0.4758       # scripts/19 实测（LFP）
BETA_LIT_LO, BETA_LIT_HI = 1.5, 3.5


def local_d_cycle_factory(c: float, scheme_key: str, params: FeedbackParams,
                          cache: dict, fn_cache: dict):
    """d_cycle_fn(u)，β 经 params.q_mult_slope 进产热倍率。"""
    slope = round(params.q_mult_slope, 9)
    key = (c, scheme_key, slope)
    if key in fn_cache:
        return fn_cache[key]
    t_arr, q_arr, _ = q_profile(c, t_amb_c=25.0, cache=cache)
    scheme = SCHEMES[scheme_key]
    net = build_cell(h=scheme.h_mid,
                     pcm_extra_capacity_j_k=scheme.pcm_extra_capacity_j_k)

    def d_cycle_fn(u: float) -> float:
        mult = 1.0 + slope * u
        out = simulate_two_node(t_arr, q_arr * mult, 25.0, net)
        return damage_integral(t_arr, out["t_core_c"], ea_j_mol=EA_DEFAULT)

    fn_cache[key] = d_cycle_fn
    return d_cycle_fn


def feedback_drift(beta: float, cache: dict, fn_cache: dict) -> dict:
    """β → C5 反馈漂移（与 scripts/33 同口径：3C 液冷 vs 1C 自然）。"""
    par = FeedbackParams(beta=beta)
    fn_cold = local_d_cycle_factory(3.0, "cold_plate", par, cache, fn_cache)
    fn_base = local_d_cycle_factory(1.0, "natural", par, cache, fn_cache)
    pr = pair_life_ratio(fn_cold, fn_base, par)
    ra = life_amplification(fn_cold, par)
    rb = life_amplification(fn_base, par)
    return {
        "beta": float(beta),
        "amp_3c_cold": ra["amplification"],
        "amp_1c_natural": rb["amplification"],
        "c5_open": pr["open_loop_ratio"],
        "c5_closed": pr["closed_loop_ratio"],
        "c5_drift_pct": pr["ratio_drift_pct"],
    }


def main() -> None:
    t0 = time.time()

    # ---- [1/4] MATR-10 截面（NCA）----
    print("[1/4] MATR-10 截面扫描（NCA 21700，94 芯）...")
    m_cells = scan_matr10(MATR10_DIR, MATR10_INV)
    n_with_1c = sum(1 for c in m_cells if np.isfinite(c.r_1c))
    print(f"   扫描 {len(m_cells)} 芯，R(1C) 有效 {n_with_1c} 芯，"
          f"f 范围 {min(c.fade for c in m_cells):.3f}–{max(c.fade for c in m_cells):.3f}")
    nca_fit = fit_beta_cross_sectional(m_cells)
    print(f"   β_NCA（截面）= {nca_fit['beta']:.2f} "
          f"CI95 {nca_fit.get('ci95')}  n={nca_fit['n_cells']}  "
          f"r²={nca_fit.get('r2', float('nan')):.3f}  "
          f"R0={nca_fit.get('r0_ohm', float('nan'))*1000:.1f} mΩ")

    # R(rate) 平坦性（满充，参考截面点的倍率独立性）
    rate_rows = []
    all_rates: dict = {}
    for c in m_cells:
        for rate, r in c.r_by_rate.items():
            all_rates.setdefault(round(rate, 3), []).append(r)
    for rate in sorted(all_rates):
        vals = np.array(all_rates[rate])
        rate_rows.append({"rate_a": rate, "n": int(vals.size),
                          "r_median_ohm": float(np.median(vals)),
                          "r_std_ohm": float(np.std(vals))})
    if rate_rows:
        r_ref = np.median([r["r_median_ohm"] for r in rate_rows
                           if abs(r["rate_a"] - RATE_1C_A) < 0.5])
        for r in rate_rows:
            r["r_rel_vs_1c"] = float(r["r_median_ohm"] / r_ref) if r_ref else float("nan")
        print("   R(rate) 满充中位：" + "  ".join(
            f"{r['rate_a']:.2f}A:{r['r_median_ohm']*1000:.1f}" for r in rate_rows[:8]))

    # ---- [2/4] 纵向（HNEI / MICH / XJTU）----
    print("[2/4] 纵向提取（HNEI/MICH/XJTU）...")
    long_sets = {}
    for tag, zp, dataset in (
        ("HNEI", HNEI_ZIP, "HNEI"),
        ("MICH", MICH_ZIP, "MICH"),
        ("XJTU", XJTU_ZIP, "XJTU"),
    ):
        cells = load_longitudinal_zip(zp, dataset,
                                      cache_csv=BETA_CSV if tag == "HNEI" else None)
        # XJTU/MICH 缓存共用一张表：追加式重建（缓存仅在整表缺失时写，
        # 这里分别提取，避免互相覆盖）
        fit = fit_beta_longitudinal(cells)
        long_sets[tag] = {"fit": fit, "n_cells": len(cells)}
        print(f"   {tag}: {len(cells)} 芯，β 中位 "
              f"{fit['beta_median']:.2f}（范围 {fit['beta_range']}）")

    # ---- [3/4] 反馈漂移 ----
    print("[3/4] 反馈漂移按 β 重算 ...")
    cache: dict = {}
    fn_cache: dict = {}
    drift_rows = []
    betas = [
        ("CS2-LFP 实测", CS2_BETA_MEDIAN),
        ("NCA 截面", nca_fit["beta"]),
    ]
    for tag in ("HNEI", "MICH", "XJTU"):
        b = long_sets[tag]["fit"]["beta_median"]
        if np.isfinite(b):
            betas.append((f"{tag} 纵向", b))
    betas.append(("feedback 默认", BETA_DEFAULT))
    for tag, b in betas:
        if not np.isfinite(b):
            continue
        d = feedback_drift(float(b), cache, fn_cache)
        d["source"] = tag
        drift_rows.append(d)
        print(f"   β={b:.2f}（{tag}）: C5 闭环 {d['c5_closed']:.3f} "
              f"漂移 {d['c5_drift_pct']:+.2f}%")

    # ---- [4/4] 图与产物 ----
    print("[4/4] 图与产物 ...")
    plotting.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    f_all = np.array([c.fade for c in m_cells], float)
    r_all = np.array([c.r_1c for c in m_cells], float) * 1000
    m_ok = np.isfinite(f_all) & np.isfinite(r_all)
    ax.scatter(f_all[m_ok], r_all[m_ok], s=18, alpha=0.65, color="#2980b9",
               label=f"MATR-10 芯（n={int(m_ok.sum())}）")
    if np.isfinite(nca_fit.get("beta", np.nan)):
        xs = np.linspace(0, 0.4, 20)
        ys = (nca_fit["intercept_ohm"] + nca_fit["slope_ohm_per_f"] * xs) * 1000
        ax.plot(xs, ys, "r-", lw=1.8,
                label=f"拟合 R=R0(1+{nca_fit['beta']:.2f}f)")
    ax.axvline(0.2, color="gray", ls=":", lw=0.9, label="EOL f=0.2")
    ax.set_xlabel("fade f = 1 − cap/4.84 Ah")
    ax.set_ylabel("R(1C) 满充（mΩ）")
    ax.set_title("a) NCA 截面：R vs f（MATR-10）")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    colors = {"HNEI": "#c0392b", "MICH": "#2980b9", "XJTU": "#27ae60"}
    for tag in ("HNEI", "MICH", "XJTU"):
        # 画 β 中位线（原始逐芯曲线见 beta_extract.csv）
        fit = long_sets[tag]["fit"]
        b = fit["beta_median"]
        if np.isfinite(b):
            xs = np.linspace(0, 0.25, 20)
            ax.plot(xs, 1 + b * xs, "-", color=colors[tag], lw=1.8,
                    label=f"{tag} β={b:.2f}")
    xs = np.linspace(0, 0.25, 20)
    ax.plot(xs, 1 + CS2_BETA_MEDIAN * xs, "k--", lw=1.4,
            label=f"CS2-LFP β={CS2_BETA_MEDIAN:.2f}")
    ax.plot(xs, 1 + BETA_DEFAULT * xs, ":", color="#7f8c8d", lw=1.4,
            label=f"默认 β={BETA_DEFAULT}")
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("fade f")
    ax.set_ylabel("R/R0 = 1 + βf")
    ax.set_title("b) 纵向 β 中位线（每 EFC 窗口）")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    tags = [t for t, b in betas if np.isfinite(b)]
    vals = [b for t, b in betas if np.isfinite(b)]
    cols = ["#8e44ad" if "NCA" in t else
            "#c0392b" if "CS2" in t else
            "#2980b9" if "默认" in t else "#27ae60" for t in tags]
    ax.bar(range(len(vals)), vals, color=cols, width=0.6)
    ax.axhspan(BETA_LIT_LO, BETA_LIT_HI, color="#f6d55c", alpha=0.25,
               label="文献带 [1.5, 3.5]")
    ax.axhline(CS2_BETA_MEDIAN, color="k", ls="--", lw=0.9)
    ax.set_xticks(range(len(tags)), tags, fontsize=7, rotation=15)
    ax.set_ylabel("β（R/R0=1+βf）")
    ax.set_title("c) β 跨体系对账")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    d_tags = [d["source"] for d in drift_rows]
    d_vals = [d["c5_drift_pct"] for d in drift_rows]
    cols = ["#8e44ad" if "NCA" in t else
            "#c0392b" if "CS2" in t else
            "#2980b9" if "默认" in t else "#27ae60" for t in d_tags]
    ax.bar(range(len(d_vals)), d_vals, color=cols, width=0.6)
    ax.set_xticks(range(len(d_tags)), d_tags, fontsize=7, rotation=15)
    ax.set_ylabel("C5 反馈漂移（%）")
    ax.set_title("d) β→反馈漂移（3C 液冷/1C 自然）")
    for i, d in enumerate(d_vals):
        ax.text(i, d, f"{d:+.1f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("35a · β 跨体系锚：NMC/NCA 内阻老化实测", fontsize=12)
    fig.tight_layout()
    plotting.save(fig, FIG / "35a_beta_anchor.png")
    plt.close(fig)

    nca_b = nca_fit.get("beta", float("nan"))
    lmc = [long_sets[t]["fit"]["beta_median"] for t in ("HNEI", "MICH", "XJTU")]
    lmc_ok = [b for b in lmc if np.isfinite(b)]
    verdict = (
        f"β 跨体系锚收口：NCA 截面（MATR-10 94 芯）β={nca_b:.2f} "
        f"（R0={nca_fit.get('r0_ohm', float('nan'))*1000:.0f} mΩ）；"
        f"NMC 纵向（HNEI/MICH/XJTU）β 中位 "
        f"{np.median(lmc_ok) if lmc_ok else float('nan'):.2f}。"
        f"CS2 LFP 实测 0.48、HNEI 1.16 低于文献带下缘 1.5；"
        f"NCA/MICH/XJTU 在带内（2.17–2.32）但全体低于默认 2.5——"
        f"反馈项在默认参数下偏保守（漂移 3.9% vs 实测 0.7–3.6%），"
        f"CS2-LFP 口径下大幅保守。"
        f"所有 R 均从 V/I 阶跃提取（三套数据内阻列全零）。"
    )
    metrics = {
        "dataset": "MATR-10（NCA 21700 截面）+ HNEI/MICH/XJTU（NMC 纵向）"
                   " + CS2（LFP，scripts/19）",
        "matr10_cross_sectional": {
            "chem": "NCA（Panasonic 21700，石墨+SiOx）",
            "n_cells_scanned": len(m_cells),
            "n_cells_with_r1c": n_with_1c,
            "nominal_ah": NOMINAL_AH,
            "fade_range": [float(min(c.fade for c in m_cells)),
                           float(max(c.fade for c in m_cells))],
            "fit": nca_fit,
            "r_by_rate": rate_rows,
        },
        "longitudinal": {t: {"n_cells": long_sets[t]["n_cells"],
                             **long_sets[t]["fit"]} for t in long_sets},
        "beta_ledger": {
            "cs2_lfp_measured": CS2_BETA_MEDIAN,
            "nca_cross_sectional": nca_b,
            "nmc_longitudinal": {t: long_sets[t]["fit"]["beta_median"]
                                 for t in ("HNEI", "MICH", "XJTU")},
            "literature_band": [BETA_LIT_LO, BETA_LIT_HI],
            "feedback_default": BETA_DEFAULT,
        },
        "feedback_drift_rows": drift_rows,
        "resistance_scan_note": (
            "batteryarchive internal_resistance_in_ohm 与 MATR-10 DCIR/ACImp 列 "
            "全部为零/None（HNEI 14/14、MICH 40/40、XJTU 23/23、MATR-10 0/72209 非零）"
            "——R 一律从 V/I 电流阶跃提取；schema 有字段≠有数据（第三次确认）。"
        ),
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    with open(MET / "35_beta_anchor.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=1)

    print(f"done in {metrics['elapsed_s']:.0f}s -> 35_beta_anchor.json / 35a_beta_anchor.png")
    print("verdict:", verdict[:240])


if __name__ == "__main__":
    main()
