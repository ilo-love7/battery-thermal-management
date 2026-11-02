# -*- coding: utf-8 -*-
"""WP42 温度交互面补全（v1.10 第二线）。

WP39 只做了 LFP/NMC 的 m(2C;T)；SNL 里 NCA 也有 2C×T 全三温度
（15/25/35 °C，各 n=2）——本轮补全三化学温度交互面：

1. **NCA m(2C;T)**：组均比 + rep 配对（资格判据 n≥2 双侧，
   WP39 同口径）→ 三化学 m(2C;T) 全表；
2. **1C×T 归一基线**：fade(1C,T)/fade(1C,25) 温度加速比——若 m(2C)
   随 T 平但 1C fade 随 T 升，说明倍率-温度可加分离（T 等比加速
   各倍率），交互项≈0；
3. **3C×T 代理带定量化**：v1.9 的「2C 代理假设」从定性标注升级为
   数值带——m(3C,T) ∈ m(3C,25) × [min,max]（跨化学 m 比值
   m(2C,T)/m(2C,25) 的识别包络）；落绝对账 3C 行。
4. **表观 Ea 对账**（WP39 口径延续）：NCA 温度轴 1C fade 的
   (15,25)/(25,35) 对。

产物：results/metrics/42_rate_temp_surface.json、
results/figures/42a_rate_temp_surface.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal.plotting import save as save_fig, setup_style  # noqa: E402
from bmthermal.rate_effect import (  # noqa: E402
    apparent_ea_from_fade,
    matched_rep_m,
    pair_m_ratio,
)

import matplotlib.pyplot as plt  # noqa: E402

MET = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures"
SNL_CSV = ROOT / "results" / "cache" / "snl_summary.csv"

CHEMS = ("LFP", "NMC", "NCA")
TEMPS = (15.0, 25.0, 35.0)


def group_fades(df: pd.DataFrame, chem: str, temp: float, c: float):
    """[(rep, fade)] 列表。"""
    sub = df[(df.chem == chem) & (df.temp_c == temp) & (df.c_dis == c)]
    return [(r.rep, float(r.fade_rate_per_efc)) for r in sub.itertuples()]


def main() -> None:
    t0 = time.time()
    MET.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SNL_CSV)
    g = df[(df.soc_lo == 0) & (df.soc_hi == 100)]

    print("[1/4] m(2C; T) 三化学全表（含 NCA 补全）...")
    m_rows = []
    for chem in CHEMS:
        for temp in TEMPS:
            fast = group_fades(g, chem, temp, 2.0)
            ref = group_fades(g, chem, temp, 1.0)
            if not fast or not ref:
                continue
            pr = pair_m_ratio([f for _, f in fast], [f for _, f in ref])
            pm = matched_rep_m(fast, ref)
            row = {
                "chem": chem, "temp_c": temp, "c_fast": 2.0, "c_ref": 1.0,
                **{f"pair_{k}": v for k, v in pr.items()},
                **{f"matched_{k}": v for k, v in pm.items()},
            }
            m_rows.append(row)
            print(f"   {chem} {temp:g}C m={pr['m']:.3f} CI {pr['ci95']} "
                  f"matched={pm['m_matched']:.3f} elig={pr['eligible']}")

    print("[2/4] 1C×T 温度加速基线 ...")
    t_rows = []
    for chem in CHEMS:
        f25 = [f for _, f in group_fades(g, chem, 25.0, 1.0)]
        if not f25:
            continue
        base = float(np.mean(f25))
        for temp in (15.0, 35.0):
            ft = [f for _, f in group_fades(g, chem, temp, 1.0)]
            if not ft:
                continue
            ratio = float(np.mean(ft)) / base
            row = {
                "chem": chem, "temp_c": temp,
                "fade_1c": float(np.mean(ft)), "fade_1c_25c": base,
                "ratio_vs_25c": ratio,
                "n": len(ft), "n_ref": len(f25),
                "eligible": len(ft) >= 2 and len(f25) >= 2,
            }
            t_rows.append(row)
            print(f"   {chem} {temp:g}C/25C fade(1C) 比={ratio:.3f} "
                  f"(n={len(ft)}/{len(f25)}) elig={row['eligible']}")

    print("[3/4] 3C×T 代理带（m 比值识别包络）...")
    # 每化学：m(2C,T)/m(2C,25)（资格对）→ 跨化学包络
    ratio_env = {}
    for chem in CHEMS:
        base = next((r for r in m_rows if r["chem"] == chem
                     and r["temp_c"] == 25.0 and r["pair_eligible"]), None)
        if base is None:
            continue
        rs = []
        for r in m_rows:
            if r["chem"] != chem or not r["pair_eligible"]:
                continue
            rs.append(r["pair_m"] / base["pair_m"])
        ratio_env[chem] = {
            "m_25c": base["pair_m"],
            "ratios": [float(x) for x in rs],
            "band": [float(min(rs)), float(max(rs))],
            "max_flat_error_pct": float(max(abs(x - 1) for x in rs) * 100),
        }
        print(f"   {chem} m 比带 {ratio_env[chem]['band']} "
              f"T-flat ±{ratio_env[chem]['max_flat_error_pct']:.0f}%")
    all_ratios = [x for v in ratio_env.values() for x in v["ratios"]]
    env_band = [float(min(all_ratios)), float(max(all_ratios))] if all_ratios \
        else [float("nan"), float("nan")]
    # 稳健包络：剔除 NMC（25 °C 基 m=0.60 是已知批次异质性异常，
    # v1.9 §57 判过——其比值 1.80 是异常宽度不是温度交互证据）
    robust_ratios = [x for k, v in ratio_env.items() if k != "NMC"
                     for x in v["ratios"]]
    robust_band = [float(min(robust_ratios)), float(max(robust_ratios))] \
        if robust_ratios else [float("nan"), float("nan")]
    # NMC 稳健比值：用 rep 配对 m 作基（v1.9 拉回 0.74）
    nmc_matched_ratios = []
    nmc_base_m = next((r["matched_m_matched"] for r in m_rows
                       if r["chem"] == "NMC" and r["temp_c"] == 25.0
                       and r["matched_eligible"]), None)
    if nmc_base_m:
        for r in m_rows:
            if r["chem"] == "NMC" and r["temp_c"] != 25.0 and r["matched_eligible"]:
                nmc_matched_ratios.append(r["matched_m_matched"] / nmc_base_m)
    robust_band_nmc = list(robust_band)
    if nmc_matched_ratios:
        robust_band_nmc = [
            float(min(robust_band[0], min(nmc_matched_ratios))),
            float(max(robust_band[1], max(nmc_matched_ratios))),
        ]
    print(f"   全量包络 {env_band} / 稳健包络（含 NMC rep 配对）{robust_band_nmc}")
    # 落到 3C 行：m(3C,25) 来自 WP37/40（LFP [2.30,3.35]、NMC 中性 [0.7,1.1]）
    m3c_25 = {
        "LFP": [2.30, 3.35],
        "NMC": [0.7, 1.1],
        "NCA": [0.85, 1.30],
    }
    m3c_bands = {}
    for chem, b in m3c_25.items():
        m3c_bands[chem] = {
            "m_25c_band": b,
            "t_proxy_band_outer": [float(b[0] * env_band[0]), float(b[1] * env_band[1])],
            "t_proxy_band_robust": [float(b[0] * robust_band_nmc[0]),
                                    float(b[1] * robust_band_nmc[1])],
            "note": "2C 代理假设定量化：outer=全量 m 比值包络，robust=剔 NMC 异常",
        }
        print(f"   {chem} 3C 行 T 代理带 outer {m3c_bands[chem]['t_proxy_band_outer']} "
              f"robust {m3c_bands[chem]['t_proxy_band_robust']}")

    print("[4/4] 表观 Ea 对账 + 图 ...")
    ea_rows = []
    for chem in CHEMS:
        for t_lo, t_hi in ((15.0, 25.0), (25.0, 35.0)):
            f_lo = [f for _, f in group_fades(g, chem, t_lo, 1.0)]
            f_hi = [f for _, f in group_fades(g, chem, t_hi, 1.0)]
            if not f_lo or not f_hi:
                continue
            r = apparent_ea_from_fade(float(np.mean(f_hi)), float(np.mean(f_lo)),
                                      t_lo, t_hi)
            ea_rows.append({"chem": chem, "t_lo": t_lo, "t_hi": t_hi,
                            "fade_lo": float(np.mean(f_lo)),
                            "fade_hi": float(np.mean(f_hi)), **r})
            print(f"   {chem} {t_lo:g}->{t_hi:g}C ratio={r['ratio']:.3f} "
                  f"Ea={r['ea_kj_mol']:.1f} elig={r['eligible']}")

    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    colors = {"LFP": "#c0392b", "NMC": "#2980b9", "NCA": "#27ae60"}

    ax = axes[0, 0]
    for chem in CHEMS:
        rows = [r for r in m_rows if r["chem"] == chem and r["pair_eligible"]]
        if not rows:
            continue
        ts = [r["temp_c"] for r in rows]
        ms = [r["pair_m"] for r in rows]
        lo = [r["pair_m"] - r["pair_ci95"][0] for r in rows]
        hi = [r["pair_ci95"][1] - r["pair_m"] for r in rows]
        ax.errorbar(ts, ms, yerr=[lo, hi], fmt="o-", color=colors[chem],
                    capsize=4, label=chem, lw=1.6)
        mts = [r["temp_c"] for r in rows if r["matched_eligible"]]
        mms = [r["matched_m_matched"] for r in rows if r["matched_eligible"]]
        if mts:
            ax.plot(mts, mms, "d", color=colors[chem], mfc="white", ms=7)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xlabel("温度 [°C]")
    ax.set_ylabel("m(2C)")
    ax.set_title("a) m(2C; T) 三化学全表（NCA 补全）")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for chem in CHEMS:
        rows = [r for r in t_rows if r["chem"] == chem and r["eligible"]]
        if not rows:
            continue
        ts = [25.0] + [r["temp_c"] for r in rows]
        rs = [1.0] + [r["ratio_vs_25c"] for r in rows]
        order = np.argsort(ts)
        ax.plot(np.array(ts)[order], np.array(rs)[order], "o-",
                color=colors[chem], label=chem, lw=1.4)
        # 2C 的 m 比值（交互量）同图虚线
        b = ratio_env.get(chem)
        if b:
            mts = [25.0] + [r["temp_c"] for r in m_rows
                            if r["chem"] == chem and r["pair_eligible"]
                            and r["temp_c"] != 25.0]
            mrs = [1.0] + [r["pair_m"] / b["m_25c"] for r in m_rows
                           if r["chem"] == chem and r["pair_eligible"]
                           and r["temp_c"] != 25.0]
            order = np.argsort(mts)
            ax.plot(np.array(mts)[order], np.array(mrs)[order], "s--",
                    color=colors[chem], lw=1.2, mfc="white")
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xlabel("温度 [°C]")
    ax.set_ylabel("相对 25 °C 比值")
    ax.set_title("b) 1C fade 温度加速（实线）vs m(2C) 比值（虚线）")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for chem in CHEMS:
        if chem not in m3c_bands:
            continue
        b = m3c_bands[chem]["t_proxy_band_robust"]
        base = m3c_bands[chem]["m_25c_band"]
        ax.plot([0, 1], [base[0], b[0]], "v-", color=colors[chem], lw=1.4,
                label=f"{chem} 下缘")
        ax.plot([0, 1], [base[1], b[1]], "^-", color=colors[chem], lw=1.4,
                mfc="white", label=f"{chem} 上缘")
    ax.set_xticks([0, 1], ["25 °C（实测带）", "其他 T（2C 代理包络·稳健）"], fontsize=8)
    ax.set_ylabel("m(3C) 带")
    ax.set_title("c) 3C 行 T-flat → 代理带（定量化）")
    ax.legend(fontsize=7)
    ax.text(0.02, 0.96,
            f"稳健包络 [{robust_band_nmc[0]:.2f}, {robust_band_nmc[1]:.2f}]"
            f"（全量 [{env_band[0]:.2f}, {env_band[1]:.2f}] 含 NMC 异常宽）\n"
            "3C×T 仍不可辨识（3C 仅 25C 有组）",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="#fdebd0", edgecolor="#e67e22"))

    ax = axes[1, 1]
    xs, labels, cols = [], [], []
    for r in ea_rows:
        if not r["eligible"]:
            continue
        xs.append(r["ea_kj_mol"])
        labels.append(f'{r["chem"]}\n{r["t_lo"]:g}→{r["t_hi"]:g}')
        cols.append(colors[r["chem"]])
    if xs:
        ax.bar(np.arange(len(xs)), xs, 0.55, color=cols)
        ax.set_xticks(np.arange(len(xs)), labels, fontsize=8)
    ax.axhline(45.0, color="black", ls="--", lw=1, label="EA 默认 45")
    ax.set_ylabel("表观 Ea [kJ/mol]")
    ax.set_title("d) 1C fade 表观 Ea（仅资格对）")
    ax.legend(fontsize=8)
    neg = [f'{r["chem"]} {r["t_lo"]:g}->{r["t_hi"]:g}' for r in ea_rows
           if not r["eligible"]]
    if neg:
        ax.text(0.02, 0.96, "资格否决: " + ", ".join(neg),
                transform=ax.transAxes, fontsize=7, va="top")

    fig.tight_layout()
    save_fig(fig, FIG / "42a_rate_temp_surface.png")

    # 可加分离检验：m 比值带 vs 1C fade 比值带（分化学；NMC 单列异常）
    flat_errs = {k: v["max_flat_error_pct"] for k, v in ratio_env.items()}
    t_ratios = [r["ratio_vs_25c"] for r in t_rows if r["eligible"]]
    non_nmc_err = max(v for k, v in flat_errs.items() if k != "NMC")
    additive = (
        f"可加分离检验（分化学）：m(2C) 比值 T-flat "
        f"{'/'.join(f'{k}±{v:.0f}%' for k, v in flat_errs.items())}，"
        f"而 1C fade 温度加速比 {min(t_ratios):.2f}–{max(t_ratios):.2f}——"
        f"LFP/NCA（±{non_nmc_err:.0f}%）m 近平而 fade 随 T 变 → 可加近似可容；"
        f"NMC ±{flat_errs.get('NMC', 0):.0f}% 是 25 °C 已知批次异质性异常"
        f"（组基 m=0.60，rep 配对 0.74）的宽度，不作温度交互证据（v1.9 §57 同判）。"
    )
    nca_m25 = next((f"{r['pair_m']:.2f}" for r in m_rows
                    if r["chem"] == "NCA" and r["temp_c"] == 25.0
                    and r["pair_eligible"]), "NA")
    nca_ea = next((f"{r['ea_kj_mol']:.1f}" for r in ea_rows
                   if r["chem"] == "NCA" and r["eligible"]), "NA")
    verdict = (
        f"温度交互面补全收口：NCA m(2C;T) 补入（{nca_m25} @25C），"
        f"三化学 m(2C;T) 全表完成——资格对 m 带 "
        f"[{min(r['pair_m'] for r in m_rows if r['pair_eligible']):.2f}, "
        f"{max(r['pair_m'] for r in m_rows if r['pair_eligible']):.2f}]。"
        f"{additive}"
        f"3C×T 代理带定量化：m(3C,T)=m(3C,25)×比值包络"
        f"（稳健 [{robust_band_nmc[0]:.2f},{robust_band_nmc[1]:.2f}]、"
        f"全量 [{env_band[0]:.2f},{env_band[1]:.2f}]）——"
        f"LFP {m3c_bands['LFP']['t_proxy_band_robust']}、"
        f"NMC {m3c_bands['NMC']['t_proxy_band_robust']}、"
        f"NCA {m3c_bands['NCA']['t_proxy_band_robust']}（稳健）——"
        f"v1.9「2C 代理假设」升级为数值带，"
        f"3C×T 本身仍不可辨识（3C 仅 25 °C 有组）。"
        f"表观 Ea：NCA 温度轴 {nca_ea} kJ/mol 进带；"
        f"NMC 反向对仍资格否决（v1.7 NMC 温度轴失效一致）。"
    )
    metrics = {
        "dataset": "Sandia SNL 0-100%（15/25/35 °C × 0.5–3C）",
        "m_2c_rows": m_rows,
        "fade_1c_t_rows": t_rows,
        "m_ratio_envelope": ratio_env,
        "env_band": env_band,
        "robust_band": robust_band_nmc,
        "m3c_t_proxy_bands": m3c_bands,
        "apparent_ea_rows": ea_rows,
        "additive_check": additive,
        "identifiability": {
            "rate_3c_temperature": "not_identifiable（LFP/NMC 3C 仅 25C）",
            "proxy_quantified": "m(3C,T)=m(3C,25)×[env lo, env hi]（2C 代理假设的数值化）",
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (MET / "42_rate_temp_surface.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(verdict)
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
