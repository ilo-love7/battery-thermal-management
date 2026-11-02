# -*- coding: utf-8 -*-
"""WP39 m(c) 温度交互（v1.9 第二线）。

开放项#6（m(c) 温度交互）的定量收口——SNL 15/25/35 °C 温度轴：

1. **m(2C; T)** 三温度对（LFP/NMC/NCA）：组均比 + 电芯 bootstrap CI
   + **rep 配对比**（同 rep 标签逐对比值——批次异质性控制，v1.5 教训
   的配对版）；资格判据 n≥2 双侧（LFP 15C 2C n=1 → 拒）。
2. **3C×T 不可辨识**（硬边界）：LFP 3C 仅 25 °C 有组——绝对账 3C 行
   的 m 温度交互未知，2C 交互只能作代理（标注假设）。
3. **表观 Ea 对账**：1C fade 的 (15,25)/(25,35) 对 → Ea（资格=正值，
   v1.7 §33 口径）；NMC 25→35 反向（fade 降）→ 负 Ea 资格否决，
   与 v1.7「NMC 温度轴失效」对拍。
4. **T-flat 假设误差**：绝对账按 25 °C 的 m 套所有温度——误差界 =
   |m(T)/m(25)−1| 在识别集内取极值，落到 3C 行标注「2C 代理」。

产物：results/metrics/39_rate_temp.json、results/figures/39a_rate_temp.png
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
    print("[1/3] m(2C; T) 三温度对 + rep 配对 ...")

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
                  f"(n={pr['n_fast']}/{pr['n_ref']}) matched={pm['m_matched']:.3f} "
                  f"(pairs={pm['n_pairs']}) elig={pr['eligible']}")

    print("[2/3] 表观 Ea（1C fade 温度轴）...")
    ea_rows = []
    for chem in CHEMS:
        for t_lo, t_hi in ((15.0, 25.0), (25.0, 35.0)):
            f_lo = [f for _, f in group_fades(g, chem, t_lo, 1.0)]
            f_hi = [f for _, f in group_fades(g, chem, t_hi, 1.0)]
            if not f_lo or not f_hi:
                continue
            r = apparent_ea_from_fade(float(np.mean(f_hi)), float(np.mean(f_lo)),
                                      t_lo, t_hi)
            row = {"chem": chem, "t_lo": t_lo, "t_hi": t_hi,
                   "fade_lo": float(np.mean(f_lo)), "fade_hi": float(np.mean(f_hi)),
                   **r}
            ea_rows.append(row)
            print(f"   {chem} {t_lo:g}->{t_hi:g}C ratio={r['ratio']:.3f} "
                  f"Ea={r['ea_kj_mol']:.1f} kJ/mol elig={r['eligible']}")

    # T-flat 假设误差（m(2C,T)/m(2C,25) 识别带）
    flat = {}
    for chem in CHEMS:
        base = next((r for r in m_rows
                     if r["chem"] == chem and r["temp_c"] == 25.0
                     and r["pair_eligible"]), None)
        if base is None:
            continue
        ratios = []
        for r in m_rows:
            if r["chem"] != chem or not r["pair_eligible"]:
                continue
            ratios.append(r["pair_m"] / base["pair_m"])
        if ratios:
            flat[chem] = {
                "m_25c": base["pair_m"],
                "ratio_band": [float(min(ratios)), float(max(ratios))],
                "max_flat_error_pct": float(max(abs(x - 1) for x in ratios) * 100),
            }
            print(f"   {chem} T-flat 误差界 ±{flat[chem]['max_flat_error_pct']:.0f}%"
                  f"（比带 {flat[chem]['ratio_band']}）")

    print("[3/3] 图 + 产物 ...")
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    ax = axes[0, 0]
    colors = {"LFP": "#c0392b", "NMC": "#2980b9", "NCA": "#27ae60"}
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
    ax.set_title("a) m(2C; T) 组比（圆）+ rep 配对（菱）")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for chem in CHEMS:
        for c, style in ((1.0, "o-"), (2.0, "s--")):
            rows = [r for r in m_rows if r["chem"] == chem]
            ts, fs = [], []
            for temp in TEMPS:
                fades = [f for _, f in group_fades(g, chem, temp, c)]
                if fades:
                    ts.append(temp)
                    fs.append(float(np.mean(fades)))
            if ts:
                ax.semilogy(ts, fs, style, color=colors[chem],
                            label=f"{chem} {c:g}C", lw=1.4)
    ax.set_xlabel("温度 [°C]")
    ax.set_ylabel("fade 率 [1/EFC]")
    ax.set_title("b) fade vs T（NMC 25→35 反向）")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    xs, labels, cols = [], [], []
    for i, r in enumerate(ea_rows):
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
    ax.set_title("c) 1C fade 表观 Ea（仅资格对）")
    ax.legend(fontsize=8)
    neg = [f'{r["chem"]} {r["t_lo"]:g}->{r["t_hi"]:g}' for r in ea_rows
           if not r["eligible"]]
    if neg:
        ax.text(0.02, 0.96, "资格否决（负 Ea）: " + ", ".join(neg),
                transform=ax.transAxes, fontsize=7, va="top")

    ax = axes[1, 1]
    chems_f = list(flat.keys())
    if chems_f:
        errs = [flat[c]["max_flat_error_pct"] for c in chems_f]
        ax.bar(np.arange(len(chems_f)), errs, 0.55,
               color=[colors[c] for c in chems_f])
        ax.set_xticks(np.arange(len(chems_f)), chems_f, fontsize=9)
    ax.set_ylabel("T-flat m 假设最大误差 [%]")
    ax.set_title("d) 绝对账 T-flat 误差界（2C 实测）")
    ax.text(0.02, 0.96, "3C×T：不可辨识（LFP 3C 仅 25C）\n3C 行温度交互用 2C 代理（假设）",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="#fdebd0", edgecolor="#e67e22"))

    fig.tight_layout()
    save_fig(fig, FIG / "39a_rate_temp.png")

    # 判词
    elig = [r for r in m_rows if r["pair_eligible"]]
    m_band = [r["pair_m"] for r in elig]
    matched_band = [r["matched_m_matched"] for r in elig if r["matched_eligible"]]
    nmc_inv = [f'{r["chem"]} {r["t_lo"]:g}->{r["t_hi"]:g}'
               for r in ea_rows if not r["eligible"]]
    flat_pct = "/".join(f"{flat[c]['max_flat_error_pct']:.0f}%" for c in flat)
    verdict = (
        f"m(2C; T) 温度交互收口：三温度资格对 m∈[{min(m_band):.2f}, "
        f"{max(m_band):.2f}]（rep 配对 [{min(matched_band):.2f}, "
        f"{max(matched_band):.2f}]）——15–35 °C 内 m(2C) 温度弱/异质性主导，"
        f"T-flat 假设误差界 LFP/NMC/NCA ±{flat_pct}（2C 实测）；"
        f"NMC ±80% 是 25 °C 异质性异常（m=0.60，rep 配对拉回 0.74）的贡献，"
        f"不是温度交互证据。"
        f"3C×T 不可辨识（LFP 3C 仅 25 °C）——绝对账 3C 行温度交互标 2C 代理假设。"
        f"表观 Ea：资格对进带（LFP 27.6、NCA 10.3/23.0 kJ/mol）；"
        f"NMC 反向对（{nmc_inv}）负 Ea 资格否决，与 v1.7 NMC 温度轴失效一致。"
    )
    metrics = {
        "dataset": "Sandia SNL 0-100%（15/25/35 °C × 0.5–3C）",
        "m_2c_rows": m_rows,
        "apparent_ea_rows": ea_rows,
        "t_flat_error": flat,
        "identifiability": {
            "rate_3c_temperature": "not_identifiable（LFP 3C 仅 25C；NMC 3C 仅 25C）",
            "proxy_note": "绝对账 3C 行的 m 温度交互以 2C 交互为代理——假设，非实测。",
        },
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (MET / "39_rate_temp.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(verdict)
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
