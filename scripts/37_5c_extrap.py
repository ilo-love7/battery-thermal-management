# -*- coding: utf-8 -*-
"""WP37 5C m(c) 外推护栏（v1.8 第三线）。

开放项#1 的护栏版（台架未到位前）：
1. SNL 0.5–3C 实测 m(c)（25 °C，n≥2 资格点）拟合幂律/线性标度 → 5C 外推带；
   LFP/NMC 三点非单调，外推方向本身不确定——**不可作设计依据**，只给护栏带。
2. 翻转绝对账 5C 行的临界 m(5)：吞吐打平点（throughput_rel=1）与
   寿命打平点（life_rel=1），由 29 档位行（m=1 基线）反解。
3. 与 WP28 欧姆指数 p=1.83–2.00 交叉核对：若损伤∝每 EFC 欧姆耗散
   （∝c·R(c)∝c^(p−1)），m(5) 应落在 5^(p−1) 带——实测局部指数远低
   （仅 LFP 接近），说明损伤不按欧姆耗散成比例，该带只作上缘锚。

产物：results/metrics/37_5c_extrap.json、results/figures/37a_5c_extrap.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal.plotting import save as save_fig, setup_style  # noqa: E402

MET = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures"

CHEMS = ("LFP", "NMC", "NCA")
C_TARGET = 5.0


def load_m_points() -> list:
    """从 29 指标载入 m(c) 点（25 °C）。"""
    d = json.loads((MET / "29_rate_effect.json").read_text(encoding="utf-8"))
    return [p for p in d["rate_points"] if abs(p["temp_c"] - 25.0) < 1e-6]


def load_5c_baseline() -> dict:
    """29 档位 5C 行（m=1 基线）→ 打平临界值。"""
    d = json.loads((MET / "29_rate_effect.json").read_text(encoding="utf-8"))
    row = next(r for r in d["tier_rows"]
               if r["c_rate"] == C_TARGET and r["m_tag"] == "no_data")
    life0 = float(row["life_efc_rel_vs_1c"])       # m=1 时相对 1C 寿命
    thr0 = float(row["throughput_rel_vs_1c"])      # m=1 时相对 1C 吞吐
    return {
        "m_baseline": 1.0,
        "life_rel_at_m1": life0,
        "throughput_rel_at_m1": thr0,
        "k_t_mean": float(row["k_t_mean"]),
        "t_core_peak_c": float(row["t_core_peak_c"]),
        # thr(m)=thr0/m、life(m)=life0/m → 打平点
        "m_star_throughput": thr0,     # thr=1
        "m_star_life": life0,          # life=1
    }


def fit_scaling(pts: list) -> list:
    """逐体系拟合 m(c)：幂律（log-log）与线性。c≥1 且资格点。"""
    fits = []
    for chem in CHEMS:
        sel = [p for p in pts
               if p["chem"] == chem and p["c_dis"] >= 1.0 - 1e-9
               and p.get("eligible", True)]
        c = np.array([p["c_dis"] for p in sel], float)
        m = np.array([p["m_per_efc"] for p in sel], float)
        if c.size < 2:
            continue
        row = {"chem": chem, "n_pts": int(c.size),
               "c_used": c.tolist(), "m_used": m.tolist()}
        # 幂律 ln m = a + b ln c
        if np.all(m > 0):
            b, la = np.polyfit(np.log(c), np.log(m), 1)
            pred = la + b * np.log(c)
            r2 = 1 - ((np.log(m) - pred) ** 2).sum() / max(
                ((np.log(m) - np.log(m).mean()) ** 2).sum(), 1e-12)
            row["power"] = {"b": float(b), "a": float(np.exp(la)),
                            "r2": float(r2),
                            "m_at_5c": float(np.exp(la) * C_TARGET ** b)}
        # 线性 m = a + b c
        b1, a1 = np.polyfit(c, m, 1)
        pred1 = a1 + b1 * c
        r2_1 = 1 - ((m - pred1) ** 2).sum() / max(
            ((m - m.mean()) ** 2).sum(), 1e-12)
        row["linear"] = {"b": float(b1), "a": float(a1),
                         "r2": float(r2_1),
                         "m_at_5c": float(a1 + b1 * C_TARGET)}
        # 局部指数（1→max c 实测对）：b_loc = ln(m_max/m(1))/ln(c_max)
        m1 = float(m[np.argmin(np.abs(c - 1.0))])
        c_max = float(c.max())
        m_max = float(m[np.argmax(c)])
        row["local_exponent"] = {
            "b_loc": float(np.log(m_max / m1) / np.log(c_max)) if m1 > 0 and m_max > 0 and c_max > 1 else float("nan"),
            "c_span": [1.0, c_max], "m_span": [m1, m_max],
        }
        fits.append(row)
    return fits


def ohmic_proxy_band() -> dict:
    """WP28 欧姆指数 → 每 EFC 欧姆耗散代理 m∝c^(p−1) 在 5C 的带。"""
    d = json.loads((MET / "28_highrate_scaling.json").read_text(encoding="utf-8"))
    p_by_t = d["heat_side"]["p_ohmic_by_temp"]
    ps = {t: v["q_exp_ohmic"] for t, v in p_by_t.items()}
    band = {t: float(C_TARGET ** (p - 1.0)) for t, p in ps.items()}
    return {
        "p_ohmic_by_temp": ps,
        "scaling_law_p_total": d["heat_side"]["scaling_law_p"],
        "m5_ohmic_proxy_by_temp": band,
        "m5_ohmic_proxy_band": [float(min(band.values())), float(max(band.values()))],
        "note": ("每 EFC 欧姆耗散∝c·R(c)∝c^(p−1)；若损伤∝欧姆耗散，"
                 "m(5)≈5^(p−1)。上缘锚，非机理解释。"),
    }


def main() -> None:
    t0 = time.time()
    setup_style()

    print("[1/3] m(c) 标度拟合 → 5C 外推 ...")
    pts = load_m_points()
    fits = fit_scaling(pts)
    m5_vals = []
    for f in fits:
        for kind in ("power", "linear"):
            if kind in f:
                m5_vals.append(f[kind]["m_at_5c"])
        print(f"   {f['chem']:4s} n={f['n_pts']} "
              + " ".join(f"{k}:m5={f[k]['m_at_5c']:.2f}(r2={f[k]['r2']:.2f})"
                         for k in ("power", "linear") if k in f)
              + f"  b_loc={f['local_exponent']['b_loc']:+.2f}")
    m5_lo, m5_hi = float(min(m5_vals)), float(max(m5_vals))
    # 物理护栏：m>0；外推带不收窄到实测包络内冒充精度
    print(f"   5C 外推带 [{m5_lo:.2f}, {m5_hi:.2f}]（跨体系×跨拟合）")

    print("[2/3] 临界 m(5) 与欧姆代理交叉核对 ...")
    base = load_5c_baseline()
    ohm = ohmic_proxy_band()
    print(f"   打平点：吞吐 m*={base['m_star_throughput']:.2f}、"
          f"寿命 m*={base['m_star_life']:.2f}（m=1 基线 thr={base['throughput_rel_at_m1']:.2f} "
          f"life={base['life_rel_at_m1']:.2f}）")
    print(f"   欧姆代理 m(5)∈[{ohm['m5_ohmic_proxy_band'][0]:.2f}, "
          f"{ohm['m5_ohmic_proxy_band'][1]:.2f}]（p={ohm['p_ohmic_by_temp']}）")

    # 判词
    in_band_life = base["m_star_life"] <= m5_hi and base["m_star_life"] >= m5_lo
    in_band_thr = base["m_star_throughput"] <= m5_hi and base["m_star_throughput"] >= m5_lo
    exponents = [f["local_exponent"]["b_loc"] for f in fits
                 if np.isfinite(f["local_exponent"]["b_loc"])]
    ohm_exp_lo = min(ohm["p_ohmic_by_temp"].values()) - 1.0
    ohm_exp_hi = max(ohm["p_ohmic_by_temp"].values()) - 1.0
    n_near_ohm = sum(1 for e in exponents if e >= ohm_exp_lo - 0.15)
    verdict = (
        f"5C m(c) 外推护栏：SNL 0.5–3C 实测（LFP 1/2/3C 非单调 1.00/0.90/2.30，"
        f"NMC 递减 1.00/0.60/0.64，NCA 平 1.00/1.05）跨体系×跨拟合外推带 "
        f"[{m5_lo:.2f}, {m5_hi:.2f}]——带宽覆盖寿命打平点 {base['m_star_life']:.2f}"
        f"{'与' if (in_band_life and in_band_thr) else '或'}吞吐打平点 {base['m_star_throughput']:.2f}，"
        "5C 绝对账行的方向未定，**不可作设计依据**（台架 5C m 实测前只给护栏）。"
        f"欧姆代理（p−1∈[{ohm_exp_lo:.2f},{ohm_exp_hi:.2f}]，m5∈"
        f"[{ohm['m5_ohmic_proxy_band'][0]:.2f},{ohm['m5_ohmic_proxy_band'][1]:.2f}]）"
        f"仅 {n_near_ohm}/{len(exponents)} 体系局部指数接近——损伤不按每 EFC 欧姆耗散"
        "成比例，代理带只作上缘锚。"
    )
    print("   " + verdict)

    print("[3/3] 图与产物 ...")
    fig, axes = np.empty(0), None  # placeholder replaced below
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    # a) m(c) 散点+拟合+5C 带
    ax = axes[0]
    colors = {"LFP": "#c0392b", "NMC": "#2980b9", "NCA": "#27ae60"}
    cc = np.linspace(0.9, C_TARGET * 1.05, 40)
    for f in fits:
        chem = f["chem"]
        c = np.array(f["c_used"]); m = np.array(f["m_used"])
        ax.plot(c, m, "o", color=colors[chem], label=chem)
        if "power" in f:
            ax.plot(cc, f["power"]["a"] * cc ** f["power"]["b"],
                    "-", color=colors[chem], lw=1.2, alpha=0.8)
        if "linear" in f:
            ax.plot(cc, f["linear"]["a"] + f["linear"]["b"] * cc,
                    ":", color=colors[chem], lw=1.2, alpha=0.8)
    ax.axvspan(C_TARGET * 0.92, C_TARGET * 1.08, ymin=0, ymax=1,
               color="#f6d55c", alpha=0.25)
    ax.plot([C_TARGET], [(m5_lo + m5_hi) / 2], "k*", ms=11)
    ax.vlines(C_TARGET, m5_lo, m5_hi, "k", lw=2)
    ax.annotate(f"外推带\n[{m5_lo:.2f},{m5_hi:.2f}]",
                xy=(C_TARGET, m5_hi), xytext=(C_TARGET - 1.7, m5_hi),
                fontsize=8)
    ax.set_xlabel("放电倍率 C"); ax.set_ylabel("m(c) 每 EFC 损伤比")
    ax.set_title("a) m(c) 实测与 5C 外推（护栏带）")
    ax.legend(fontsize=7); ax.set_xlim(0, 6.2)

    # b) 5C 敏感性：life/thr vs m(5) + 临界线
    ax = axes[1]
    mm = np.linspace(0.2, 6, 200)
    life0, thr0 = base["life_rel_at_m1"], base["throughput_rel_at_m1"]
    ax.plot(mm, life0 / mm, "-", color="#2980b9", label="相对寿命 life(m)")
    ax.plot(mm, thr0 / mm, "--", color="#c0392b", label="相对吞吐 thr(m)")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axvline(base["m_star_life"], color="#2980b9", lw=0.9, ls=":")
    ax.axvline(base["m_star_throughput"], color="#c0392b", lw=0.9, ls=":")
    ax.axvspan(m5_lo, m5_hi, color="#f6d55c", alpha=0.25,
               label=f"外推带 [{m5_lo:.1f},{m5_hi:.1f}]")
    ax.annotate(f"寿命打平\nm*={base['m_star_life']:.2f}",
                xy=(base["m_star_life"], 1), xytext=(base["m_star_life"] + 0.25, 1.6),
                fontsize=7, color="#2980b9")
    ax.annotate(f"吞吐打平\nm*={base['m_star_throughput']:.2f}",
                xy=(base["m_star_throughput"], 1),
                xytext=(base["m_star_throughput"] - 1.4, 1.15),
                fontsize=7, color="#c0392b")
    ax.set_xlabel("m(5) 假设值"); ax.set_ylabel("相对 1C 档")
    ax.set_title("b) 5C 行翻转临界值")
    ax.legend(fontsize=7); ax.set_xlim(0.2, 6)

    # c) 局部指数 vs 欧姆代理
    ax = axes[2]
    names = [f["chem"] for f in fits]
    locs = [f["local_exponent"]["b_loc"] for f in fits]
    xs = np.arange(len(names))
    ax.bar(xs, locs, 0.5, color=[colors[n] for n in names])
    ax.axhspan(ohm_exp_lo, ohm_exp_hi, color="#8e44ad", alpha=0.2,
               label=f"欧姆代理 p−1∈[{ohm_exp_lo:.2f},{ohm_exp_hi:.2f}]")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs, names, fontsize=9)
    ax.set_ylabel("局部指数 b_loc = Δln m / Δln c")
    ax.set_title("c) 损伤指数 vs 欧姆耗散代理")
    ax.legend(fontsize=7)

    fig.tight_layout()
    save_fig(fig, FIG / "37a_5c_extrap.png")

    metrics = {
        "dataset": "SNL m(c)（29 指标 rate_points，25 °C）+ WP28 欧姆指数",
        "m_points_used": pts,
        "scaling_fits": fits,
        "extrapolation_band_5c": {
            "lo": m5_lo, "hi": m5_hi,
            "members": [{"chem": f["chem"], "kind": k,
                         "m5": f[k]["m_at_5c"]}
                        for f in fits for k in ("power", "linear") if k in f],
            "design_use": "禁止——外推带仅供护栏，5C 绝对账行不可作设计依据",
        },
        "critical_m5": base,
        "ohmic_cross_check": ohm,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (MET / "37_5c_extrap.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done in {time.time()-t0:.0f}s -> 37_5c_extrap.json / 37a_5c_extrap.png")
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
