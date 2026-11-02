# -*- coding: utf-8 -*-
"""WP40 机理约束下 5C m(c) 带收窄 + 绝对账 5C 行更新（v1.9 第三线）。

WP37 的 5C 外推带 [0.21, 3.35] 太宽（覆盖双打平点，方向未定）。
WP38 的机理分解给出收窄依据：

1. **LFP**：机械型超额份额 54%（EOL，锚定 SEI 本底）→
   m(3C)=2.30 分解为 m_sei≈1.06 + m_mech≈1.25。机械损伤速率
   随 C 不减（单调性，应力幅值随倍率）→ m_mech(5C)≥1.25 →
   **m_LFP(5)≥2.30**（下界=3C 水平）；上界取 WP37 经验线性外推
   3.35。标度情景（m_mech ∝ C^0/0.5/1）给中心带 2.30–3.13。
2. **NMC**：机械份额 0（超额为负）→ 无机理升高；m<1 是异质性
   nuisance（v1.6 护栏：nuisance 不许外推）→ m(5) 中性带
   [0.7, 1.1]（rep 配对温度包络）。WP37 的 [0.21, 0.46] 递减外推
   作废（把 nuisance 外推到 5C 是错误方向）。
3. **NCA**：无 3C 组（机理不可评）；2C 平（1.05）+ 温度包络 →
   [0.85, 1.30] 中性带。

方向判定（打平点 WP37：吞吐 3.10、寿命 0.78）：
- NMC/NCA：吞吐收益方向**定为正**（带 « 3.10）——比 WP37 的
  「方向未定」升级；LFP 打平点落在收窄带 [2.30, 3.35] 上部
  76% 分位（偏赢侧，台架定夺）。
- 每 EFC 寿命：LFP/NCA 带在 0.78 上方（5C 输寿命，方向定）；
  NMC 带 [0.7, 1.1] 窄幅跨打平（仍未定）。

边界：5C×T 交互不可辨识（3C×T 已不可辨识，WP39）——T-flat
代理假设双重外推；LFP 中心情景（m_mech∝C）恰在打平 3.13≈3.10
附近，台架 5C m 实测前 LFP 行仍不可作设计依据。

产物：results/metrics/40_5c_refine.json、results/figures/40a_5c_refine.png
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

import matplotlib.pyplot as plt  # noqa: E402

MET = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures"

C_TARGET = 5.0


def load_json(name: str) -> dict:
    return json.loads((MET / name).read_text(encoding="utf-8"))


def main() -> None:
    t0 = time.time()
    MET.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    d37 = load_json("37_5c_extrap.json")
    d38 = load_json("38_mech_attribution.json")
    d29 = load_json("29_rate_effect.json")

    m_star_thr = float(d37["critical_m5"]["m_star_throughput"])   # 3.10
    m_star_life = float(d37["critical_m5"]["m_star_life"])       # 0.78
    thr0 = float(d37["critical_m5"]["throughput_rel_at_m1"])     # 3.10
    life0 = float(d37["critical_m5"]["life_rel_at_m1"])          # 0.78
    old_band = d37["extrapolation_band_5c"]
    old_members = {(m["chem"], m["kind"]): m["m5"] for m in old_band["members"]}

    print("[1/3] 机理约束带 ...")
    # LFP 分解（WP38 锚定超额）
    lfp3_boot = d38["anchored_excess"]["LFP_3C"]["bootstrap"]
    share = float(lfp3_boot["mech_share_median"])
    m3 = next(p["m_per_efc"] for p in d29["rate_points"]
              if p["chem"] == "LFP" and abs(p["c_dis"] - 3.0) < 1e-9)
    m_sei = (1.0 - share) * m3
    m_mech = share * m3
    print(f"   LFP 3C 分解：m_sei={m_sei:.2f} m_mech={m_mech:.2f}（share={share:.2f}）")

    # 标度情景 m_mech(5) = m_mech(3)·(5/3)^γ
    scenarios = {}
    for gamma, tag in ((0.0, "flat"), (0.5, "sqrt_c"), (1.0, "linear_c")):
        m5 = m_sei + m_mech * (C_TARGET / 3.0) ** gamma
        scenarios[tag] = {"gamma": gamma, "m5": float(m5)}
        print(f"   LFP 情景 {tag}: m(5)={m5:.2f}")

    lfp_lo = min(s["m5"] for s in scenarios.values())            # flat=2.30
    lfp_hi = float(old_members.get(("LFP", "linear"), 3.35))     # 经验上界
    lfp_band = (float(lfp_lo), lfp_hi)
    nmc_band = (0.7, 1.1)      # 中性带（rep 配对温度包络，v1.9 §39）
    nca_band = (0.85, 1.30)    # 中性带（2C 平 + 温度包络）

    bands = {"LFP": lfp_band, "NMC": nmc_band, "NCA": nca_band}
    print(f"   收窄带：LFP {lfp_band} NMC {nmc_band} NCA {nca_band}")

    print("[2/3] 方向判定 ...")
    verdicts = {}
    for chem, (lo, hi) in bands.items():
        # 吞吐 thr(m)=thr0/m、寿命 life(m)=life0/m（WP37 口径）
        thr_rng = (thr0 / hi, thr0 / lo)
        life_rng = (life0 / hi, life0 / lo)
        if hi < m_star_thr:
            thr_dir = "win_determined"
        elif lo > m_star_thr:
            thr_dir = "lose_determined"
        else:
            frac = (m_star_thr - lo) / (hi - lo) if hi > lo else float("nan")
            thr_dir = f"straddle(打平 {frac:.0%} 分位)"
        if lo > m_star_life:
            life_dir = "lose_determined"
        elif hi < m_star_life:
            life_dir = "win_determined"
        else:
            life_dir = "straddle"
        verdicts[chem] = {
            "band": [lo, hi],
            "throughput_rel_range": [float(x) for x in thr_rng],
            "life_rel_range": [float(x) for x in life_rng],
            "throughput_direction": thr_dir,
            "life_direction": life_dir,
        }
        print(f"   {chem}: 吞吐 {thr_dir}（rel {thr_rng[0]:.2f}–{thr_rng[1]:.2f}） "
              f"寿命 {life_dir}（rel {life_rng[0]:.2f}–{life_rng[1]:.2f}）")

    print("[3/3] 图 + 产物 ...")
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    colors = {"LFP": "#c0392b", "NMC": "#2980b9", "NCA": "#27ae60"}

    ax = axes[0, 0]
    for p in d29["rate_points"]:
        if abs(p["temp_c"] - 25.0) > 1e-6:
            continue
        ax.plot(p["c_dis"], p["m_per_efc"], "o", color=colors[p["chem"]],
                ms=7, mfc="white" if not p["eligible"] else colors[p["chem"]])
    for chem, (lo, hi) in bands.items():
        ax.plot([4.7, 5.3], [hi, hi], color=colors[chem], lw=2)
        ax.plot([4.7, 5.3], [lo, lo], color=colors[chem], lw=2)
        ax.fill_between([4.7, 5.3], lo, hi, color=colors[chem], alpha=0.25)
        ob = (old_members.get((chem, "power")), old_members.get((chem, "linear")))
        ax.plot([5.45, 5.45], [min(x for x in ob if x), max(x for x in ob if x)],
                color="gray", lw=2, alpha=0.6)
    ax.axhline(m_star_thr, color="black", ls="--", lw=1.2)
    ax.axhline(m_star_life, color="black", ls=":", lw=1.2)
    ax.text(5.6, m_star_thr, "吞吐打平 3.10", fontsize=7, va="center")
    ax.text(5.6, m_star_life, "寿命打平 0.78", fontsize=7, va="center")
    ax.set_xlim(0.3, 6.2)
    ax.set_xlabel("放电倍率 [C]")
    ax.set_ylabel("m(c)")
    ax.set_title("a) m(c) + 5C 机理约束带（灰=旧外推带）")

    ax = axes[0, 1]
    gammas = np.linspace(0, 1.5, 60)
    m5_curve = m_sei + m_mech * (C_TARGET / 3.0) ** gammas
    ax.plot(gammas, m5_curve, color=colors["LFP"], lw=2)
    for tag, s in scenarios.items():
        ax.plot(s["gamma"], s["m5"], "o", color=colors["LFP"], ms=8)
        ax.annotate(tag, (s["gamma"], s["m5"]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.axhline(m_star_thr, color="black", ls="--", lw=1.2, label="吞吐打平 3.10")
    ax.axhline(lfp_hi, color="gray", ls="-.", lw=1.2, label="经验上界 3.35")
    ax.fill_between(gammas, lfp_band[0], lfp_band[1], color=colors["LFP"], alpha=0.12)
    ax.set_xlabel("机械标度指数 γ（m_mech∝C^γ）")
    ax.set_ylabel("LFP m(5)")
    ax.set_title("b) LFP 机械标度情景")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    m_grid = np.linspace(0.2, 4.0, 200)
    ax.plot(m_grid, thr0 / m_grid, "k-", lw=1.8, label="吞吐 rel")
    ax.plot(m_grid, life0 / m_grid, "k--", lw=1.8, label="寿命 rel")
    ax.axhline(1.0, color="gray", lw=1)
    for chem, (lo, hi) in bands.items():
        ax.axvspan(lo, hi, color=colors[chem], alpha=0.15)
        ax.plot([(lo + hi) / 2], [1.0], "^", color=colors[chem], ms=9)
    ax.axvline(m_star_thr, color="black", ls="--", lw=1)
    ax.axvline(m_star_life, color="black", ls=":", lw=1)
    ax.set_xlabel("m(5)")
    ax.set_ylabel("相对 1C 的 rel")
    ax.set_title("c) 吞吐/寿命 rel 与收窄带")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        "绝对账 5C 行（机理约束收窄带）",
        "",
    ]
    for chem in ("LFP", "NMC", "NCA"):
        v = verdicts[chem]
        lo, hi = v["band"]
        lines.append(f"{chem}: m∈[{lo:.2f}, {hi:.2f}]")
        lines.append(f"  吞吐 rel {v['throughput_rel_range'][0]:.2f}"
                     f"–{v['throughput_rel_range'][1]:.2f}  {v['throughput_direction']}")
        lines.append(f"  寿命 rel {v['life_rel_range'][0]:.2f}"
                     f"–{v['life_rel_range'][1]:.2f}  {v['life_direction']}")
    lines += [
        "",
        "护栏：LFP 中心情景∝C 恰在打平附近（3.13≈3.10），",
        "台架 5C m 实测前 LFP 行仍不可作设计依据；",
        "NMC/NCA 中性带可用于方向性判断（nuisance 不外推）。",
        "5C×T 不可辨识（WP39）——T-flat 代理假设双重外推。",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="#eaf2f8", edgecolor="#2980b9"))

    fig.tight_layout()
    save_fig(fig, FIG / "40a_5c_refine.png")

    verdict = (
        f"5C m(c) 机理约束收窄：LFP [{lfp_band[0]:.2f}, {lfp_band[1]:.2f}]"
        f"（机械份额 54%→单调性下界 2.30=3C 水平，标度中心 2.30–3.13），"
        f"打平 3.10 落带上部 {verdicts['LFP']['throughput_direction']}；"
        f"NMC [{nmc_band[0]:.1f}, {nmc_band[1]:.1f}]（无机理升高，nuisance 不外推，"
        f"旧递减带 [0.21, 0.46] 作废）→ 吞吐收益**方向定为正**；"
        f"NCA [{nca_band[0]:.2f}, {nca_band[1]:.2f}] 吞吐方向定为正。"
        f"每 EFC 寿命：LFP/NCA 5C 输（带>0.78，方向定），NMC 窄跨打平仍未定。"
        f"LFP 中心情景∝C 恰在打平（3.13≈3.10）——台架前 LFP 行仍不可作设计依据；"
        f"5C×T 不可辨识（T-flat 代理双重外推）。"
    )
    metrics = {
        "dataset": "WP37 外推带 + WP38 机理分解 + WP29 m(c) + WP39 温度包络",
        "break_even": {"m_star_throughput": m_star_thr, "m_star_life": m_star_life,
                       "throughput_rel_at_m1": thr0, "life_rel_at_m1": life0},
        "lfp_decomposition": {
            "m_3c": m3, "mech_share": share,
            "m_sei": float(m_sei), "m_mech": float(m_mech),
            "scenarios": scenarios,
            "mech_monotonicity_lower_bound": float(lfp_lo),
            "empirical_upper": lfp_hi,
        },
        "refined_bands": {c: list(b) for c, b in bands.items()},
        "old_band": {"lo": old_band["lo"], "hi": old_band["hi"],
                     "note": "WP37 跨体系×跨拟合包络（NMC nuisance 递减外推已作废）"},
        "direction_verdicts": verdicts,
        "guardrails": [
            "LFP 中心情景（m_mech∝C）恰在吞吐打平附近 → 台架 5C m 前不可作设计依据",
            "NMC nuisance（m<1）不外推（v1.6 护栏）——旧 [0.21,0.46] 作废",
            "5C×T 不可辨识（3C×T 已不可辨识，WP39）——T-flat 代理双重外推",
            "NCA 无 3C 组（机理不可评）——中性带是证据缺失的保守带",
        ],
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (MET / "40_5c_refine.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(verdict)
    print(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
