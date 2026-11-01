"""29 · 倍率效应项入寿命模型：m(c) 区间 + 机理归因 + 跨倍率设计权衡。

背景（WP28 开放项 #4，报告 §38）
-------------------------------
共享振幅假设（WP23）在 3C 双向失效：LFP 3C/1C 每 EFC ×2.30、
NMC ×0.64。本脚本把 m(c) 显式入账（d = m(c)·k(T)），并回答两个问题：

1. **C5 主口径受不受影响？** 不受——C5 是同倍率对比（3C 液冷 vs
   3C 自然），m 消掉。2.14 / 识别集 [1.56, 2.64] 保留。
2. **跨倍率分档（1C 自然/2C 风冷/3C 液冷/5C）在寿命账下还成立吗？**
   这是 m(c) 与 k(T) 的竞争：LFP 3C 的 m=2.3 吃掉液冷热优势，
   NMC 3C 的 m=0.64 反而放大——分档从"普适排序"降级为
   "热学排序 + 体系相关寿命修正"。

机理归因（文献口径）：LFP 3C 升高有机制候选（开裂/接触/快充析锂），
NMC 3C 降低无机制支持 → 异质性 nuisance，不入机制账。

产出：
  results/metrics/29_rate_effect.json
  results/figures/29a_rate_effect.png
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
from bmthermal.network import build_cell, simulate_two_node  # noqa: E402
from bmthermal.heat import q_profile  # noqa: E402
from bmthermal.rate_effect import (  # noqa: E402
    cross_chem_envelope, damage_per_efc, mechanism_attribution,
    rate_points, tier_tradeoff_row,
)
from bmthermal.snl_aging import load_dataset  # noqa: E402

OUT = ROOT / "results"
MET = OUT / "metrics"
CACHE = OUT / "cache"
FIG = OUT / "figures"
for p in (MET, CACHE, FIG):
    p.mkdir(parents=True, exist_ok=True)

SNL_ZIP = Path("D:/项目/02_数据集/01_国际公开集/Sandia_SNL/SNL.zip")
CHEMS = ("LFP", "NMC", "NCA")
T_AMB = 25.0
# 跨倍率设计档（v1.0 分档）：h 取"温升 20 K 需求"（02_required_h）
TIERS = (
    ("1C 自然", 1.0, "natural"),
    ("2C 风冷", 2.0, "forced_air"),
    ("3C 液冷", 3.0, "cold_plate"),
    ("5C 液冷贴线", 5.0, "cold_plate"),
)


def load_required_h() -> dict:
    p = json.loads((MET / "02_required_h.json").read_text(encoding="utf-8"))
    return {float(r["c_rate"]): float(r["h_required_w_m2k"]) for r in p["required"]}


def main() -> None:
    t0 = time.time()

    # ---- 1) SNL m(c) 识别区间 + 机理归因 ----
    print("[1/3] SNL m(c) 识别区间 ...")
    df = load_dataset(SNL_ZIP, cache_csv=str(CACHE / "snl_summary.csv"))
    all_points = []
    attributions = []
    for chem in CHEMS:
        pts = rate_points(df, chem, temp_c=25.0)
        all_points.extend(pts)
        for p in pts:
            if p.role == "ref":
                continue
            attributions.append(mechanism_attribution(
                chem, p.c_dis, p.m_per_efc, p.c_chg_mean))
        line = "  ".join(
            f"{p.c_dis:g}C:{p.m_per_efc:.2f}({p.role[:3]},n={p.n_cells},chg={p.c_chg_mean:.2f})"
            for p in pts
        )
        print(f"   {chem:4s} {line}")
    for a in attributions:
        if a.direction != "neutral":
            print(f"   [{a.direction:8s}] {a.verdict}")

    envelope = cross_chem_envelope(all_points)
    print("   跨体系包络:", {c: f"[{v['lo']:.2f},{v['hi']:.2f}]"
                          for c, v in envelope.items()})

    # ---- 2) 跨倍率设计权衡：m(c) × k(T) ----
    print("[2/3] 跨倍率设计权衡 ...")
    h_req = load_required_h()
    qcache: dict = {}
    tier_rows = []
    thermal_only = {}
    for label, c_rate, scheme in TIERS:
        h = h_req[c_rate]
        t_arr, q_arr, info = q_profile(c_rate, t_amb_c=T_AMB, cache=qcache)
        cell = build_cell(h=h)
        sim = simulate_two_node(t_arr, q_arr, T_AMB, cell)
        core = sim["t_core_c"]
        dt_s = float(t_arr[-1] - t_arr[0])
        k_only = float(np.mean(np.exp(
            -45000.0 / 8.314 * (1.0 / (core + 273.15) - 1.0 / 298.15))))
        thermal_only[c_rate] = k_only
        # m 取 LFP / NMC 实测点（n≥2），包络上下界；5C 无 SNL 组 → 不修正（开放项）
        m_by_chem = {p.chem: p.m_per_efc for p in all_points
                     if abs(p.c_dis - c_rate) < 1e-9}
        env = envelope.get(c_rate, {"lo": 1.0, "hi": 1.0})
        if m_by_chem:
            m_tags = [("LFP", m_by_chem.get("LFP", 1.0)),
                      ("NMC", m_by_chem.get("NMC", 1.0)),
                      ("env_lo", env["lo"]), ("env_hi", env["hi"])]
            m_note = ""
        else:
            m_tags = [("no_data", 1.0), ("env_lo", 1.0), ("env_hi", 1.0)]
            m_note = "无 SNL 该倍率组，m=1 未修正（开放项）"
        # 壁钟每 EFC：同倍率充+放 + 10 min 静置（通用工况口径）
        dt_cycle = 2.0 * float(t_arr[-1] - t_arr[0]) + 600.0
        for tag, m in m_tags:
            row = tier_tradeoff_row(c_rate, m, core, dt_cycle)
            row.update({"tier": label, "scheme": scheme, "h_w_m2k": h,
                        "m_tag": tag, "m_note": m_note,
                        "t_core_peak_c": float(core.max()),
                        "duration_s": float(t_arr[-1] - t_arr[0]),
                        "dt_cycle_s": dt_cycle,
                        "q_source": info.get("source", "")})
            tier_rows.append(row)
        print(f"   {label:10s} {c_rate:g}C  h={h:6.1f}  Tc峰={core.max():5.1f}°C  "
              f"<k>={k_only:.2f}  m(LFP)={m_by_chem.get('LFP', float('nan')):.2f}  "
              f"m(NMC)={m_by_chem.get('NMC', float('nan')):.2f}")

    # 归一到 1C 档（m=1、k(T_1C)）——life/throughput 才可比
    k1 = thermal_only[1.0]
    for r in tier_rows:
        r["damage_rel_vs_1c"] = r["damage_per_efc_rel"] / (1.0 * k1)
        r["life_efc_rel_vs_1c"] = 1.0 / r["damage_rel_vs_1c"]
        r["life_hours_rel_vs_1c"] = r["life_hours_rel"] / (
            1.0 / (1.0 * k1) * tier_rows[0]["dt_s"] / 3600.0)
        r["throughput_rel_vs_1c"] = r["throughput_per_hour_rel"] / (
            3600.0 / (1.0 * k1 * tier_rows[0]["dt_s"]))

    # ---- 3) 图 + 指标 ----
    print("[3/3] 图与指标 ...")
    plotting.setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    marks = {"LFP": "o", "NMC": "s", "NCA": "^"}
    for chem in CHEMS:
        pts = [p for p in all_points if p.chem == chem]
        xs = [p.c_dis for p in pts]
        ys = [p.m_per_efc for p in pts]
        solid = [p.eligible or p.role == "ref" for p in pts]
        ax.plot(xs, ys, marks[chem] + "-", label=chem, lw=1.4)
        for p in pts:
            if p.role == "indicative":
                ax.plot([p.c_dis], [p.m_per_efc], marks[chem], mfc="white",
                        mec=ax.lines[-1].get_color())
    env_c = sorted(envelope)
    if env_c:
        ax.fill_between(env_c, [envelope[c]["lo"] for c in env_c],
                        [envelope[c]["hi"] for c in env_c],
                        alpha=0.15, color="gray", label="跨体系包络")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("放电倍率 C")
    ax.set_ylabel("m(c)：每 EFC 损伤比 vs 1C")
    ax.set_title("a) m(c) 识别区间（空心=单芯指示）")
    ax.legend(fontsize=8)

    ax = axes[1]
    cs = [t[1] for t in TIERS]
    labels = [t[0] for t in TIERS]
    x = np.arange(len(cs))
    w = 0.27

    def get_row(c, tag, key):
        return next((r[key] for r in tier_rows
                     if r["c_rate"] == c and r["m_tag"] == tag), None)

    for i, (tag, color) in enumerate((("LFP", "#c0392b"), ("NMC", "#2980b9"))):
        vals = [get_row(c, tag, "damage_rel_vs_1c") for c in cs]
        ok = [v is not None for v in vals]
        ax.bar([xx + (i - 1) * w for xx, o in zip(x, ok) if o],
               [v for v in vals if v], w, label=tag, color=color)
    ax.plot(x, [thermal_only[c] / k1 for c in cs], "k--o",
            label="仅热学 k(T)（共享振幅）", lw=1.4)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("每 EFC 损伤（相对 1C 档）")
    ax.set_title("b) m(c)×k(T) vs 仅 k(T)（5C 无 m 实测）")
    ax.legend(fontsize=8)

    ax = axes[2]
    for tag, color in (("LFP", "#c0392b"), ("NMC", "#2980b9")):
        life = [get_row(c, tag, "life_efc_rel_vs_1c") for c in cs]
        thr = [get_row(c, tag, "throughput_rel_vs_1c") for c in cs]
        ok = [v is not None for v in life]
        xs = [xx for xx, o in zip(x, ok) if o]
        ax.plot(xs, [v for v in life if v], "o-", color=color, label=f"{tag} 寿命(EFC)")
        ax.plot(xs, [v for v in thr if v], "s--", color=color, label=f"{tag} 吞吐(EFC/时)")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("相对 1C 档")
    ax.set_title("c) 相对寿命 vs 相对吞吐")
    ax.legend(fontsize=7)
    fig.tight_layout()
    plotting.save(fig, "29a_rate_effect")

    lfp3 = next((r for r in tier_rows if r["c_rate"] == 3.0 and r["m_tag"] == "LFP"), None)
    nmc3 = next((r for r in tier_rows if r["c_rate"] == 3.0 and r["m_tag"] == "NMC"), None)
    verdict = (
        "m(c) 入账后：① 同倍率 C5 不受影响（m 消掉，主口径 2.14/识别集 [1.56,2.64] 保留）；"
        f"② 跨倍率分档体系相关——3C 档 LFP 相对寿命 {lfp3['life_efc_rel_vs_1c']:.2f}"
        f"（m=2.30 吃掉热优势）vs NMC {nmc3['life_efc_rel_vs_1c']:.2f}（m=0.64 放大），"
        "『1C 自然/2C 风冷/3C 液冷』降级为热学排序+体系相关寿命修正；"
        "③ 机理归因：LFP 3C 升高有机制候选（开裂/接触/快充析锂），NMC 3C 降低无机制支持"
        "→ 异质性 nuisance，不入机制账。"
    )
    print("   " + verdict)

    out = {
        "rate_points": [p.to_dict() for p in all_points],
        "attributions": [a.to_dict() for a in attributions],
        "cross_chem_envelope": envelope,
        "tier_rows": tier_rows,
        "thermal_only_k": thermal_only,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (MET / "29_rate_effect.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成，{time.time() - t0:.1f}s → 29_rate_effect.json / 29a_rate_effect.png")


if __name__ == "__main__":
    main()
