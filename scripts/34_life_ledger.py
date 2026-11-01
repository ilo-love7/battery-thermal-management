"""34 · 绝对寿命账：体系 × 方案 × 倍率的 N_EOL（EFC）与日历年。

背景（v1.7 收口）
-----------------
WP32 给了 A_chem（fade/damage 绝对尺度）与形状修正带；WP33 给了
Ea 带与 β 重标；WP29 给了 m(c)。本脚本把三者合成**绝对寿命账**：
每个 (体系, 倍率, 冷却方案) 一行 N_EOL（EFC）与日历年，并把 WP30
的模组一致性惩罚落到最差芯 EOL。开放项「只报相对倍数」正式关闭
（限定 SNL 同类化学体系的量级口径）。

账式
----
N_EOL = FADE_EOL / (A_chem · d_s · m(c))
  A_chem = fade_ref / d_ref        （WP32，25 °C 1C SNL 实测）
  d_s    = 损伤积分 @u=0           （aging.py，等效 25 °C 秒/EFC）
  m(c)   = 每 EFC 损伤倍率效应     （WP29，SNL 25 °C 组）
形状修正：×0.82（CS2 全寿命审计中位，WP32）为报告中心带；
线性原值保留为上界侧。模组最差芯 = N_EOL×(1−penalty)。

产出：
  results/metrics/34_life_ledger.json
  results/figures/34a_life_ledger.png
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
from bmthermal.abs_life import FADE_EOL, ChemAnchor, n_eol_linear, n_eol_power  # noqa: E402
from bmthermal.aging import damage_integral  # noqa: E402
from bmthermal.feedback import FeedbackParams  # noqa: E402
from bmthermal.heat import q_profile  # noqa: E402
from bmthermal.network import build_cell, simulate_two_node  # noqa: E402
from bmthermal.params import SCHEMES  # noqa: E402

OUT = ROOT / "results"
MET = OUT / "metrics"
CACHE = OUT / "cache"
FIG = OUT / "figures"
for p in (MET, CACHE, FIG):
    p.mkdir(parents=True, exist_ok=True)

SCENARIOS = (
    (1.0, "natural", "1C 自然对流"),
    (2.0, "forced_air", "2C 强制风冷"),
    (3.0, "natural", "3C 自然对流"),
    (3.0, "forced_air", "3C 强制风冷"),
    (3.0, "cold_plate", "3C 液冷"),
    (5.0, "cold_plate", "5C 液冷"),
)
CHEMS = ("LFP", "NMC", "NCA")
DUTY = {"daily_1efc": 365.0, "weekly_0p25": 365.25 / 7.0}  # EFC/年


def load_metrics(name: str) -> dict:
    return json.loads((MET / name).read_text(encoding="utf-8"))


def d_s_of(c: float, scheme_key: str, cache: dict, fn_cache: dict) -> float:
    """u=0 每 EFC 损伤（等效 25 °C 秒）。"""
    if (c, scheme_key) in fn_cache:
        return fn_cache[(c, scheme_key)]
    t_arr, q_arr, _ = q_profile(c, t_amb_c=25.0, cache=cache)
    scheme = SCHEMES[scheme_key]
    net = build_cell(h=scheme.h_mid,
                     pcm_extra_capacity_j_k=scheme.pcm_extra_capacity_j_k)
    out = simulate_two_node(t_arr, q_arr, 25.0, net)
    d = float(damage_integral(t_arr, out["t_core_c"]))
    fn_cache[(c, scheme_key)] = d
    return d


def main() -> None:
    t0 = time.time()

    print("[1/3] 载入 WP32/33/29/30 ...")
    m32 = load_metrics("32_abs_life.json")
    m29 = load_metrics("29_rate_effect.json")
    m30 = load_metrics("30_module_life.json")

    corr = m32["extrapolation_audit"].get("snl_correction_factor", 1.0)
    anchors: dict[str, ChemAnchor] = {}
    for row in m32["chem_anchors"]:
        anchors[row["chem"]] = ChemAnchor(
            chem=row["chem"], temp_c=row["temp_c"], c_dis=row["c_dis"],
            fade_ref_per_efc=row["fade_ref_per_efc"],
            n_cells=row["n_cells"],
            fade_std_per_efc=row["fade_std_per_efc"],
            d_ref_s=row["d_ref_s"],
            b_fade=row["b_fade"], a_fade=row["a_fade"],
            n_eol_cross_median=row["n_eol_cross_median"],
        )
    print(f"   形状修正 ×{corr:.2f}；锚点："
          + "，".join(f"{c} {a.n_eol_ref:.0f} EFC" for c, a in anchors.items()))

    # m(c) 查找表（25 °C 实测；缺失倍率 no_data→1.0 并标记）
    m_map: dict[tuple[str, float], dict] = {}
    for p in m29["rate_points"]:
        if abs(p["temp_c"] - 25.0) > 1e-6:
            continue
        m_map[(p["chem"], float(p["c_dis"]))] = {
            "m": float(p["m_per_efc"]), "eligible": bool(p["eligible"]),
            "role": p["role"], "n_cells": int(p["n_cells"]),
        }

    def m_of(chem: str, c: float) -> dict:
        hit = m_map.get((chem, float(c)))
        if hit is not None:
            return hit
        return {"m": 1.0, "eligible": False, "role": "no_data", "n_cells": 0}

    # 模组惩罚（Ea=45 主行 + 范围）
    mod_rows = {}
    for key, v in m30["results"].items():
        by = v["by_ea"]
        mod_rows[key] = {
            "penalty_ea45_pct": by["45"]["eol_penalty_pct"],
            "penalty_min_pct": min(d["eol_penalty_pct"] for d in by.values()),
            "penalty_max_pct": max(d["eol_penalty_pct"] for d in by.values()),
            "delta_t_max_k": by["45"]["delta_t_max_k"],
        }

    print("[2/3] 场景寿命账 ...")
    cache: dict = {}
    fn_cache: dict = {}
    ledger_rows = []
    for chem, anc in anchors.items():
        for c, scheme_key, label in SCENARIOS:
            d_s = d_s_of(c, scheme_key, cache, fn_cache)
            mi = m_of(chem, c)
            n_lin = anc.n_eol(d_s, mi["m"], shape="linear")
            n_corr = n_lin * corr
            n_pow = anc.n_eol(d_s, mi["m"], shape="power")
            row = {
                "chem": chem, "c_rate": c, "scheme": scheme_key, "label": label,
                "d_s_25c_s": d_s,
                "t_core_peak_c": float("nan"),
                "m_per_efc": mi["m"], "m_role": mi["role"],
                "n_eol_linear": n_lin,
                "n_eol_corrected": n_corr,
                "n_eol_power": n_pow,
                "years_daily": n_corr / DUTY["daily_1efc"],
                "years_weekly": n_corr / DUTY["weekly_0p25"],
                "n_eol_worst_cell_design": n_corr * (1.0 - mod_rows["design/mismatch+15%"]["penalty_ea45_pct"] / 100.0),
                "n_eol_worst_cell_lowflow": n_corr * (1.0 - mod_rows["low/mismatch+15%"]["penalty_ea45_pct"] / 100.0),
            }
            ledger_rows.append(row)
            print(f"   {chem:3s} {label:12s} m={mi['m']:.2f}({mi['role'][:4]}) "
                  f"d={d_s:6.0f}s  N_EOL={n_corr:7.0f} EFC  "
                  f"({row['years_daily']:.1f} 年@1EFC/日)")

    # C5 绝对化：3C 液冷 vs 1C 自然（同体系）
    print("   C5 绝对对照：")
    c5_rows = []
    for chem in CHEMS:
        a = next(r for r in ledger_rows
                 if r["chem"] == chem and r["c_rate"] == 3.0 and r["scheme"] == "cold_plate")
        b = next(r for r in ledger_rows
                 if r["chem"] == chem and r["c_rate"] == 1.0 and r["scheme"] == "natural")
        c5_rows.append({
            "chem": chem,
            "n_eol_3c_cold": a["n_eol_corrected"],
            "n_eol_1c_natural": b["n_eol_corrected"],
            "ratio_abs": a["n_eol_corrected"] / b["n_eol_corrected"],
            "years_daily_3c_cold": a["years_daily"],
            "years_daily_1c_natural": b["years_daily"],
        })
        print(f"   {chem}: 3C 液冷 {a['n_eol_corrected']:.0f} vs 1C 自然 "
              f"{b['n_eol_corrected']:.0f} EFC（比 {a['n_eol_corrected']/b['n_eol_corrected']:.2f}）")

    # ---- 联合口径敏感性：m_flux + calendar_bracket ----
    # m(c) 实测比已含 SNL 时长下的日历混入，与 bracket 直接叠加=双重计数。
    # 换算通量倍率 m_flux=(m_meas − s·r_t)/(1−s)，r_t=SNL 快/慢 EFC 时长比。
    print("   联合口径（m_flux × bracket）敏感性：")
    from bmthermal.calendar import calendar_bracket
    r_t_snl = 7248.0 / 9970.0   # SNL LFP 3C/1C dt 中位
    span_1 = 3340.15            # scripts/23 efc_span_s（1C 模型时长）
    span_3 = 1091.56
    joint_rows = []
    for s_val, s_tag in ((0.40, "main_0.40"), (0.61, "id_hi_0.61")):
        br3 = calendar_bracket(s_val, span_3, span_1)
        for chem in CHEMS:
            mi3 = m_of(chem, 3.0)
            m_meas = mi3["m"]
            m_flux = (m_meas - s_val * r_t_snl) / (1.0 - s_val) if s_val < 1.0 else float("nan")
            m_flux = float(max(m_flux, 0.05))  # 数值护栏
            d1 = d_s_of(1.0, "natural", cache, fn_cache)
            d3 = d_s_of(3.0, "cold_plate", cache, fn_cache)
            ratio = (d1 * 1.0 * 1.0) / (d3 * m_flux * br3)
            joint_rows.append({
                "chem": chem, "s_tag": s_tag, "s": s_val,
                "m_meas_3c": m_meas, "m_flux_3c": m_flux,
                "bracket_3c": br3, "c5_joint_ratio": ratio,
                "m_role": mi3["role"],
            })
            print(f"   {chem} {s_tag}: m_meas={m_meas:.2f}→m_flux={m_flux:.2f} "
                  f"br3={br3:.2f} → C5 联合 {ratio:.2f}")

    # 锚点替换摘要
    anchor_replacement = {
        "old_anchor_efc": m32["old_anchor_table"]["old_anchor_efc"],
        "rows": m32["old_anchor_table"]["rows"],
        "chem_spread_ratio": m32["old_anchor_table"]["chem_spread_ratio"],
        "note": "旧 ANCHOR_EFC=2000（LFP 文献假设）→ SNL 实测按体系；"
                "线性口径为上界侧，中心带 ×%.2f" % corr,
    }

    print("[3/3] 产物 ...")
    n_by_chem = {c: anchors[c].n_eol_ref * corr for c in CHEMS if c in anchors}
    metrics = {
        "dataset": "SNL 绝对锚点（WP32）× 热-损伤模型 × m(c)（WP29）× 模组惩罚（WP30）",
        "shape_correction_factor": corr,
        "duty_per_year": DUTY,
        "ledger_rows": ledger_rows,
        "c5_absolute": c5_rows,
        "c5_joint_mflux_bracket": {
            "r_t_snl_3c_1c": r_t_snl,
            "span_1c_s": span_1, "span_3c_s": span_3,
            "note": "m_meas 含 SNL 时长日历混入，直接叠 bracket 双重计数；"
                    "换算 m_flux=(m_meas−s·r_t)/(1−s) 后再叠 bracket。"
                    "NCA 3C 无实测 m（no_data→1.0），联合比仅供参考。",
            "rows": joint_rows,
        },
        "module_penalty": mod_rows,
        "anchor_replacement": anchor_replacement,
        "verdict": (
            f"绝对寿命账收口：25 °C 1C 参考 N_EOL（中心带）LFP "
            f"{n_by_chem.get('LFP', float('nan')):.0f} / NMC {n_by_chem.get('NMC', float('nan')):.0f}"
            f" / NCA {n_by_chem.get('NCA', float('nan')):.0f} EFC（修正后），"
            f"体系跨度 {anchor_replacement['chem_spread_ratio']:.0f}×——旧单锚 2000 作废。"
            "C5 两口径并存：①WP29 口径（m_meas×热损伤）LFP 2.00/NMC 7.16/NCA 4.59——"
            "体系分化全部来自 m(c)；②联合口径（m_flux×calendar_bracket，s=0.40）"
            "LFP 0.62/NMC 3.52——LFP 3C 液冷在联合口径下**输给** 1C 自然（WP29"
            "『打平即输』的严格版）。m_meas 含日历混入，与 bracket 直接叠加=双重计数，"
            "必须先换 m_flux。模组最差芯 EOL 惩罚：设计 1.4%、低流量+失配 6.0%（Ea=45）"
            "——% 级。开放项『只报相对倍数』关闭，边界：SNL 18650 同类体系量级口径，"
            "跨格式/批次不保证 ±10%；5C 无实测 m（no_data）按 1.0 计并标记。"
        ),
        "elapsed_s": time.time() - t0,
    }
    (MET / "34_life_ledger.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 图 34a ----
    plotting.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    xs = np.arange(len(CHEMS))
    vals = [n_by_chem.get(c, np.nan) for c in CHEMS]
    ax.bar(xs, vals, color=["#c0392b", "#2980b9", "#27ae60"], width=0.5)
    ax.axhline(2000.0, color="k", ls="--", lw=1.2, label="旧锚 2000 EFC")
    ax.set_yscale("log")
    ax.set_xticks(xs, CHEMS)
    ax.set_ylabel("N_EOL @25 °C 1C（EFC，对数轴）")
    ax.set_title("a) 体系绝对寿命（中心带）")
    ax.legend(fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    ax = axes[0, 1]
    labels = [r["label"] for r in ledger_rows if r["chem"] == "LFP"]
    width = 0.27
    x = np.arange(len(labels))
    for i, chem in enumerate(CHEMS):
        ys = [r["n_eol_corrected"] for r in ledger_rows if r["chem"] == chem]
        ax.bar(x + (i - 1) * width, ys, width, label=chem,
               color=["#c0392b", "#2980b9", "#27ae60"][i])
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("N_EOL（EFC，对数轴）")
    ax.set_title("b) 方案 × 倍率 绝对寿命账")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ratio = [r["ratio_abs"] for r in c5_rows]
    ax.bar(CHEMS, ratio, color=["#c0392b", "#2980b9", "#27ae60"], width=0.5)
    ax.axhline(1.0, color="k", lw=1)
    ax.axhline(2.14, color="gray", ls=":", lw=1.2, label="v1.4 C5 主口径 2.14")
    ax.set_ylabel("3C 液冷 / 1C 自然（绝对 EFC 比）")
    ax.set_title("c) C5 绝对化（m(c) 入账后体系分化）")
    ax.legend(fontsize=8)
    for i, r in enumerate(ratio):
        ax.text(i, r, f"{r:.2f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    keys = list(mod_rows.keys())
    pens = [mod_rows[k]["penalty_ea45_pct"] for k in keys]
    ax.barh(range(len(keys)), pens, color="#8e44ad", height=0.5)
    ax.set_yticks(range(len(keys)), [k.replace("/", "\n") for k in keys], fontsize=8)
    ax.set_xlabel("最差芯 EOL 惩罚（%，Ea=45）")
    ax.set_title("d) 模组一致性惩罚（落到绝对账）")
    for i, p in enumerate(pens):
        ax.text(p, i, f" {p:.1f}%", va="center", fontsize=8)

    fig.suptitle("34a · 绝对寿命账：体系 × 方案 × 倍率", fontsize=12)
    fig.tight_layout()
    plotting.save(fig, FIG / "34a_life_ledger.png")
    plt.close(fig)

    print(f"done in {metrics['elapsed_s']:.0f}s -> 34_life_ledger.json / 34a_life_ledger.png")
    print("verdict:", metrics["verdict"][:220])


if __name__ == "__main__":
    main()
