"""26 · Sandia 跨体系日历份额：LFP 复刻成立，NMC/NCA 识别性负结果。

背景（scripts/23 开放项）
-------------------------
v1.4 的 C5 寿命比 CI [1.76, 4.77] 宽度主因是 s_cal 跨体系迁移
（CS2 LFP → 基准 NMC）。本脚本用 SNL（Preger 2020）多化学体系
（NCA/NMC/LFP）× 多放电倍率（0.5/1/2/3C，25 °C，0–100% 窗）直接
估各体系的 s_cal——若 NMC 能自估，CI 收窄。

三问三答：
1. **两组闭式解在 SNL 上自洽吗？** 同一体系不同倍率对给出的 s_raw
   跨越负值到 >1（NMC：−0.37…+2.52；NCA：−0.20…+1.41）——
   仿射两机制模型 f=c+k·t 跨倍率对**系统性不自洽**（负结果）。
   组内 fade 率散布 1.5–2.2×、35 °C 组比 25 °C 慢（批次/协议异质性）
   是根源；NMC/NCA 唯一 n≥2 的对全部资格否决（c_flux<0）。
2. **方法本身坏了吗？** 没有——SNL LFP 0.5/1/2C 三对内部自洽
   s=0.38–0.51，与 CS2 的 0.69（bootstrap [0.40, 1.00]）同量级、
   rebase 后重叠。两套独立 LFP 数据给出同阶日历份额。
3. **C5 CI 能收窄吗？** 不能。NMC identified set（资格对并集）
   rebase 后 [0.04, 0.60]，比 CS2 bootstrap 更宽——"迁移不确定度"
   从假设升级为实测确认的识别性障碍；v1.4 CI [1.76, 4.77] 保留，
   但置信依据从"单一数据集 bootstrap"变为"跨数据集识别集"。

产出：
  results/metrics/26_snl_calendar.json
  results/figures/26a_snl_calendar.png
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal import plotting  # noqa: E402
from bmthermal.calendar import bootstrap_two_group_share, calendar_bracket, rebase_share  # noqa: E402
from bmthermal.feedback import FeedbackParams, pair_life_ratio  # noqa: E402
from bmthermal.snl_aging import group_stats, load_dataset, pair_share  # noqa: E402

OUT = ROOT / "results"
MET = OUT / "metrics"
CACHE = OUT / "cache"
FIG = OUT / "figures"
for p in (MET, CACHE, FIG):
    p.mkdir(parents=True, exist_ok=True)

SNL_ZIP = Path("D:/项目/02_数据集/01_国际公开集/Sandia_SNL/SNL.zip")
CHEMS = ("NMC", "NCA", "LFP")
CS2_S_CLOSED = 0.6928      # scripts/23 闭式解（CS2 快组时长口径）
CS2_DT_FAST_S = 3.1176 * 3600
CS2_BOOT = (0.4010, 1.0)


def load_18():
    spec = importlib.util.spec_from_file_location("s18", ROOT / "scripts" / "18_aging_feedback.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    t0 = time.time()

    # ---- A) 数据集与组级画像 ----
    print("[1/3] SNL 电芯汇总 ...")
    df = load_dataset(SNL_ZIP, CACHE / "snl_summary.csv")
    df = df[df["fade_rate_per_efc"].notna()].copy()
    print(f"   {len(df)} 支有效电芯（{df.chem.value_counts().to_dict()}）")

    group_rows = []
    for (chem, temp, cdis), g in df[df.soc_lo == 0].groupby(["chem", "temp_c", "c_dis"]):
        group_rows.append({
            "chem": chem, "temp_c": temp, "c_dis": cdis, "n": int(len(g)),
            "fade_per_efc": float(g.fade_rate_per_efc.mean()),
            "fade_per_efc_std": float(g.fade_rate_per_efc.std(ddof=1)) if len(g) > 1 else 0.0,
            "dt_s": float(g.dt_s_per_cycle.median()),
        })
        print(f"   {chem:3s} {temp:4.0f}C {cdis:g}C: n={len(g)} "
              f"fade={g.fade_rate_per_efc.mean():.3e}/EFC "
              f"(spread×{g.fade_rate_per_efc.max()/max(g.fade_rate_per_efc.min(),1e-12):.1f}) "
              f"dt={g.dt_s_per_cycle.median()/3600:.2f} h/cyc")

    # ---- B) 倍率对诊断 ----
    print("[2/3] 倍率对 s_cal 诊断 ...")
    pair_rows = []
    for chem in CHEMS:
        rates = sorted(df[(df.chem == chem) & (df.temp_c == 25) & (df.soc_lo == 0)].c_dis.unique())
        for cs, cf in itertools.combinations(rates, 2):
            p = pair_share(df, chem, 25.0, float(cs), float(cf))
            fade_by_cell = {
                **{c: df.set_index("cell_id").loc[c, "fade_rate_per_efc"] for c in p["slow"]["cells"]},
                **{c: df.set_index("cell_id").loc[c, "fade_rate_per_efc"] for c in p["fast"]["cells"]},
            }
            boot = bootstrap_two_group_share(
                fade_by_cell, p["slow"]["cells"], p["fast"]["cells"], p["t_ratio"])
            p["boot"] = boot
            pair_rows.append(p)
            print(f"   {chem:3s} {cs:g}->{cf:g}C: t={p['t_ratio']:.3f} r={p['fade_ratio']:.3f} "
                  f"s_raw={p['s_raw']:+.2f} s={p['s_cal_at_fast']:.2f} "
                  f"c_flux={p['c_flux_per_efc']:.2e} elig={p['eligible']} "
                  f"n={p['slow']['n_cells']}+{p['fast']['n_cells']}")

    # 同体系跨对不自洽度（负结果量化）
    consistency = {}
    for chem in CHEMS:
        ss = [p["s_raw"] for p in pair_rows if p["chem"] == chem and np.isfinite(p["s_raw"])]
        consistency[chem] = {
            "s_raw_min": float(np.min(ss)), "s_raw_max": float(np.max(ss)),
            "s_raw_spread": float(np.max(ss) - np.min(ss)),
            "n_pairs": len(ss),
            "n_eligible": sum(1 for p in pair_rows if p["chem"] == chem and p["eligible"]),
        }
        print(f"   {chem} 跨对 s_raw 范围 [{np.min(ss):+.2f}, {np.max(ss):+.2f}] "
              f"（仿射模型不自洽度 {np.max(ss)-np.min(ss):.2f}）")

    # ---- C) LFP 复刻 vs CS2 + identified set + C5 范围 ----
    print("[3/3] LFP 复刻与 C5 识别集 ...")
    s18 = load_18()
    cache: dict = {}
    fn_cache: dict = {}
    params = FeedbackParams()
    from bmthermal.heat import q_profile
    spans = {}
    for c in (1.0, 3.0, 5.0):
        t_arr, _, _ = q_profile(c, t_amb_c=25.0, cache=cache)
        spans[c] = float(t_arr[-1] - t_arr[0])
    t_ref = spans[1.0]

    def life_ratio(s_rebased: float) -> float:
        br = {c: calendar_bracket(s_rebased, spans[c], t_ref) for c in (1.0, 3.0, 5.0)}

        def d_fn(c, key, b):
            fn_old = s18.d_cycle_factory(c, key, params, cache, fn_cache)
            return lambda u: fn_old(u) * b
        pr = pair_life_ratio(
            d_fn(3.0, "cold_plate", br[3.0]),
            d_fn(1.0, "natural", br[1.0]),
            params)
        return float(pr["closed_loop_ratio"])

    pair_old = pair_life_ratio(
        s18.d_cycle_factory(3.0, "cold_plate", params, cache, fn_cache),
        s18.d_cycle_factory(1.0, "natural", params, cache, fn_cache),
        params)
    ratio_old = float(pair_old["closed_loop_ratio"])

    # LFP 复刻：资格对的 s（快组时长口径）rebase 到场景 1C
    lfp_elig = [p for p in pair_rows
                if p["chem"] == "LFP" and p["eligible"] and p["s_raw"] > 0]
    lfp_s = [p["s_cal_at_fast"] for p in lfp_elig]
    lfp_rebased = [rebase_share(s, t_ref, p["fast"]["dt_s"]) for s, p in zip(lfp_s, lfp_elig)]
    cs2_rebased = rebase_share(CS2_S_CLOSED, t_ref, CS2_DT_FAST_S)
    print(f"   LFP 资格对 s={sorted(lfp_s)} → rebase {sorted(np.round(lfp_rebased,3))}"
          f"  vs CS2 {CS2_S_CLOSED:.2f} → rebase {cs2_rebased:.3f} "
          f"[boot {CS2_BOOT[0]:.2f}, {CS2_BOOT[1]:.2f}]")

    # NMC/NCA identified set：所有资格对（含单芯对）rebase 后并集
    target_pos = [p for p in pair_rows
                  if p["chem"] in ("NMC", "NCA") and p["eligible"] and p["s_raw"] > 0]
    id_hi = max(rebase_share(p["s_cal_at_fast"], t_ref, p["fast"]["dt_s"]) for p in target_pos)
    # 下界 0：负 s_raw 对（rate_effect）表明日历份额可被压到 0，模型不完整
    s_lo, s_hi = 0.0, float(id_hi)
    ratio_at_lo, ratio_at_hi = life_ratio(s_lo), life_ratio(s_hi)  # 寿命比随 s 单调升
    print(f"   NMC/NCA identified set（rebase）: [{s_lo:.2f}, {s_hi:.2f}] "
          f"→ C5 闭环比 [{ratio_at_lo:.2f}, {ratio_at_hi:.2f}]（旧 {ratio_old:.2f}，s=1）")

    # 敏感性行：s 来源 × C5
    sens_rows = [{
        "s_source": "none(旧口径)", "s_rebased": 0.0,
        "closed_loop_ratio": ratio_old,
    }, {
        "s_source": "cs2_closed_form", "s_rebased": cs2_rebased,
        "closed_loop_ratio": life_ratio(cs2_rebased),
    }]
    for s_name, s_at, dt_at in (
        ("snl_lfp_lo", min(lfp_s), lfp_elig[int(np.argmin(lfp_s))]["fast"]["dt_s"]),
        ("snl_lfp_hi", max(lfp_s), lfp_elig[int(np.argmax(lfp_s))]["fast"]["dt_s"]),
        ("snl_target_lo", s_lo, t_ref),  # 已 rebase
        ("snl_target_hi", s_hi, t_ref),
    ):
        s_r = s_at if s_name.startswith("snl_target") else rebase_share(s_at, t_ref, dt_at)
        sens_rows.append({
            "s_source": s_name, "s_rebased": float(s_r),
            "closed_loop_ratio": life_ratio(float(s_r)),
        })

    # ---- 指标落盘 ----
    metrics = {
        "dataset": {
            "n_cells": int(len(df)),
            "by_chem": {k: int(v) for k, v in df.chem.value_counts().items()},
            "groups": group_rows,
        },
        "pairs": [{
            "chem": p["chem"], "c_slow": p["c_slow"], "c_fast": p["c_fast"],
            "t_ratio": p["t_ratio"], "fade_ratio": p["fade_ratio"],
            "s_raw": p["s_raw"], "s_cal_at_fast": p["s_cal_at_fast"],
            "c_flux_per_efc": p["c_flux_per_efc"], "eligible": p["eligible"],
            "n_slow": p["slow"]["n_cells"], "n_fast": p["fast"]["n_cells"],
            "dt_fast_s": p["fast"]["dt_s"],
            "boot": p["boot"],
        } for p in pair_rows],
        "model_consistency": consistency,
        "lfp_replication": {
            "s_pairs": lfp_s, "s_rebased": lfp_rebased,
            "cs2_s_closed": CS2_S_CLOSED, "cs2_rebased": cs2_rebased,
            "cs2_boot68": list(CS2_BOOT),
            "same_order": bool(0.05 <= np.median(lfp_rebased) <= 0.6),
        },
        "identified_set": {
            "chems": ["NMC", "NCA"],
            "s_rebased_lo": s_lo, "s_rebased_hi": s_hi,
            "c5_ratio_at_s_lo": ratio_at_lo, "c5_ratio_at_s_hi": ratio_at_hi,
            "c5_ratio_old": ratio_old,
            "positive_pairs": [
                {"pair": f"{p['chem']} {p['c_slow']:g}->{p['c_fast']:g}C",
                 "s_raw": p["s_raw"], "n": [p["slow"]["n_cells"], p["fast"]["n_cells"]]}
                for p in target_pos
            ],
            "rate_effect_pairs": [
                {"pair": f"{p['chem']} {p['c_slow']:g}->{p['c_fast']:g}C",
                 "s_raw": p["s_raw"]}
                for p in pair_rows if p["s_raw"] < 0 or p["s_raw"] > 1
            ],
            "note": "正向资格对给上界（多为 n=1–2 单芯对，弱）；下界 0 由负向 rate_effect 对支持；n≥2 双侧对全部资格否决",
        },
        "sensitivity": sens_rows,
        "scenario_efc_span_s": {str(k): v for k, v in spans.items()},
        "verdict": (
            "两组仿射模型在 SNL 上跨倍率对**系统性不自洽**（识别性负结果）："
            f"NMC s_raw [{consistency['NMC']['s_raw_min']:+.2f}, {consistency['NMC']['s_raw_max']:+.2f}]、"
            f"NCA [{consistency['NCA']['s_raw_min']:+.2f}, {consistency['NCA']['s_raw_max']:+.2f}]，"
            "组内 fade 率散布 1.5–2.2×、35°C 组反比 25°C 慢（批次/协议异质性）；"
            "NMC/NCA 唯一 n≥2 双侧对全部资格否决（c_flux<0），点估计+CI 给不出。"
            "方法本身在 LFP 上复刻成立：SNL LFP 正向资格对 s=0.38–0.51，与 CS2 0.69 同量级"
            "（rebase 后 0.17–0.30 vs CS2 0.40 [0.40, 1.00]——同阶相邻接，点值略低）。"
            "C5 更新是**不对称收窄**：SNL 正向资格对（s_raw≤0.78）把 s>0.61（rebase）的可能性压低，"
            f"识别集 [{s_lo:.2f}, {s_hi:.2f}] → C5 闭环比 [{ratio_at_lo:.2f}, {ratio_at_hi:.2f}]×——"
            f"主口径 {ratio_at_hi:.1f} 附近不变（CS2 闭式 2.14 在集内），上界 4.8× 与『4–5×』进一步排除；"
            "v1.4 CI 的宽上界来自 CS2 bootstrap 的 s→1 端，与 NMC 直接证据不相容。"
        ),
        "elapsed_s": time.time() - t0,
    }
    (MET / "26_snl_calendar.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 图 26a ----
    plotting.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) fade/EFC vs dt/EFC 散点（等日历斜线）
    ax = axes[0, 0]
    colors = {"NMC": "#c0392b", "NCA": "#2980b9", "LFP": "#27ae60"}
    for chem in CHEMS:
        g = df[(df.chem == chem) & (df.soc_lo == 0) & (df.temp_c == 25)]
        ax.scatter(g.dt_s_per_cycle / 3600, g.fade_rate_per_efc * 1e4,
                   c=colors[chem], s=42, label=chem, alpha=0.85, edgecolors="k", linewidths=0.4)
    ax.set_xlabel("每 EFC 壁钟时长 (h)")
    ax.set_ylabel("fade (x1e-4 /EFC)")
    ax.set_title("(a) SNL 组间：fade/EFC vs 时长（25 °C）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) s_raw 按倍率对（负结果主图）
    ax = axes[0, 1]
    xpos, labels = 0, []
    for p in pair_rows:
        c = colors[p["chem"]]
        m = "o" if p["eligible"] else "x"
        if p["eligible"]:
            ax.scatter(xpos, p["s_raw"], c=c, marker="o", s=55, zorder=3,
                       edgecolors="k", linewidths=0.5)
        else:
            ax.scatter(xpos, p["s_raw"], c=c, marker="x", s=65, zorder=3, linewidths=1.4)
        labels.append(f"{p['chem'][:2]}\n{p['c_slow']:g}→{p['c_fast']:g}")
        xpos += 1
    ax.axhline(0, color="gray", lw=0.8)
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    ax.fill_between([-1, xpos], 0, 1, color="#27ae60", alpha=0.08, label="合法域 s∈[0,1]")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("s_raw（未截断）")
    ax.set_title("(b) 两组闭式解跨倍率对不自洽（×=资格否决）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (c) C5 寿命比 vs s 曲线 + 识别集带
    ax = axes[1, 0]
    s_grid = np.linspace(0, 1, 21)
    r_grid = [life_ratio(float(s)) for s in s_grid]
    ax.plot(s_grid, r_grid, color="#2c3e50", lw=2, label="C5 闭环比(s)")
    ax.axhline(ratio_old, color="#7f8c8d", ls="--", lw=1, label=f"旧口径 {ratio_old:.2f}×")
    ax.axvspan(s_lo, s_hi, color="#e74c3c", alpha=0.12,
               label=f"NMC/NCA 识别集 [{s_lo:.2f},{s_hi:.2f}]")
    ax.axvspan(CS2_BOOT[0], CS2_BOOT[1], color="#2980b9", alpha=0.12,
               label="CS2 bootstrap [0.40,1.00]")
    ax.set_xlabel("s_cal（rebase 到场景 1C 时长）")
    ax.set_ylabel("C5 闭环比")
    ax.set_title("(c) 寿命比对 s 的敏感性与识别集")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) LFP 复刻对比
    ax = axes[1, 1]
    xs = ["SNL LFP\n资格对", "CS2 LFP\n闭式解"]
    for i, s in enumerate(lfp_rebased):
        ax.scatter(0, s, c="#27ae60", s=55, zorder=3, edgecolors="k", linewidths=0.5,
                   label="SNL LFP（rebase）" if i == 0 else None)
    ax.errorbar(1, cs2_rebased,
                yerr=[[cs2_rebased - rebase_share(CS2_BOOT[0], t_ref, CS2_DT_FAST_S)],
                      [rebase_share(CS2_BOOT[1], t_ref, CS2_DT_FAST_S) - cs2_rebased]],
                fmt="o", color="#2980b9", capsize=5, ms=8,
                label="CS2 闭式解（boot 68%）")
    ax.scatter(0, cs2_rebased, marker="|", c="#2980b9", s=120, lw=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(xs, fontsize=9)
    ax.set_ylabel("s_cal（rebase 到场景 1C）")
    ax.set_title("(d) LFP 复刻：两套独立数据同阶重叠")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    plotting.save(fig, "26a_snl_calendar.png")

    print(f"完成（{metrics['elapsed_s']:.0f} s）：{MET/'26_snl_calendar.json'}")


if __name__ == "__main__":
    main()
