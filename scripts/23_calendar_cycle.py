"""23 · 日历-循环老化定量分解：0.5C 反快之谜 & 寿命比口径修正。

两问两答：
1. **scripts/19 的负结果（0.5C 每 EFC 衰减快于 1.0C）里日历占多少？**
   实测每 EFC **壁钟时长**比只有 1.34×（充电+静置稀释了放电 2×），组间
   每 EFC 衰减比 1.235 → 闭式解 s_cal（``two_group_share``，定义在快组
   EFC 时长上）+ 曲线拟合细化（f = a_i·N^b + k_cal·t^p，b 固定）。
2. **日历-循环分解会不会修正 C5 寿命比（4.7×）？**
   旧损伤口径 D=∫arrh·dt 是纯「热秒数」——给高倍率 EFC 打了日历式的
   时长折扣；通量损伤每 EFC 应大致恒定。修正乘子（calendar_bracket，
   s_cal 先 ``rebase_share`` 到场景参考时长——CS2 每循环 3–4 h 与场景
   0.3–1 h/EFC 不同量纲，不 rebase 会错搬份额）：
       d_total = d_old · [s + (1−s)·t_ref/t_efc]
   高倍率端乘子 >1 → 方案间寿命比向 1 收缩。收缩多少是本脚本的数字。

口径声明：
- s_cal 来自 CS2（1.1 Ah LFP），迁移到基准电芯（NMC 体系）是**跨体系
  近似**，敏感性随 s_cal 68% CI 与 p 档给出；
- 驻车段（实车占比大）只放大日历份额（rest 敏感性表：驻车时长 κ×t_ref
  施加于两方案的每 EFC 日历项）；
- p=0.5 主口径（SEI √t）；p=0.75 敏感性。

产出：
  results/metrics/23_calendar_cycle.json
  results/figures/23a_calendar_cycle.png
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal import plotting  # noqa: E402
from bmthermal.calendar import (  # noqa: E402
    bootstrap_two_group_share,
    calendar_bracket,
    calibrate_from_share,
    fit_decomposition,
    rebase_share,
    two_group_share,
)
from bmthermal.cs2_aging import analyze_cell, extract_cell, list_cs2_cells  # noqa: E402
from bmthermal.feedback import FeedbackParams, pair_life_ratio  # noqa: E402

OUT = ROOT / "results"
MET = OUT / "metrics"
CACHE = OUT / "cache"
MET.mkdir(parents=True, exist_ok=True)

COHORT = ["CS2_33", "CS2_34", "CS2_7", "CS2_35", "CS2_36", "CS2_37", "CS2_38"]
SLOW = ["CS2_33", "CS2_34", "CS2_7"]            # ~0.5C
FAST = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]  # ~1.0C
P_CAL = 0.5
P_SENS = 0.75


def load_18():
    spec = importlib.util.spec_from_file_location("s18", ROOT / "scripts" / "18_aging_feedback.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def per_cycle_dt_s(df) -> float:
    """段内逐循环壁钟时长中位数（跨段 reset 处剔除）。"""
    t = df["t_end_s"].to_numpy(float)
    mem = df["member"].to_numpy()
    dts = []
    for m in np.unique(mem):
        seg = t[mem == m]
        d = np.diff(seg)
        d = d[(d > 10) & (d < 1e5)]
        if d.size:
            dts.append(np.median(d))
    return float(np.median(dts)) if dts else np.nan


def main() -> None:
    t0 = time.time()

    # ---- A) CS2 分解 ----
    print("[1/3] CS2 逐循环数据 ...")
    cells: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    fade_rate: dict[str, float] = {}
    for info in list_cs2_cells():
        cid = info["cell"]
        if cid not in COHORT:
            continue
        df = extract_cell(info["path"], cache_dir=str(CACHE))
        rep = analyze_cell(df)
        n = df["cycle"].to_numpy(float)
        cap = df["cap_ah"].to_numpy(float)
        f = 1.0 - cap / rep["cap0_ah"]
        dt_s = per_cycle_dt_s(df)
        cells[cid] = (n, f, dt_s)
        fade_rate[cid] = rep["fade_rate_per_cycle"]
        print(f"   {cid}: n={len(df)} dt={dt_s/3600:.2f} h/cyc  "
              f"fade_rate={rep['fade_rate_per_cycle']:.2e}/cyc")

    dt_slow = float(np.median([cells[c][2] for c in SLOW]))
    dt_fast = float(np.median([cells[c][2] for c in FAST]))
    t_ratio = dt_slow / dt_fast
    fr_slow = float(np.mean([fade_rate[c] for c in SLOW]))
    fr_fast = float(np.mean([fade_rate[c] for c in FAST]))
    s_closed = two_group_share(fr_slow, fr_fast, t_ratio)     # 定义在 dt_fast
    print(f"   两组闭式解: 壁钟 t_ratio={t_ratio:.2f}（放电时长 ~2× 被充电/静置稀释）"
          f"  r={fr_slow/fr_fast:.3f}  s_cal@cs2={s_closed:.3f}")

    # 主 CI：闭式解按电芯 bootstrap（形状拟合 CI 含 0，不能作主估计）
    boot = bootstrap_two_group_share(fade_rate, SLOW, FAST, t_ratio, n_boot=2000)
    print(f"   闭式 bootstrap: s_cal={boot['median']:.3f} "
          f"[{boot['lo68']:.3f}, {boot['hi68']:.3f}]")

    # 形状拟合（弱佐证）：b 固定 + 共享振幅
    fit5 = fit_decomposition(cells, p_cal=P_CAL)
    fit75 = fit_decomposition(cells, p_cal=P_SENS)
    s_fit = fit5.share_at.get(0.10, np.nan)
    print(f"   形状拟合（弱）: b={fit5.b:.2f}  s_cal@f=0.1={s_fit:.3f}  "
          f"r²={fit5.r2:.3f}  k_cal={fit5.k_cal:.2e}（形状不足以独立识别，CI 含 0）")

    # 闭式校准的分解（图与分解表用这个——不是拟合）
    b_cal = fit5.b
    a_cal, k_cal = calibrate_from_share(s_closed, f_work=0.10, n_work=500.0,
                                       dt_s=dt_fast, b=b_cal, p_cal=P_CAL)
    print(f"   闭式校准分解: a={a_cal:.4f}  k_cal={k_cal:.3e} (N=500, f=0.1 工作点)")

    decomp_rows = []
    for group, names, dt in (("0.5C", SLOW, dt_slow), ("1.0C", FAST, dt_fast)):
        per_efc = float(np.mean([fade_rate[c] for c in names]))
        f_c = a_cal * 500.0**b_cal
        f_k = k_cal * (500.0 * dt)**P_CAL
        decomp_rows.append({
            "group": group, "n_cells": len(names), "dt_h_per_cycle": dt / 3600,
            "fade_per_efc_mean": per_efc,
            "f_cyc_at_N500": f_c, "f_cal_at_N500": f_k,
            "share_at_N500": f_k / (f_c + f_k),
        })

    # ---- B) C5 寿命比修正 ----
    print("[2/3] feedback 闭环 × 日历括号 ...")
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
    print(f"   场景 EFC 时长: 1C={t_ref/3600:.2f} h  3C={spans[3.0]/3600:.2f} h  "
          f"5C={spans[5.0]/3600:.2f} h（CS2 每循环 {dt_fast/3600:.1f}–{dt_slow/3600:.1f} h）")

    # 主口径：闭式 s_cal rebase 到场景 1C 参考时长
    s_main = rebase_share(s_closed, t_ref, dt_fast)
    print(f"   s_cal: cs2@快组={s_closed:.3f} → rebase 到 1C 场景 = {s_main:.3f}")

    def bracket(c: float, s: float) -> float:
        return calendar_bracket(s, spans[c], t_ref)

    brackets = {c: bracket(c, s_main) for c in (1.0, 3.0, 5.0)}
    print(f"   括号（主口径）: 1C={brackets[1.0]:.3f}  3C={brackets[3.0]:.3f}  "
          f"5C={brackets[5.0]:.3f}")

    def d_fn(c, key, br):
        fn_old = s18.d_cycle_factory(c, key, params, cache, fn_cache)
        return lambda u: fn_old(u) * br

    pair_old = pair_life_ratio(
        s18.d_cycle_factory(3.0, "cold_plate", params, cache, fn_cache),
        s18.d_cycle_factory(1.0, "natural", params, cache, fn_cache),
        params)
    pair_new = pair_life_ratio(
        d_fn(3.0, "cold_plate", brackets[3.0]),
        d_fn(1.0, "natural", brackets[1.0]),
        params)
    print(f"   C5 闭环比: 旧 {pair_old['closed_loop_ratio']:.3f} → 新 {pair_new['closed_loop_ratio']:.3f}")

    # 敏感性：s 来源 × rebase
    sens_rows = []
    s_sources = [("closed_form", s_closed), ("fit", s_fit),
                 ("boot_lo", boot["lo68"]), ("boot_hi", boot["hi68"])]
    for s_name, s_cs2 in s_sources:
        if not np.isfinite(s_cs2) or s_cs2 <= 0:
            continue
        s_r = rebase_share(float(s_cs2), t_ref, dt_fast)
        pr = pair_life_ratio(
            d_fn(3.0, "cold_plate", bracket(3.0, s_r)),
            d_fn(1.0, "natural", bracket(1.0, s_r)),
            params)
        sens_rows.append({
            "s_source": s_name, "s_cal_cs2": float(s_cs2), "s_cal_rebased": s_r,
            "bracket_3c": bracket(3.0, s_r),
            "closed_loop_ratio": pr["closed_loop_ratio"],
            "drift_vs_old_pct": (pr["closed_loop_ratio"] / pair_old["closed_loop_ratio"] - 1) * 100,
        })
    s_fit75 = fit75.share_at.get(0.10, np.nan)
    if np.isfinite(s_fit75) and s_fit75 > 0:
        s_r = rebase_share(float(s_fit75), t_ref, dt_fast)
        pr = pair_life_ratio(
            d_fn(3.0, "cold_plate", bracket(3.0, s_r)),
            d_fn(1.0, "natural", bracket(1.0, s_r)),
            params)
        sens_rows.append({
            "s_source": "fit_p0.75", "s_cal_cs2": float(s_fit75), "s_cal_rebased": s_r,
            "bracket_3c": bracket(3.0, s_r),
            "closed_loop_ratio": pr["closed_loop_ratio"],
            "drift_vs_old_pct": (pr["closed_loop_ratio"] / pair_old["closed_loop_ratio"] - 1) * 100,
        })

    # 每 EFC 损伤 vs 倍率（旧 vs 新）
    rate_rows = []
    for c, key in ((1.0, "natural"), (3.0, "cold_plate"), (5.0, "cold_plate")):
        fn_old = s18.d_cycle_factory(c, key, params, cache, fn_cache)
        d_old = float(fn_old(0.0))
        rate_rows.append({
            "case": f"{c:g}C_{key}", "c_rate": c,
            "d_old_per_efc": d_old, "bracket": brackets[c],
            "d_new_per_efc": d_old * brackets[c],
        })

    # rest 敏感性：每 EFC 驻车 κ×t_ref（两方案同绝对量），日历项 ∝ (t_s + κ·t_ref)
    rest_rows = []
    for kappa in (0.0, 1.0, 3.0):
        def br_k(c, s=s_main, kappa=kappa):
            ts = spans[c]
            # d_new/d_old = [(1−s) + s·(ts+κ·t_ref)/t_ref]·t_ref/ts
            return ((1.0 - s) + s * (ts + kappa * t_ref) / t_ref) * t_ref / ts
        pr = pair_life_ratio(
            d_fn(3.0, "cold_plate", br_k(3.0)),
            d_fn(1.0, "natural", br_k(1.0)),
            params)
        rest_rows.append({
            "rest_over_ref_efc": kappa,
            "bracket_1c": br_k(1.0), "bracket_3c": br_k(3.0),
            "closed_loop_ratio": pr["closed_loop_ratio"],
        })

    ratio_new = pair_new["closed_loop_ratio"]
    drift_main = (ratio_new / pair_old["closed_loop_ratio"] - 1) * 100
    ci_rows = [r for r in sens_rows if r["s_source"] in ("closed_form", "boot_lo", "boot_hi")]
    s_lo = min(r["closed_loop_ratio"] for r in ci_rows)
    s_hi = max(r["closed_loop_ratio"] for r in ci_rows)
    verdict = (
        f"0.5C 反快之谜定量：壁钟时长比 {t_ratio:.2f}（放电 2× 被充电/静置稀释）、"
        f"每 EFC 衰减比 {fr_slow/fr_fast:.3f} → 日历份额 s_cal={s_closed:.2f}"
        f"（CS2 快组时长口径，闭式解；bootstrap [{boot['lo68']:.2f}, {boot['hi68']:.2f}]；"
        f"rebase 到场景 1C 后 {s_main:.2f}）"
        f"——反快主因是日历混入，组间非热差异次要。"
        f"形状拟合**不能**独立识别日历项（k_cal→0，CI 含 0，7 芯噪声下曲线形状"
        f"信息不足；可识别的是组间水平差）——负结果已落盘。"
        f"口径修正后 C5 闭环比 {pair_old['closed_loop_ratio']:.2f}× → "
        f"{ratio_new:.2f}×（{drift_main:+.0f}%；3C 端括号 {brackets[3.0]:.2f}——"
        f"旧『热秒数』口径给高倍率 EFC 打了日历式折扣，通量损伤被低估）；"
        f"闭式 CI 下区间 [{s_lo:.2f}, {s_hi:.2f}]×；"
        f"驻车进一步压缩（κ=3 时 {rest_rows[-1]['closed_loop_ratio']:.2f}×）。"
        f"结论：『4–5×』是**日历机制缺失下的上界**；修正后主口径 "
        f"{ratio_new:.1f}×，闭式 CI 下 [{s_lo:.1f}, {s_hi:.1f}]×"
        f"（跨体系迁移 CS2 LFP→基准 NMC 是主不确定源）。"
    )
    print("判词:", verdict)

    payload = {
        "cs2_decomposition": {
            "t_ratio_slow_fast": t_ratio, "dt_h_slow": dt_slow / 3600, "dt_h_fast": dt_fast / 3600,
            "fade_per_efc_slow": fr_slow, "fade_per_efc_fast": fr_fast,
            "fade_ratio": fr_slow / fr_fast,
            "s_cal_closed_form_at_cs2_fast": s_closed,
            "s_cal_closed_form_bootstrap": boot,
            "s_cal_rebased_to_1c_scenario": s_main,
            "calibrated_split": {"b": b_cal, "a": a_cal, "k_cal": k_cal,
                                 "work_point": {"N": 500.0, "f": 0.10}},
            "shape_fit_p0.5": {"b": fit5.b, "k_cal": fit5.k_cal, "r2": fit5.r2,
                               "share_at": fit5.share_at,
                               "negative_result": "k_cal→0：形状不足以独立识别日历项，CI 含 0；主估计用闭式解"},
            "shape_fit_p0.75": {"b": fit75.b, "k_cal": fit75.k_cal, "share_at": fit75.share_at},
            "per_group": decomp_rows,
        },
        "life_ratio_correction": {
            "efc_span_s": {f"{c:g}": spans[c] for c in (1.0, 3.0, 5.0)},
            "brackets": {f"{c:g}": brackets[c] for c in (1.0, 3.0, 5.0)},
            "c5_pair_old": pair_old, "c5_pair_new": pair_new,
            "sensitivity": sens_rows,
            "rest_sensitivity": rest_rows,
            "per_efc_damage": rate_rows,
        },
        "assumptions": [
            "s_cal 跨体系迁移（CS2 LFP → 基准电芯）近似，CI 已给出",
            "两机制共用 Ea（温度标度不分机制）",
            "p_cal=0.5 主口径，0.75 敏感性",
            "s_cal 定义在 CS2 快组 EFC 时长，应用前 rebase 到场景参考时长",
            "驻车段按每 EFC κ×t_ref 绝对量施加于两方案",
        ],
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (MET / "23_calendar_cycle.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    # ---- 图 23a ----
    plotting.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]

    def _smooth(x, win=25):
        if x.size < win:
            return x
        k = np.ones(win) / win
        pad = np.pad(x, (win // 2, win - 1 - win // 2), mode="edge")
        return np.convolve(pad, k, mode="valid")[: x.size]

    for cid in SLOW:
        n, f, dt = cells[cid]
        ax.plot(n, _smooth(f) * 100, color="#B45309", alpha=0.35, lw=0.8)
    for cid in FAST:
        n, f, dt = cells[cid]
        ax.plot(n, _smooth(f) * 100, color="#0E7490", alpha=0.35, lw=0.8)
    grid = np.geomspace(50, 2000, 60)
    for color, dt, names, lab in (("#B45309", dt_slow, SLOW, "0.5C"),
                                  ("#0E7490", dt_fast, FAST, "1.0C")):
        f_c = a_cal * grid**b_cal
        f_k = k_cal * (grid * dt)**P_CAL
        ax.plot(grid, (f_c + f_k) * 100, color=color, lw=2, label=f"{lab} 校准分解")
        ax.fill_between(grid, f_c * 100, (f_c + f_k) * 100, color=color, alpha=0.15)
        ax.plot(grid, f_c * 100, color=color, lw=1, ls="--")
    ax.plot([], [], color="gray", ls="--", lw=1, label="通量分量 a·N^b")
    ax.plot([], [], color="gray", alpha=0.3, lw=6, label="日历分量 k·t^p")
    ax.set_xscale("log")
    ax.set_ylim(0, 30)
    ax.set_xlabel("循环数 N")
    ax.set_ylabel("衰减 %")
    ax.set_title("a) CS2 衰减分解（闭式校准，b 固定；细线为平滑实测）")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]
    for color, dt, names, lab in (("#B45309", dt_slow, SLOW, "0.5C 组 dt"),
                                  ("#0E7490", dt_fast, FAST, "1.0C 组 dt")):
        f_c = a_cal * grid**b_cal
        f_k = k_cal * (grid * dt)**P_CAL
        ax.plot(grid, f_k / (f_c + f_k), color=color, lw=2, label=lab)
    ax.axhline(s_closed, color="k", ls=":", lw=1, label=f"闭式解（CS2 快组）{s_closed:.2f}")
    ax.axhline(s_main, color="#7C3AED", ls="-.", lw=1.2,
               label=f"rebase 到 1C 场景 {s_main:.2f}")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("循环数 N")
    ax.set_ylabel("日历份额 s_cal(N)")
    ax.set_title("b) 日历份额与参考时长 rebase")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    # 图里只画主口径与闭式 CI（形状拟合行留在 JSON）
    show = [r for r in sens_rows if r["s_source"] in ("closed_form", "boot_lo", "boot_hi")]
    labels = ["旧口径\n(热秒数)"] + [r["s_source"] for r in show]
    vals = [pair_old["closed_loop_ratio"]] + [r["closed_loop_ratio"] for r in show]
    colors = ["#6B7280"] + ["#1D4ED8"] * len(show)
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(pair_old["open_loop_ratio"], color="#B45309", ls="--", lw=1.2,
               label=f"开环比 {pair_old['open_loop_ratio']:.2f}")
    ax.set_xticks(range(len(vals)), labels, fontsize=7)
    ax.set_ylabel("C5 寿命比（3C液冷/1C自然）")
    ax.set_title("c) 日历-循环修正后的寿命比")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    cs = [r["c_rate"] for r in rate_rows]
    w = 0.3
    ax.bar(np.array(cs) - w / 2, [r["d_old_per_efc"] for r in rate_rows], w,
           label="旧（热秒数）", color="#6B7280")
    ax.bar(np.array(cs) + w / 2, [r["d_new_per_efc"] for r in rate_rows], w,
           label="修正后（主口径）", color="#1D4ED8")
    ax.set_xlabel("倍率 C")
    ax.set_ylabel("每 EFC 损伤（任意单位）")
    ax.set_title("d) 修正主要作用于高倍率端")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("23 · 日历-循环老化分解与寿命比口径修正", fontsize=13)
    fig.tight_layout()
    plotting.save(fig, "23a_calendar_cycle.png")
    print(f"[落盘] 23_calendar_cycle.json / 23a_calendar_cycle.png  ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
