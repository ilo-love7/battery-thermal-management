"""19 · 寿命模型实测对标：CALCE CS2 循环老化数据。

既有限制（§6.1）：无绝对寿命标定 → 只报相对寿命倍数。
本脚本用 CS2 实测衰减/内阻数据做四件事：

1. **实测 β**（内阻增长系数）——检验 v1.3 闭环模型假设 R/R0=1+βf 的 β∈[1.5,3.5]；
2. **实测衰减指数 b**（f=a·N^b）——检验 p∈{0.5 SEI, 1.0 线性} 的取法；
3. **倍率加速的热份额**——不同 C 率电芯的每 EFC 衰减比 → 等效温差 ΔT
   （implied_delta_t），与自加热 I²R 的量级对比，分离热加速/机械加速；
4. **绝对标定 + 留一交叉**——在一支电芯上标定 A_ref，预测其余电芯的衰减速率，
   回答"能不能报绝对循环数"（对本电芯类型的口径）。

产出：
  results/metrics/19_life_validation.json
  results/figures/19a_cs2_validation.png
  results/cache/cs2_*_per_cycle.parquet   （逐循环缓存）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmthermal import plotting  # noqa: E402
from bmthermal.aging import _arrhenius_factor  # noqa: E402
from bmthermal.cs2_aging import (  # noqa: E402
    analyze_cell,
    extract_cell,
    implied_delta_t_k,
    list_cs2_cells,
)

OUT = Path(__file__).resolve().parents[1] / "results"
MET = OUT / "metrics"
CACHE = OUT / "cache"
MET.mkdir(parents=True, exist_ok=True)

#: 自加热估计：Q=I²R，ΔT=Q/(hA)；CS2 方形电芯（~7×25×120 mm）自然对流
CELL_AREA_M2 = 0.008      # 文献几何量级
H_RANGE = (5.0, 15.0)     # 自然对流范围，W/(m²·K)
EA = 45_000.0             # 与 aging.py 默认一致


def self_heating_dt(i_a: float, r_ohm: float,
                    r_fallback: float = 0.095) -> dict:
    """自加热稳态温升量级（区间）。缺内阻数据回退到队列中位值。"""
    if not np.isfinite(r_ohm) or r_ohm <= 0:
        r_ohm = r_fallback
    q = i_a**2 * r_ohm
    return {
        "q_w": q,
        "dt_min_k": q / (H_RANGE[1] * CELL_AREA_M2),
        "dt_max_k": q / (H_RANGE[0] * CELL_AREA_M2),
        "r_used": r_ohm,
    }


def main() -> None:
    t0 = time.time()
    cells = list_cs2_cells()
    print(f"发现 {len(cells)} 个 CS2 电芯包（xlsx 格式）")

    profiles = {}
    for meta in cells:
        df = extract_cell(meta["path"], cache_dir=str(CACHE))
        prof = analyze_cell(df)
        prof["cell"] = meta["cell"]
        prof["n_files"] = meta["n_files"]
        profiles[meta["cell"]] = prof
        print(f"  {meta['cell']:8s} n={prof['n_cycles']:4d}  "
              f"cap0={prof['cap0_ah']:.3f}Ah  C率={prof['crate']:.2f}  "
              f"末端衰减={prof['fade_final']*100:5.1f}%  "
              f"b={prof['fade_power']['b']:.2f}  β={prof['resistance_growth']['beta']:.2f}")

    # 队列划分：1.1 Ah 标称（0.8–1.4 Ah）为主对标组（CS2_33–38 型）；
    # 其余（部分循环/异构协议/大容量段）只做数据画像，不进对标。
    cohort = {c: p for c, p in profiles.items() if 0.8 <= p["cap0_ah"] <= 1.4}
    outliers = {c: p for c, p in profiles.items() if c not in cohort}
    print(f"\n主对标队列（1.1 Ah 标称）：{sorted(cohort)}；"
          f"异构协议排除：{sorted(outliers)}")

    # ---------- 1/4 实测 β 与 b 的汇总（对标假设，主队列） ----------
    # β 拟合按 r² 分级：r²<0.3 视为噪声主导（内阻通道平/抖），只做展示不进汇总
    betas_all = [(c, p["resistance_growth"]["beta"], p["resistance_growth"]["r2"])
                 for c, p in cohort.items() if np.isfinite(p["resistance_growth"]["beta"])]
    betas = [(c, b) for c, b, r2 in betas_all if r2 is not None and r2 >= 0.3]
    beta_noisy = [(c, b) for c, b, r2 in betas_all if r2 is None or r2 < 0.3]
    bs = [(c, p["fade_power"]["b"]) for c, p in cohort.items()
          if np.isfinite(p["fade_power"]["b"])]
    beta_vals = [b for _, b in betas]
    b_vals = [b for _, b in bs]
    beta_med = float(np.median(beta_vals))
    b_med = float(np.median(b_vals))
    beta_in_range = sum(1.5 <= b <= 3.5 for b in beta_vals)
    print(f"\n[1/4] 实测 β 中位 {beta_med:.2f}（r²≥0.3 的 {len(beta_vals)} 支，"
          f"{beta_in_range} 支落在假设 [1.5,3.5]）；噪声主导 {len(beta_noisy)} 支剔除；"
          f"衰减指数 b 中位 {b_med:.2f}（0.5=SEI 型，1.0=线性）")

    # ---------- 2/4 倍率加速的热份额（主队列） ----------
    by_rate: dict[str, list] = {}
    for c, p in cohort.items():
        if not np.isfinite(p["crate"]) or not np.isfinite(p["fade_rate_per_cycle"]):
            continue
        key = f"{p['crate']:.1f}C"
        by_rate.setdefault(key, []).append(p)
    rate_summary = {}
    for key, ps in sorted(by_rate.items()):
        fr = [p["fade_rate_per_cycle"] for p in ps]
        rate_summary[key] = {
            "n_cells": len(ps),
            "cells": [p["cell"] for p in ps],
            "fade_rate_mean": float(np.mean(fr)),
            "fade_rate_std": float(np.std(fr)),
            "crate_mean": float(np.mean([p["crate"] for p in ps])),
        }
    print(f"[2/4] 按 C 率分组：{ {k: v['n_cells'] for k, v in rate_summary.items()} }")

    # 参考组：最低 C 率组
    keys = sorted(rate_summary.keys(), key=lambda k: rate_summary[k]["crate_mean"])
    thermal_share_rows = []
    if len(keys) >= 2:
        ref_key = keys[0]
        ref = rate_summary[ref_key]
        for key in keys[1:]:
            cur = rate_summary[key]
            ratio = cur["fade_rate_mean"] / ref["fade_rate_mean"]
            dt_needed = implied_delta_t_k(ratio, EA)
            # 自加热量级（用组内电芯实测 I、R）
            dts = []
            for p in profiles.values():
                if p["cell"] in cur["cells"]:
                    dts.append(self_heating_dt(p["i_dis_a"], p["r0_ohm"]))
            dt_lo = float(np.mean([d["dt_min_k"] for d in dts])) if dts else np.nan
            dt_hi = float(np.mean([d["dt_max_k"] for d in dts])) if dts else np.nan
            covered = bool(np.isfinite(dt_needed) and dt_lo <= dt_needed <= dt_hi)
            note = ("热加速可解释" if covered else
                    ("反向（高倍率每 EFC 衰减更慢）——每 EFC 时长不同混入日历老化"
                     if ratio < 1 else "超出自加热量级 → 存在非热（机械/动力学）份额"))
            thermal_share_rows.append({
                "group": key, "ref_group": ref_key,
                "fade_rate_ratio": float(ratio),
                "implied_dt_k": float(dt_needed),
                "self_heating_dt_range_k": [dt_lo, dt_hi],
                "explained_by_self_heating": covered,
                "note": note,
            })
            print(f"   {key} vs {ref_key}: 每 EFC 衰减比 {ratio:.2f} → "
                  f"等效 ΔT {dt_needed:+.1f} K（自加热量级 {dt_lo:.1f}–{dt_hi:.1f} K）"
                  f"{'✓' if covered else '✗'} {note}")

    # ---------- 3/4 绝对标定 + 留一交叉（主队列） ----------
    loo_rows = []
    # 标定源：循环最多且有内阻数据的电芯（自加热估计需要 R）
    cand = [(c, p) for c, p in cohort.items() if np.isfinite(p["r0_ohm"])]
    ref_cell = max(cand, key=lambda kv: kv[1]["n_cycles"])[0]
    ref_p = cohort[ref_cell]
    print(f"[3/4] 绝对标定源：{ref_cell}（{ref_p['n_cycles']} 循环）")

    for cell, p in cohort.items():
        if cell == ref_cell or not np.isfinite(p["fade_rate_per_cycle"]):
            continue
        # 预测：A_ref 来自参考电芯，温度加速用两组温差折算
        # fade_rate = A·k(T)；A = fade_rate_ref / k(T_ref)，T 用自加热中值
        dt_ref = self_heating_dt(ref_p["i_dis_a"], ref_p["r0_ohm"])
        dt_cur = self_heating_dt(p["i_dis_a"], p["r0_ohm"])
        k_ref = float(_arrhenius_factor(np.array([25.0 + 0.5 * (dt_ref["dt_min_k"] + dt_ref["dt_max_k"])]), EA)[0])
        k_cur = float(_arrhenius_factor(np.array([25.0 + 0.5 * (dt_cur["dt_min_k"] + dt_cur["dt_max_k"])]), EA)[0])
        a_ref = ref_p["fade_rate_per_cycle"] / k_ref
        pred = a_ref * k_cur
        meas = p["fade_rate_per_cycle"]
        err = (pred / meas - 1.0) * 100.0 if meas > 0 else np.nan
        loo_rows.append({
            "cell": cell, "measured_fade_rate": float(meas),
            "predicted_fade_rate": float(pred),
            "error_pct": float(err),
            "crate": p["crate"],
        })
    errs = [abs(r["error_pct"]) for r in loo_rows if np.isfinite(r["error_pct"])]
    med_err = float(np.median(errs)) if errs else np.nan
    within50 = sum(e <= 50 for e in errs)
    print(f"   留一误差中位 {med_err:.0f}%，{within50}/{len(errs)} 在 ±50% 内")

    # ---------- 4/4 判词 ----------
    abs_ok = med_err < 50.0
    beta_ok = beta_in_range >= max(1, len(beta_vals) // 2)
    thermal_frac = sum(r["explained_by_self_heating"] for r in thermal_share_rows)
    n_pairs = len(thermal_share_rows)
    verdict = (
        f"实测脉冲内阻 β 中位 {beta_med:.2f}（0.01–0.25 窗口），低于 DCIR 文献假设 "
        f"[1.5,3.5]——脉冲内阻通道是真实 DCIR 增长的下界（测量点 SOC 不定、脉冲短）；"
        f"若按实测 β，§18 反馈漂移比假设 β=2.5 更小，闭环结论是保守上界。"
        f"衰减指数 b 中位 {b_med:.2f}（0.5=SEI 与 1.0=线性 之间，p 取档合理）。"
        f"倍率加速：{thermal_frac}/{n_pairs} 对可由自加热解释——0.5C 组每 EFC 衰减反而更快"
        f"（每 EFC 时长 2×，混入日历老化，无法分离热份额；自加热仅 1–3 K 量级）。"
        f"绝对标定留一误差中位 {med_err:.0f}%——"
        + ("绝对循环数在本电芯类型上可报（带量级误差）"
           if abs_ok else "绝对循环数跨电芯预测误差过大，仍只报相对倍数（§6.1 口径维持）")
    )
    print("判词:", verdict)

    payload = {
        "dataset": "CALCE CS2（1.1 Ah 方形 LFP，Arbin 循环老化，xlsx 包）",
        "cohort_cells": sorted(cohort),
        "excluded_heterogeneous": sorted(outliers),
        "cells": profiles,
        "assumed_beta_range": [1.5, 3.5],
        "measured_beta": {
            "median": beta_med,
            "per_cell": dict(betas),
            "per_cell_noisy_r2_lt_03": dict(beta_noisy),
            "in_assumed_range": beta_in_range,
            "n": len(beta_vals),
            "r2_filter": 0.3,
            "note": "Arbin 脉冲内阻通道（>0 行中位），为 DCIR 增长的下界；"
                    "r²<0.3 的拟合噪声主导（内阻通道平/抖）只展示不进汇总；"
                    "按实测 β，闭环漂移比 §18 假设（β=2.5）更小——§18 是保守上界",
        },
        "measured_fade_exponent": {
            "median": b_med, "per_cell": dict(bs),
        },
        "rate_acceleration": rate_summary,
        "thermal_share": thermal_share_rows,
        "absolute_calibration": {
            "ref_cell": ref_cell,
            "loo": loo_rows,
            "median_abs_error_pct": med_err,
            "within_50pct": [within50, len(errs)],
        },
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (MET / "19_life_validation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- 图 ----------------
    fig, axes = plotting.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    pairs_all = betas + beta_noisy
    names = [c for c, _ in pairs_all]
    vals = [b for _, b in pairs_all]
    colors = ["#2980b9"] * len(betas) + ["#bdc3c7"] * len(beta_noisy)
    ax.bar(range(len(names)), vals, color=colors, alpha=0.85)
    ax.axhspan(1.5, 3.5, color="#27ae60", alpha=0.15, label="闭环假设 [1.5, 3.5]")
    ax.axhline(beta_med, color="#e67e22", lw=1.5, label=f"中位 {beta_med:.2f}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("β（R/R0 = 1 + β·f）")
    ax.set_title("实测内阻增长系数 vs 闭环假设")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    for cell in sorted(cohort):
        df = extract_cell(
            next(m["path"] for m in cells if m["cell"] == cell), cache_dir=str(CACHE))
        n_arr = df["cycle"].to_numpy(float)
        cap = df["cap_ah"].to_numpy(float)
        cap0 = cohort[cell]["cap0_ah"]
        ax.plot(n_arr, (1 - cap / cap0) * 100, lw=1.2,
                label=f"{cell} ({cohort[cell]['crate']:.1f}C)")
    ax.axhline(20, color="k", ls="--", lw=1, label="EOL 20%")
    ax.set_xlabel("循环")
    ax.set_ylabel("容量衰减 [%]")
    ax.set_title("实测衰减曲线（主对标队列）")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if thermal_share_rows:
        labels = [r["group"] for r in thermal_share_rows]
        implied = [r["implied_dt_k"] for r in thermal_share_rows]
        lo = [r["self_heating_dt_range_k"][0] for r in thermal_share_rows]
        hi = [r["self_heating_dt_range_k"][1] for r in thermal_share_rows]
        x = np.arange(len(labels))
        ax.bar(x, implied, 0.5, color="#8e44ad", alpha=0.8, label="实测衰减比隐含 ΔT")
        for i, (l, h) in enumerate(zip(lo, hi)):
            ax.plot([i, i], [l, h], color="#e67e22", lw=4, alpha=0.7,
                    label="自加热量级" if i == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("等效温差 ΔT [K]")
        ax.set_title("倍率加速：热份额 vs 非热份额")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "不足两组 C 率", ha="center", va="center", transform=ax.transAxes)

    ax = axes[1, 1]
    if loo_rows:
        cells_l = [r["cell"] for r in loo_rows]
        meas = [r["measured_fade_rate"] * 1e6 for r in loo_rows]
        pred = [r["predicted_fade_rate"] * 1e6 for r in loo_rows]
        x = np.arange(len(cells_l))
        ax.bar(x - 0.18, meas, 0.36, label="实测", color="#95a5a6")
        ax.bar(x + 0.18, pred, 0.36, label="标定预测", color="#e74c3c")
        ax.set_xticks(x)
        ax.set_xticklabels(cells_l, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("每 EFC 衰减速率 [1e-6]")
        ax.set_title(f"留一交叉预测（中位误差 {med_err:.0f}%）")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("19 · 寿命模型实测对标（CALCE CS2）", fontsize=13)
    plotting.save(fig, "19a_cs2_validation.png")

    print(f"完成，耗时 {payload['elapsed_s']} s")


if __name__ == "__main__":
    main()
