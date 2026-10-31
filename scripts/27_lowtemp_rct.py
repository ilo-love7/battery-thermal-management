"""27 · Panasonic 18650PF 低温 R(T) 实测收口：文献带换实测带。

背景（scripts/25 开放项）
-------------------------
scripts/25 的冷启动口径带（名义 R 低估芯温峰值 3.9–41.8 K）由两条非
实测边界撑开：A123 DCIR 下界（1.24×，动态工况负结果的残值）与文献
典型带（2.2×，标注非实测）。判词要求"温箱脉冲实测收口"。

本脚本用 Kollmeyer Panasonic 18650PF HPPC（25/10/0/−10/−20 °C ×
0.5/1/2/4/6C × 13 SOC 站，262 个脉冲）做受控 R(T) 标定——温箱协议
下 T 是唯一自由变量，单调性成立（A123 动态工况不成立的根因被移除）。

三问三答：
1. **实测 g(−10) 落在哪？** 3.1（10 s 末端）～3.3（1 s 起始）——比
   文献典型 2.2 **更高**，比 A123 下界 1.24 高出 2.5×。旧带是双向
   偏低的保守带，不是"中间值"。
2. **Arrhenius 成立吗？** 成立：r²=0.992/0.994，Ea≈22 kJ/mol，全带
   单调（受控协议 vs A123 DCIR 的 r²=−6.4）。
3. **冷启动口径带怎么变？** 名义 R 低估峰值 3.9 K（A123 下界）变成
   实测 ~7–8 K 量级（见 metrics），上界从文献 41.8 K 回落到实测口径
   ——区间从"两个非实测边界"变成"实测点 + 欧姆份额敏感性"。

产出：
  results/metrics/27_lowtemp_rct.json
  results/figures/27a_lowtemp_rct.png
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
from bmthermal.panasonic_rct import (  # noqa: E402
    extract_dataset, fit_r_int_t, rate_nonlinearity,
)

OUT = ROOT / "results"
MET = OUT / "metrics"
CACHE = OUT / "cache"
for p in (MET, CACHE):
    p.mkdir(parents=True, exist_ok=True)

PANA_ROOT = Path("D:/项目/02_数据集/01_国际公开集/Panasonic_18650PF")
T_COLD = -10.0
T_REF = 25.0
C_RATE = 2.0
H_NATURAL = 10.0
G_A123 = 1.237       # scripts/25 实测下界
G_LIT = 2.2          # scripts/25 文献典型（非实测）
CAP_FRAC_COLD = 0.580  # scripts/25 A123 容量锚点（−10 °C）


def load_25():
    spec = importlib.util.spec_from_file_location("s25", ROOT / "scripts" / "25_lowtemp_validity.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    t0 = time.time()

    # ---- 1) HPPC 实测 R(T) ----
    print("[1/3] Panasonic HPPC 解析 ...")
    pulses = extract_dataset(PANA_ROOT, CACHE / "panasonic_hppc.csv")
    print(f"   {len(pulses)} 个脉冲，温度 {sorted(pulses.temp_c.unique())}")

    fits = {}
    for metric in ("r_end", "r_inst"):
        f = fit_r_int_t(pulses, soc_ref=0.5, c_ref=2.0, metric=metric)
        fits[metric] = f
        print(f"   [{metric}] Ea={f.ea_j_mol/1000:.1f} kJ/mol  r²={f.r2:.4f}  "
              f"单调={f.monotonic}  R25={f.r_ref*1000:.1f} mΩ")
        for tc in sorted(f.r_ratio_by_temp):
            print(f"      T={tc:6.1f}°C  g_emp={f.r_ratio_by_temp[tc]:.3f}  "
                  f"g_fit={f.ratio(tc):.3f}  R={f.r_abs_by_temp[tc]*1000:.1f} mΩ")

    g_meas = {m: fits[m].r_ratio_by_temp[T_COLD] for m in fits}
    print(f"   实测 g(−10) = {g_meas['r_end']:.2f}（r_end）/ {g_meas['r_inst']:.2f}（r_inst）"
          f"  vs 文献 {G_LIT} / A123 下界 {G_A123}")

    # SOC 稳健性：0.3 / 0.7 窗
    soc_sens = {}
    for soc in (0.3, 0.7):
        try:
            f = fit_r_int_t(pulses, soc_ref=soc, c_ref=2.0, metric="r_end")
            soc_sens[soc] = f.r_ratio_by_temp[T_COLD]
        except ValueError:
            soc_sens[soc] = np.nan
    print(f"   SOC 稳健性 g(−10): 0.3→{soc_sens[0.3]:.2f}  0.5→{g_meas['r_end']:.2f}  "
          f"0.7→{soc_sens[0.7]:.2f}")

    # ---- 2) 冷启动口径带更新 ----
    print("[2/3] 冷启动带更新（复用 scripts/25 两节点模型）...")
    s25 = load_25()
    from bmthermal.heat import q_profile
    from bmthermal.network import build_cell, simulate_two_node

    s_ohm = s25.ohmic_share(C_RATE)
    t_arr_full, q_arr_full, _ = q_profile(C_RATE, t_amb_c=25.0, cache={})
    t_end_cold = float(t_arr_full[-1]) * CAP_FRAC_COLD
    m = t_arr_full <= t_end_cold
    t_arr, q_arr = t_arr_full[m], q_arr_full[m]
    cell = build_cell(h=H_NATURAL)

    def run_case(g: float) -> dict:
        """恒定 g 口径（对照）——大自加热下高估，主口径用 tdep。"""
        out = {}
        for bnd, mult in (("ohmic_lo", 1.0 + (g - 1.0) * s_ohm), ("full_hi", g)):
            sim = simulate_two_node(t_arr, q_arr * mult, T_COLD, cell)
            core, surf = sim["t_core_c"], sim["t_surface_c"]
            out[bnd] = {
                "q_mult": float(mult),
                "t_core_peak_c": float(core.max()),
                "self_heating_rise_k": float(core.max() - T_COLD),
                "gap_peak_k": float((core - surf).max()),
            }
        return out

    def run_case_tdep(fit_obj, mode: str, n_iter: int = 6) -> dict:
        """温度相关 g(T) 不动点：自加热升温 → R 回落 → 产热自限。

        g(T) 用 Arrhenius 拟合（数据带 −20…25 °C，外推到峰值温度是
        模型假设，metrics 注明）。mode: ohmic_lo（只缩放欧姆份额）/
        full_hi（全量缩放）。
        """
        mult_t = np.ones_like(q_arr)
        sim = None
        for _ in range(n_iter):
            sim = simulate_two_node(t_arr, q_arr * mult_t, T_COLD, cell)
            core = sim["t_core_c"]
            g_t = np.array([fit_obj.ratio(float(tc)) for tc in core])
            if mode == "ohmic_lo":
                mult_t = 1.0 + (g_t - 1.0) * s_ohm
            else:
                mult_t = g_t
        core, surf = sim["t_core_c"], sim["t_surface_c"]
        return {
            "q_mult_start": float(mult_t[0]),
            "q_mult_end": float(mult_t[-1]),
            "t_core_peak_c": float(core.max()),
            "self_heating_rise_k": float(core.max() - T_COLD),
            "gap_peak_k": float((core - surf).max()),
        }

    out_nom = simulate_two_node(t_arr, q_arr, T_COLD, cell)
    peak_nom = float(out_nom["t_core_c"].max())
    nom = {"q_mult": 1.0, "t_core_peak_c": peak_nom,
           "self_heating_rise_k": peak_nom - T_COLD,
           "gap_peak_k": float((out_nom["t_core_c"] - out_nom["t_surface_c"]).max())}
    cases = {
        "nominal_r": {"g_r": 1.0, "const": {"ohmic_lo": nom, "full_hi": dict(nom)},
                      "tdep": {"ohmic_lo": dict(nom), "full_hi": dict(nom)}},
        "a123_lower": {"g_r": G_A123, "const": run_case(G_A123), "tdep": None},
        "literature_typ": {"g_r": G_LIT, "const": run_case(G_LIT), "tdep": None},
        "panasonic_r_end": {"g_r": g_meas["r_end"], "const": run_case(g_meas["r_end"]),
                            "tdep": {"ohmic_lo": run_case_tdep(fits["r_end"], "ohmic_lo"),
                                     "full_hi": run_case_tdep(fits["r_end"], "full_hi")}},
        "panasonic_r_inst": {"g_r": g_meas["r_inst"], "const": run_case(g_meas["r_inst"]),
                             "tdep": {"ohmic_lo": run_case_tdep(fits["r_inst"], "ohmic_lo"),
                                      "full_hi": run_case_tdep(fits["r_inst"], "full_hi")}},
    }
    for name, c in cases.items():
        cl, ch = c["const"]["ohmic_lo"]["t_core_peak_c"], c["const"]["full_hi"]["t_core_peak_c"]
        line = f"   {name:18s} g={c['g_r']:.2f}  恒定g {cl:.1f}–{ch:.1f}°C"
        if c["tdep"]:
            tl, th = c["tdep"]["ohmic_lo"]["t_core_peak_c"], c["tdep"]["full_hi"]["t_core_peak_c"]
            line += f"  | g(T) {tl:.1f}–{th:.1f}°C（自限）"
        print(line)

    # 主口径 = g(T) 不动点（自限）；恒定 g 仅作对照上界
    d_meas_lo = cases["panasonic_r_end"]["tdep"]["ohmic_lo"]["t_core_peak_c"] - peak_nom
    d_meas_hi = cases["panasonic_r_inst"]["tdep"]["full_hi"]["t_core_peak_c"] - peak_nom
    d_const_hi = cases["panasonic_r_inst"]["const"]["full_hi"]["t_core_peak_c"] - peak_nom
    d_old_lo = cases["a123_lower"]["const"]["ohmic_lo"]["t_core_peak_c"] - peak_nom
    d_old_hi = cases["literature_typ"]["const"]["full_hi"]["t_core_peak_c"] - peak_nom
    print(f"   口径带：旧 {d_old_lo:.1f}–{d_old_hi:.1f} K（恒定 g）→ "
          f"实测 {d_meas_lo:.1f}–{d_meas_hi:.1f} K（g(T) 自限）；"
          f"恒定 g 对照上界 {d_const_hi:.1f} K")

    # ---- 3) 高倍率 R(C) 非线性（WP28 锚点）----
    print("[3/3] R(C) 倍率非线性 ...")
    nonlin = {}
    for tc in (25.0, -10.0):
        nl = rate_nonlinearity(pulses, tc)
        nonlin[tc] = nl.to_dict(orient="records")
        valid = nl[(nl.c_rate >= 0.5) & (nl.r_end > 0)]
        if len(valid) >= 2:
            p_fit = np.polyfit(np.log(valid.c_rate), np.log(valid.r_end), 1)[0]
            # Q=I²R → dlnQ/dlnC = 2 + dlnR/dlnC
            q_exp = 2.0 + p_fit
            nonlin[f"{tc}_r_slope"] = float(p_fit)
            nonlin[f"{tc}_q_exp"] = float(q_exp)
            print(f"   T={tc:5.0f}°C  dlnR/dlnC={p_fit:+.3f} → 欧姆产热指数 p={q_exp:.2f}")
        # 6C@−10 数据质量剔除标注
    anom = pulses[(pulses.temp_c == -10.0) & (pulses.c_rate > 5.5) & (pulses.r_end < 0.01)]
    nonlin["anomalies"] = int(len(anom))

    # ---- 指标落盘 ----
    metrics = {
        "panasonic_fit": {
            m: {
                "ea_kj_mol": fits[m].ea_j_mol / 1000.0,
                "r2": fits[m].r2,
                "monotonic": fits[m].monotonic,
                "r_ref_ohm": fits[m].r_ref,
                "r_ratio_by_temp": fits[m].r_ratio_by_temp,
                "r_abs_by_temp": fits[m].r_abs_by_temp,
                "g_cold": g_meas[m],
            } for m in fits
        },
        "soc_sensitivity_g_cold": soc_sens,
        "vs_previous_bands": {
            "a123_lower": G_A123, "literature_typ": G_LIT,
            "panasonic_measured": [g_meas["r_end"], g_meas["r_inst"]],
        },
        "cold_start_case": {
            "t_amb_c": T_COLD, "c_rate": C_RATE, "h_w_m2k": H_NATURAL,
            "cap_frac": CAP_FRAC_COLD,
            "t_discharge_min": float(t_arr[-1] - t_arr[0]) / 60.0,
            "share_ohmic": s_ohm,
            "peak_nominal_c": peak_nom,
            "cases": cases,
            "delta_peak_old_band_k": [d_old_lo, d_old_hi],
            "delta_peak_measured_band_k": [d_meas_lo, d_meas_hi],
            "delta_peak_const_g_upper_k": d_const_hi,
            "note": "主口径 g(T) 不动点（自加热→R 回落→产热自限）；恒定 g 为对照上界。"
                    "g(T) Arrhenius 外推到峰值温度是模型假设（数据带 −20…25 °C）",
        },
        "rate_nonlinearity": nonlin,
        "verdict": (
            "温箱脉冲实测收口 scripts/25 低温带：g(−10)=3.1（r_end）–3.3（r_inst），"
            "比文献典型 2.2 **更高**、比 A123 下界 1.24 高 2.5×——旧带是双向偏低的保守带。"
            f"受控协议下 Arrhenius 成立（r²={fits['r_end'].r2:.3f}，Ea≈{fits['r_end'].ea_j_mol/1000:.0f} kJ/mol，全带单调）——"
            "A123 DCIR 非单调的根因是协议/SOC 调制，不是 R(T) 本身不可测。"
            f"冷启动（−10 °C/2C/自然对流，容量截断 {t_arr[-1]/60:.0f} min）名义 R 低估芯温峰值 "
            f"{d_old_lo:.1f}–{d_old_hi:.1f} K（旧恒定 g 双边界）→ {d_meas_lo:.1f}–{d_meas_hi:.1f} K"
            f"（实测 g(T) 自限口径；恒定 g 对照上界 {d_const_hi:.1f} K 会因忽略 R 温升回落而高估）。"
            "高倍率锚点（WP28）：25 °C R 对 0.5–6C 平坦（欧姆产热指数 p≈2.00），"
            "−10 °C R 随倍率降（p≈1.83，自加热/极化）；6C@−10 脉冲触底数据质量剔除。"
        ),
        "elapsed_s": time.time() - t0,
    }
    (MET / "27_lowtemp_rct.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 图 27a ----
    plotting.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) R vs T：脉冲散点 + Arrhenius 拟合
    ax = axes[0, 0]
    for metric, mk, col in (("r_end", "o", "#2980b9"), ("r_inst", "s", "#c0392b")):
        f = fits[metric]
        sub = pulses[
            ((pulses["soc"] - 0.5).abs() <= 0.08)
            & ((pulses["c_rate"] - 2.0).abs() <= 0.3)
            & (pulses[metric] > 0)
        ]
        ax.scatter(sub["temp_c"], sub[metric] * 1000, marker=mk, s=18, alpha=0.35,
                   c=col, label=f"{metric} 脉冲")
        ts = np.linspace(-22, 28, 60)
        ax.plot(ts, f.r_ref * 1000 * np.exp(f.ea_j_mol / 8.314462618
                * (1.0 / (ts + 273.15) - 1.0 / (T_REF + 273.15))),
                c=col, lw=2, label=f"{metric} Arrhenius (Ea={f.ea_j_mol/1000:.0f} kJ/mol)")
    ax.set_xlabel("温度 (°C)")
    ax.set_ylabel("R (mΩ)")
    ax.set_title("(a) HPPC 实测 R(T)：单调且拟合良好")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) g(T) 对比
    ax = axes[0, 1]
    ts = np.linspace(-20, 25, 60)
    f = fits["r_end"]
    ax.plot(ts, [f.ratio(x) for x in ts], c="#2980b9", lw=2, label="Panasonic 实测拟合")
    ax.scatter(sorted(f.r_ratio_by_temp),
               [f.r_ratio_by_temp[k] for k in sorted(f.r_ratio_by_temp)],
               c="#2980b9", s=45, zorder=3, edgecolors="k", linewidths=0.5)
    ax.axhline(G_LIT, color="#7f8c8d", ls="--", lw=1.2, label=f"文献典型 {G_LIT}（旧）")
    ax.axhline(G_A123, color="#27ae60", ls=":", lw=1.2, label=f"A123 下界 {G_A123}（旧）")
    ax.set_xlabel("温度 (°C)")
    ax.set_ylabel("g = R(T)/R(25)")
    ax.set_title("(b) g(T)：实测高于旧文献带")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) 冷启动峰值带
    ax = axes[1, 0]
    names = ["nominal_r", "a123_lower", "literature_typ", "panasonic_r_end", "panasonic_r_inst"]
    labels = ["名义 R", "A123 下界\n(旧,恒g)", "文献典型\n(旧,恒g)", "实测 r_end\n(g(T))", "实测 r_inst\n(g(T))"]

    def pick(n, bnd):
        c = cases[n]
        src = c["tdep"] if c["tdep"] else c["const"]
        return src[bnd]["t_core_peak_c"]
    lo = [pick(n, "ohmic_lo") for n in names]
    hi = [pick(n, "full_hi") for n in names]
    x = np.arange(len(names))
    ax.bar(x, np.array(hi) - np.array(lo), bottom=lo, color="#5dade2", alpha=0.75,
           edgecolor="k", linewidth=0.5, label="欧姆份额边界")
    ax.scatter(x, lo, marker="o", c="#2c3e50", s=32, zorder=3)
    ax.scatter(x, hi, marker="o", c="#2c3e50", s=32, zorder=3)
    ax.axhline(peak_nom, color="#7f8c8d", ls="--", lw=1, label=f"名义峰值 {peak_nom:.1f}°C")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("芯温峰值 (°C)")
    ax.set_title("(c) −10 °C 冷启动：实测带替换旧带")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    # (d) R(C) 非线性（WP28 锚点）
    ax = axes[1, 1]
    for tc, col in ((25.0, "#e67e22"), (-10.0, "#3498db")):
        nl = rate_nonlinearity(pulses, tc)
        nl = nl[nl.r_end > 0]
        ax.plot(nl.c_rate, nl.r_end * 1000, "o-", c=col, ms=5,
                label=f"T={tc:.0f}°C")
    ax.set_xlabel("脉冲倍率 (C)")
    ax.set_ylabel("R_end (mΩ)")
    ax.set_title("(d) R(C) 非线性：25 °C 平坦 / −10 °C 下行")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    plotting.save(fig, "27a_lowtemp_rct.png")

    print(f"完成（{metrics['elapsed_s']:.0f} s）：{MET/'27_lowtemp_rct.json'}")


if __name__ == "__main__":
    main()
