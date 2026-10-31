"""25 · 低温有效性边界 + R_int(T) 标定（A123 多温度数据）。

背景
----
全项目此前 R_int 为常数（25 °C 口径）。A123 数据（DST/US06/FUDS 在
−10~50 °C 八温度点 × 两芯）尝试标定 R(T) 形状。

**负结果（主发现之一）**：Arbin DCIR 列在动态工况下被协议/SOC/电流幅度
调制，跨温度非单调（30–50 °C 反而上翘）——公开动态工况数据**标不了**
R(T)，只能给出下界（−10 °C 实测仅 ~1.2×，文献典型 LFP/石墨 DCIR 为
2–2.5×）。容量(T) 则是可靠的实测锚点（−10 °C 掉 ~42%）。

三问三答：
1. R(T) 形状多大？——双口径：A123 实测下界 + 文献典型值带（标注非实测）；
2. **−10 °C 冷启动**：名义 R 口径低估多少温升/gap？（按 g 带给区间；
   自加热既是'免费预热'也是热风险——两侧都定量）
3. 低温下 scripts/20 的脉冲-弛豫辨识还准吗？（辨识法不依赖 R 先验——
   τ 指纹直接反解，低温照样可用；缺先验硬用名义 R 才出错）

口径声明：
- R(T) 带：g(−10) ∈ {1.2（A123 实测下界）, 2.2（文献典型，非实测）}，
  典型带用等效 Arrhenius 形状（g(−10)=2.2 反推 Ea≈15 kJ/mol）；
- 低温产热修正双边界：只放大欧姆份额（下界）vs 全产热等比（上界，
  低温极化也随 R 增长）；
- 容量效应（实测）：−10 °C 可用容量 ~58% → 同 Ah 需求有效倍率 ×1.7。

产出：
  results/metrics/25_lowtemp_validity.json
  results/figures/25a_lowtemp_validity.png
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal import plotting  # noqa: E402
from bmthermal.a123_thermal import extract_a123_dataset, fit_r_int_t  # noqa: E402
from bmthermal.heat import q_profile  # noqa: E402
from bmthermal.network import build_cell, simulate_two_node  # noqa: E402
from bmthermal.observer import fit_relaxation_tau, identify_r_int_k_w  # noqa: E402
from bmthermal.params import R_GAS  # noqa: E402

OUT = ROOT / "results"
MET = OUT / "metrics"
CACHE = OUT / "cache"
MET.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

T_COLD = -10.0
C_RATE = 2.0
H_NATURAL = 10.0
T_REF = 25.0
G_LIT_COLD = 2.2          # 文献典型 LFP/石墨 DCIR：−10 °C 约 2–2.5×（非实测）


def ohmic_share(c_rate: float) -> float:
    try:
        p = json.loads((MET / "01_heat_sweep.json").read_text(encoding="utf-8"))
        rows = [r for r in p["sweep"] if abs(r["c_rate"] - c_rate) < 1e-6]
        if rows:
            return float(rows[0]["share_ohmic"])
    except Exception:
        pass
    return 0.47


def relax_trace(cell, q_hot: float, t_amb_c: float, t_hot_s: float = 1800.0,
                t_relax_s: float = 1200.0, dt: float = 5.0):
    t = np.arange(0.0, t_hot_s + t_relax_s + dt, dt)
    q = np.where(t < t_hot_s, q_hot, 0.0)
    out = simulate_two_node(t, q, t_amb_c, cell)
    return t, out["t_core_c"], out["t_surface_c"], t_hot_s


def main() -> None:
    t0 = time.time()

    # ---- 1) A123：容量(T) 实测锚点 + DCIR(T) 负结果 ----
    print("[1/3] A123 多温度数据解析 ...")
    df = extract_a123_dataset(cache_csv=str(CACHE / "a123_r_t.csv"))
    fit = fit_r_int_t(df, t_ref_c=T_REF)
    ts = sorted(fit.r_ratio_by_temp)
    print(f"   {len(df)} 行，温度: {ts}")
    for tc in ts:
        print(f"   T={tc:6.1f}°C  R/R25={fit.r_ratio_by_temp[tc]:.3f}  "
              f"cap={fit.cap_by_temp[tc]:.2f} Ah")
    g_meas_cold = fit.r_ratio_by_temp.get(T_COLD, np.nan)
    print(f"   负结果：Arrhenius 全带拟合 Ea={fit.ea_r_j_mol/1000:.1f} kJ/mol  "
          f"r²={fit.r2:.3f}（DCIR 列非单调——标不了 R(T)，只作下界）")

    cap_ref = fit.cap_by_temp.get(T_REF)
    cap_cold = fit.cap_by_temp.get(T_COLD)
    cap_frac = cap_cold / cap_ref if cap_ref else np.nan
    print(f"   容量实测：−10 °C {cap_cold:.2f} Ah / 25 °C {cap_ref:.2f} Ah "
          f"= {cap_frac:.0%}（同 Ah 需求有效倍率 ×{1/cap_frac:.1f}）")

    # 文献典型带：g(T)=exp(Ea_lit/R(1/T−1/T_ref))，Ea 由 g(−10)=2.2 反推
    t_k_cold = T_COLD + 273.15
    t_k_ref = T_REF + 273.15
    ea_lit = np.log(G_LIT_COLD) / (1.0 / t_k_cold - 1.0 / t_k_ref) * R_GAS

    def g_lit(t_c: float) -> float:
        return float(np.exp(ea_lit / R_GAS * (1.0 / (t_c + 273.15) - 1.0 / t_k_ref)))

    print(f"   文献典型带：g(−10)={G_LIT_COLD}（标注非实测），等效 Ea≈{ea_lit/1000:.0f} kJ/mol")

    # ---- 2) −10 °C 冷启动：g 双口径 × 产热修正双边界 ----
    print("[2/3] −10 °C 冷启动场景 ...")
    s_ohm = ohmic_share(C_RATE)
    t_arr_full, q_arr_full, _ = q_profile(C_RATE, t_amb_c=25.0, cache={})
    # 低温可用容量截断：−10 °C 只能放出 cap_frac 份额 → 放电时长同比缩短
    # （不截断会得到"2C 连续放到 100 °C"的非物理外推——真实 BMS 也会限流）
    t_end_cold = float(t_arr_full[-1]) * cap_frac
    m = t_arr_full <= t_end_cold
    t_arr, q_arr = t_arr_full[m], q_arr_full[m]
    print(f"   放电窗：25 °C 口径 {t_arr_full[-1]/60:.0f} min → "
          f"−10 °C 容量截断 {t_end_cold/60:.0f} min")
    cell = build_cell(h=H_NATURAL)
    q_nom = float(q_arr.mean())

    cases = {}
    for g_name, g in (("a123_lower", float(g_meas_cold)), ("literature_typ", G_LIT_COLD)):
        mult_lo = 1.0 + (g - 1.0) * s_ohm
        mult_hi = g
        for bnd, mult in (("ohmic_lo", mult_lo), ("full_hi", mult_hi)):
            out = simulate_two_node(t_arr, q_arr * mult, T_COLD, cell)
            core, surf = out["t_core_c"], out["t_surface_c"]
            cases[f"{g_name}/{bnd}"] = {
                "g_r": g, "q_mult": mult,
                "t_core_peak_c": float(core.max()),
                "self_heating_rise_k": float(core.max() - T_COLD),
                "gap_peak_k": float((core - surf).max()),
            }
            print(f"   {g_name:14s} {bnd:9s} Q×{mult:.2f}  峰 {core.max():.1f}°C  "
                  f"自加热 +{core.max()-T_COLD:.1f} K  gap {(core-surf).max():.2f} K")
    out_nom = simulate_two_node(t_arr, q_arr, T_COLD, cell)
    cases["nominal_r"] = {
        "g_r": 1.0, "q_mult": 1.0,
        "t_core_peak_c": float(out_nom["t_core_c"].max()),
        "self_heating_rise_k": float(out_nom["t_core_c"].max() - T_COLD),
        "gap_peak_k": float((out_nom["t_core_c"] - out_nom["t_surface_c"]).max()),
    }
    peak_nom = cases["nominal_r"]["t_core_peak_c"]
    d_peak_meas = cases["a123_lower/ohmic_lo"]["t_core_peak_c"] - peak_nom
    d_peak_lit = cases["literature_typ/full_hi"]["t_core_peak_c"] - peak_nom
    gap_err_lit = (cases["literature_typ/ohmic_lo"]["gap_peak_k"]
                   - cases["nominal_r"]["gap_peak_k"])

    # ---- 3) 低温脉冲-弛豫辨识（不依赖 R 先验） ----
    print("[3/3] 低温弛豫辨识 ...")
    r_nom = cell.r_int_k_w
    g_case = G_LIT_COLD           # 用典型带做辨识压力案例
    r_true = r_nom * g_case
    cell_cold = _replace(cell, r_int_k_w=r_true)
    q_hot = q_nom
    t_rel, core_rel, surf_rel, t_rs = relax_trace(cell_cold, q_hot, T_COLD)
    fit_tau = fit_relaxation_tau(t_rel, surf_rel, t_rs + 30.0, t_rel[-1])
    r_ident = identify_r_int_k_w(fit_tau["tau_s"], cell)
    ident_err_pct = (r_ident / r_true - 1.0) * 100 if np.isfinite(r_ident) else np.nan
    gap_true = q_hot * r_true
    gap_nom = q_hot * r_nom
    print(f"   τ_fit={fit_tau['tau_s']:.0f} s (r²={fit_tau['r2']:.3f})  "
          f"R_ident={r_ident:.4f} vs 真值 {r_true:.4f} K/W  误差 {ident_err_pct:+.1f}%")
    print(f"   硬用名义 R: gap 低估 {gap_true-gap_nom:.2f} K（{q_hot:.1f} W 工况）")

    # ---- 判词 ----
    verdict = (
        f"A123 负结果：动态工况 DCIR 列非单调（全带拟合 r²={fit.r2:.1f}）——"
        f"公开数据标不了 R(T)，只能给下界（−10 °C 实测 {g_meas_cold:.2f}×）；"
        f"文献典型带 g(−10)≈{G_LIT_COLD}×（非实测标注）与下界构成口径区间。"
        f"容量实测可靠：−10 °C 掉 {(1-cap_frac)*100:.0f}%（同 Ah 需求有效倍率 ×{1/cap_frac:.1f}，"
        f"放电窗按可用容量截断 {t_end_cold/60:.0f} min）。"
        f"冷启动（−10 °C/2C/自然对流）名义 R 口径低估芯温峰值 "
        f"{d_peak_meas:.1f} K（实测下界）～{d_peak_lit:.1f} K（典型带上界），"
        f"低估 gap 最多 {gap_err_lit:.1f} K——自加热是免费预热也是热风险，"
        f"两侧都在该量级；脉冲-弛豫辨识不依赖 R 先验（合成数据误差 "
        f"{ident_err_pct:+.0f}%），低温照样可用，硬用名义 R 才有 "
        f"{gap_true-gap_nom:.1f} K 的 gap 口径错。"
        f"有效性边界补温度轴：≤0 °C 工况必须用 R(T) 带修正产热/gap/辨识排程。"
    )
    print("判词:", verdict)

    payload = {
        "a123_measurement": {
            "r_ratio_by_temp": fit.r_ratio_by_temp,
            "r_ohm_by_temp": fit.r_ohm_by_temp,
            "cap_by_temp": fit.cap_by_temp,
            "arrhenius_full_band": {"ea_r_j_mol": fit.ea_r_j_mol, "r2": fit.r2,
                                    "negative_result": "DCIR 列非单调，全带拟合不可用——只作下界"},
            "g_cold_lower_bound": g_meas_cold,
            "cap_frac_at_cold": cap_frac,
        },
        "literature_band": {
            "g_cold_typical": G_LIT_COLD,
            "ea_equiv_j_mol": ea_lit,
            "note": "文献典型 LFP/石墨 DCIR 低温倍数（非实测）；等效 Ea 由 g(−10)=2.2 反推",
        },
        "cold_start_case": {
            "t_amb_c": T_COLD, "c_rate": C_RATE, "h_w_m2k": H_NATURAL,
            "share_ohmic": s_ohm,
            "cases": cases,
            "delta_peak_k_measured_lower": d_peak_meas,
            "delta_peak_k_literature_upper": d_peak_lit,
            "gap_underestimate_k_literature": gap_err_lit,
        },
        "lowtemp_identification": {
            "g_assumed": g_case,
            "tau_fit_s": fit_tau["tau_s"], "tau_r2": fit_tau["r2"],
            "r_ident_k_w": r_ident, "r_true_k_w": r_true,
            "ident_err_pct": ident_err_pct,
            "gap_true_k": gap_true, "gap_nominal_k": gap_nom,
        },
        "assumptions": [
            "R(T) 双口径：A123 DCIR 下界（非单调，负结果）+ 文献典型带（标注非实测）",
            "产热修正双边界：欧姆份额放大（下界）/ 全量等比（上界）",
            "冷启动取自然对流 h=10、2C 放电（Chen2020 q_profile 口径）",
            "辨识验证为合成弛豫数据（真值已知），非 A123 实测弛豫",
        ],
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (MET / "25_lowtemp_validity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    # ---- 图 25a ----
    plotting.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) R(T)/R25：实测（下界）+ 文献典型带
    ax = axes[0, 0]
    rr = [fit.r_ratio_by_temp[t] for t in ts]
    ax.plot(ts, rr, "o", color="#1D4ED8", ms=7, label="A123 DCIR 实测（下界口径）")
    tt = np.linspace(-10, 50, 60)
    ax.plot(tt, [g_lit(t) for t in tt], "-", color="#B45309", lw=2,
            label=f"文献典型带（非实测，g(−10)={G_LIT_COLD}）")
    ax.fill_between(tt, 1.0, [g_lit(t) for t in tt], color="#B45309", alpha=0.10)
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("温度 °C")
    ax.set_ylabel("R(T) / R(25°C)")
    ax.set_title("a) R(T) 口径区间（实测非单调→只作下界）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) 容量 vs T（实测锚点）
    ax = axes[0, 1]
    caps = [fit.cap_by_temp[t] for t in ts]
    ax.plot(ts, caps, "s", color="#047857", ms=7, label="A123 实测容量")
    ax.axhline(cap_ref, color="gray", ls=":", lw=0.8, label=f"25 °C = {cap_ref:.2f} Ah")
    ax.annotate(f"−10 °C 仅 {cap_frac:.0%}", (T_COLD, cap_cold),
                xytext=(15, 10), textcoords="offset points", fontsize=8, color="#047857")
    ax.set_xlabel("温度 °C")
    ax.set_ylabel("放电容量 Ah")
    ax.set_title("b) 容量 vs 温度（实测可靠锚点）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) 冷启动自加热轨迹（口径带）
    ax = axes[1, 0]
    lines = [("nominal_r", "#6B7280", "名义 R（25 °C 口径）", "-"),
             ("a123_lower/ohmic_lo", "#0E7490", "实测下界×欧姆修正", "-"),
             ("literature_typ/ohmic_lo", "#1D4ED8", "典型带×欧姆修正", "--"),
             ("literature_typ/full_hi", "#B45309", "典型带×全量（上界）", "-")]
    for key, color, lab, ls in lines:
        mult = cases[key]["q_mult"]
        out = simulate_two_node(t_arr, q_arr * mult, T_COLD, cell)
        ax.plot(t_arr / 60, out["t_core_c"], ls, color=color, lw=2, label=lab)
    ax.axhline(25, color="green", ls=":", lw=1, label="25 °C（最佳窗口）")
    ax.set_xlabel("时间 min")
    ax.set_ylabel("芯温 °C")
    ax.set_title(f"c) −10 °C/2C 冷启动自加热（口径带 {d_peak_meas:.1f}–{d_peak_lit:.1f} K）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) 辨识与口径误差
    ax = axes[1, 1]
    gap_ident = float(r_ident) * q_hot if np.isfinite(r_ident) else gap_true
    labels = ["名义 R\n(不辨识)", "弛豫辨识\n(低温)"]
    gap_vals = [gap_nom, gap_ident]
    ax.bar(labels, gap_vals, color=["#6B7280", "#1D4ED8"], width=0.5)
    ax.axhline(gap_true, color="#B45309", ls="--", lw=1.5,
               label=f"真值 gap（g={g_case:.1f}）{gap_true:.1f} K")
    ax.set_ylabel(f"芯-壳 gap @ {q_hot:.1f} W（K）")
    ax.set_title("d) 低温 gap 口径：先验 vs 辨识")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("25 · 低温有效性边界与 R_int(T) 标定", fontsize=13)
    fig.tight_layout()
    plotting.save(fig, "25a_lowtemp_validity.png")
    print(f"[落盘] 25_lowtemp_validity.json / 25a_lowtemp_validity.png  ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
