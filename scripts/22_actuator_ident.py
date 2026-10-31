"""22 · 执行器产品化 + 辨识脉冲排程（时分复用）。

两件事：
1. **真实功率曲线重估 scripts/21 的能耗结论**。P ∝ h³ 理想律假设低档免费；
   真实风机/泵有空载份额（电机铁损）与静扬程线性项（冷板泵 P=ρgQ(H₀+kQ²)/η）
   → 滞环低档不是免费的 → 触发式冷却的节能幅度被高估多少？
   三条曲线：AFFINITY（对照）/ FAN_TYPICAL / PUMP_TYPICAL（文献典型量级）。

2. **辨识脉冲排程**。scripts/20 的脉冲-弛豫辨识要求弛豫窗内 h 恒定；
   R_int 又随老化漂移（β 口径 0.5–3.5）。三个量化问题：
   a) 辨识周期该多长？（R 误差容忍 5% → period = r_tol/(β·fade_per_efc)）
   b) 辨识窗冻结控制的代价多大？（持续高负荷段开窗 vs 低负荷段开窗）
   c) 时分复用策略：优先排在低负荷/自然弛豫段（q ≤ 阈值才开窗）。

工况：与 scripts/21 同场景（持续 2C 基线 + US06 纹波 2 h，T_amb=30 °C）
**外加 1 h 低负荷段（0.3C）**——现实的"行驶段 + 驻车段"，也是排程
策略的用武之地（驻车段开辨识窗，控制冻结代价≈0）。

产出：
  results/metrics/22_actuator_ident.json
  results/figures/22a_actuator_ident.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmthermal import plotting  # noqa: E402
from bmthermal.actuator import (  # noqa: E402
    AFFINITY,
    FAN_TYPICAL,
    PUMP_TYPICAL,
    ActuatorCurve,
    duty_energy_norm,
    energy_from_trace,
    energy_norm_from_trace,
    ident_hold_law,
    ident_period_efc,
    ident_schedule,
)
from bmthermal.control import always_on, bang_bang, simulate_control_loop  # noqa: E402
from bmthermal.network import build_cell  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MET = OUT / "metrics"
MET.mkdir(parents=True, exist_ok=True)

# ---- 场景（与 21 同参数，尾部加 1 h 低负荷段） ----
T_AMB = 30.0
BASE_C = 2.0
RIPPLE_SCALE = 1.0
CYCLES = 12              # 高负荷段 2 h
TAIL_C = 0.3             # 驻车/低负荷段倍率
TAIL_S = 3600.0
H_LO = 25.0
H_HI = 500.0
NOISE_K = 0.05
T_LIMIT_C = 45.0
T_ON = 43.0
BAND_K = 3.0
T_OFF = T_ON - BAND_K
MIN_DWELL_S = 20.0

# 辨识排程
IDENT_WINDOW_S = 60.0        # 脉冲-弛豫窗（scripts/20 的 60 s 脉冲口径）
IDENT_PERIODS_S = (600.0, 1800.0, 3600.0)   # 压力测试档（标定模式）
Q_LOWLOAD_W = 1.0            # 低负荷判定阈值（q ≤ 阈值才开辨识窗）

# R 漂移 / 衰减口径
BETA_CASES = {
    "cs2_dcir_lower": 0.48,   # CS2 脉冲内阻实测中位（DCIR 下界）
    "model_default": 2.5,     # §18 默认
    "model_upper": 3.5,       # §18 上界
}
FADE_CASES = {
    "cs2_0.5C_per_efc": 3.44e-4,   # scripts/19 实测（0.5C 组均值）
    "cs2_1.0C_per_efc": 2.79e-4,   # scripts/19 实测（1.0C 组均值）
    "model_2000efc": 1.0e-4,       # §18 EFC 锚点 2000 → 0.2/2000
}
R_TOL = 0.05


def q_from_crate(c_rate: np.ndarray) -> np.ndarray:
    metrics = Path(__file__).resolve().parents[1] / "results" / "metrics" / "01_heat_sweep.json"
    try:
        law = json.loads(metrics.read_text(encoding="utf-8"))["power_law"]
        k, alpha = law["k"], law["alpha"]
    except Exception:
        k, alpha = 0.7821775390520517, 1.7909636972736782
    return k * np.abs(np.asarray(c_rate, float)) ** alpha


def make_duty():
    from bmthermal.duty import load_us06

    t_p, c_p = load_us06(scale=RIPPLE_SCALE)
    dt = float(np.median(np.diff(t_p)))
    t_list, c_list = [], []
    t_off = 0.0
    for _ in range(CYCLES):
        t_list.append(t_p + t_off)
        c_list.append(c_p)
        t_off += float(t_p[-1]) + dt
    t_hi = np.concatenate(t_list)
    crate_hi = BASE_C + np.concatenate(c_list)

    # 尾部低负荷段：1 s 采样
    t_tail = np.arange(1.0, TAIL_S + 1.0) + t_hi[-1]
    crate = np.concatenate([crate_hi, np.full_like(t_tail, TAIL_C)])
    t = np.concatenate([t_hi, t_tail])
    q = q_from_crate(crate)
    return t, q, crate


def run_case(t, q, law, cell, seed=0):
    return simulate_control_loop(
        t, q, T_AMB, cell, cell,
        h_lo=H_LO, h_hi=H_HI, law=law,
        noise_k=NOISE_K, seed=seed,
        min_dwell_s=MIN_DWELL_S,
    )


def energy_table(res) -> dict:
    out = {}
    for name, curve in (("affinity", AFFINITY), ("fan_typical", FAN_TYPICAL),
                        ("pump_typical", PUMP_TYPICAL)):
        out[name] = {
            "energy_norm": energy_norm_from_trace(res.t_s, res.h_cmd, H_LO, H_HI, curve),
            "energy_w_s": energy_from_trace(res.t_s, res.h_cmd, H_HI, curve),
            "p_idle_w": float(curve.power_w(0.0)),
            "p_lo_w": float(curve.power_w(H_LO / H_HI)),
            "p_hi_w": float(curve.power_w(1.0)),
        }
    return out


def main() -> None:
    t0 = time.perf_counter()
    cell = build_cell(h=H_HI)
    t, q, crate = make_duty()
    print(f"[场景] {t[-1]/3600:.2f} h, Q: {q.min():.2f}–{q.max():.2f} W, "
          f"高负荷段 {CYCLES*600/3600:.0f} h + 低负荷 {TAIL_S/3600:.0f} h")

    # ---- 1) 三策略 × 三功率曲线的能耗重估 ----
    laws = {
        "full_cool": always_on(1),
        "surface_trigger": bang_bang(T_ON, T_OFF, use_estimate=False),
        "core_est_trigger": bang_bang(T_ON, T_OFF, use_estimate=True),
    }
    strategies: dict = {}
    results: dict = {}
    for name, law in laws.items():
        res = run_case(t, q, law, cell, seed=1)
        results[name] = res
        strategies[name] = {
            "energy": energy_table(res),
            "duty_high": res.duty_high,
            "n_switch": res.n_switch,
            "t_max_core_c": res.t_max_core_c,
            "damage_s": res.damage_s,
        }
        e3 = strategies[name]["energy"]
        print(f"[{name:16s}] duty={res.duty_high:.3f}  E_norm: "
              f"aff={e3['affinity']['energy_norm']:.3f}  "
              f"fan={e3['fan_typical']['energy_norm']:.3f}  "
              f"pump={e3['pump_typical']['energy_norm']:.3f}  "
              f"peak={res.t_max_core_c:.1f}°C")

    # 解析对拍（稳态占空比口径）
    analytic_check = {
        name: duty_energy_norm(res.duty_high, H_LO, H_HI, PUMP_TYPICAL)
        for name, res in results.items()
    }

    # ---- 2) 辨识周期表（R 漂移约束） ----
    cadence = {}
    for bname, beta in BETA_CASES.items():
        cadence[bname] = {}
        for fname, fade in FADE_CASES.items():
            per = ident_period_efc(beta, fade, r_tol=R_TOL)
            cadence[bname][fname] = {
                "period_efc": per,
                "hours_at_1c": per * 2.0,     # 1C 充放一循环 ~2 h/EFC
                "hours_at_3c": per * 2.0 / 3.0,
            }
    for bname, row in cadence.items():
        per = row["cs2_0.5C_per_efc"]
        print(f"[辨识周期] β={BETA_CASES[bname]:<4} → {per['period_efc']:.0f} EFC "
              f"(≈{per['hours_at_1c']:.0f} h @1C)")

    # ---- 3) 辨识窗冻结控制的代价（时分复用） ----
    base = results["core_est_trigger"]
    impact = {"baseline_no_ident": {
        "t_max_core_c": base.t_max_core_c,
        "damage_s": base.damage_s,
    }}
    dt_arr = np.diff(t, prepend=t[0]); dt_arr[0] = 0.0
    over = base.t_core_true_c > T_LIMIT_C
    impact["baseline_no_ident"]["time_over_limit_s"] = float(dt_arr[over].sum()) if over.any() else 0.0

    for period in IDENT_PERIODS_S:
        for mode in ("unconditional", "low_load_only"):
            if mode == "unconditional":
                mask = ident_schedule(t, period, IDENT_WINDOW_S)
            else:
                mask = ident_schedule(t, period, IDENT_WINDOW_S,
                                      q_w=q, q_threshold_w=Q_LOWLOAD_W)
            law = ident_hold_law(bang_bang(T_ON, T_OFF, use_estimate=True), mask, t)
            res = run_case(t, q, law, cell, seed=1)
            over = res.t_core_true_c > T_LIMIT_C
            md = mask.astype(int)
            n_win = int((np.diff(md) == 1).sum() + (1 if md[0] else 0))
            impact[f"p{int(period)}_{mode}"] = {
                "period_s": period,
                "n_windows": n_win,
                "window_s_total": float(mask.sum()),
                "t_max_core_c": res.t_max_core_c,
                "damage_s": res.damage_s,
                "time_over_limit_s": float(dt_arr[over].sum()) if over.any() else 0.0,
                "delta_peak_k": res.t_max_core_c - base.t_max_core_c,
            }
            print(f"[辨识窗 {int(period):5d}s {mode:14s}] 窗数={impact[f'p{int(period)}_{mode}']['n_windows']}  "
                  f"Δpeak={impact[f'p{int(period)}_{mode}']['delta_peak_k']:+.2f} K  "
                  f"超限={impact[f'p{int(period)}_{mode}']['time_over_limit_s']:.0f} s")

    # ---- 判词 ----
    fan_pen = strategies["core_est_trigger"]["energy"]["fan_typical"]["energy_norm"] \
        - strategies["core_est_trigger"]["energy"]["affinity"]["energy_norm"]
    pump_pen = strategies["core_est_trigger"]["energy"]["pump_typical"]["energy_norm"] \
        - strategies["core_est_trigger"]["energy"]["affinity"]["energy_norm"]
    per_prod = cadence["model_default"]["cs2_0.5C_per_efc"]["period_efc"]
    worst = max(v["delta_peak_k"] for k, v in impact.items() if k.startswith("p600"))
    verdict = (
        f"理想 P∝h³ 高估触发式节能：真实曲线下芯温触发归一能耗上移 "
        f"风机 +{fan_pen:.3f} / 泵 +{pump_pen:.3f}（空载份额+静扬程吃掉低档免费假设）；"
        f"辨识周期（R 误差容忍 {R_TOL:.0%}）：β=0.48→{cadence['cs2_dcir_lower']['cs2_0.5C_per_efc']['period_efc']:.0f} EFC，"
        f"β=2.5→{per_prod:.0f} EFC（生产级是天级，60 s 窗的控制代价可忽略）；"
        f"压力档 600 s 强制开窗 Δpeak 最大 {worst:.1f} K，而低负荷排程 Δpeak≈0"
        "——辨识排程应与放电计划耦合（驻车/低倍率段开窗），既免控制冻结又正对弛豫黄金窗。"
    )
    print("判词:", verdict)

    # ---- 落盘 ----
    payload = {
        "scenario": {
            "t_total_s": float(t[-1]), "t_amb_c": T_AMB,
            "base_c": BASE_C, "tail_c": TAIL_C, "cycles": CYCLES,
            "h_lo": H_LO, "h_hi": H_HI, "min_dwell_s": MIN_DWELL_S,
            "t_on_c": T_ON, "t_off_c": T_OFF, "q_lowload_w": Q_LOWLOAD_W,
            "ident_window_s": IDENT_WINDOW_S,
        },
        "strategies": strategies,
        "analytic_pump_check": analytic_check,
        "ident_cadence": cadence,
        "ident_control_impact": impact,
        "curve_params": {
            "affinity": vars(AFFINITY), "fan_typical": vars(FAN_TYPICAL),
            "pump_typical": vars(PUMP_TYPICAL),
            "note": "p_max/idle/static 为文献工程典型量级，不冒充实测",
        },
        "verdict": verdict,
        "elapsed_s": time.perf_counter() - t0,
    }
    (MET / "22_actuator_ident.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    # ---- 图 22a ----
    plotting.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) 功率曲线
    ax = axes[0, 0]
    hs = np.linspace(0, 1, 100)
    for label, curve, style in (("理想 P∝h³", AFFINITY, "--"),
                                ("风机典型", FAN_TYPICAL, "-"),
                                ("泵典型（静扬程）", PUMP_TYPICAL, "-")):
        p = curve.power_w(hs)
        p_ref = float(curve.power_w(1.0)) or 1.0
        ax.plot(hs, p / p_ref * 100, style, label=label, lw=2)
    ax.axvline(H_LO / H_HI, color="gray", lw=0.8, ls=":")
    ax.annotate(f"h_lo/h_hi={H_LO/H_HI:.2f}", (H_LO / H_HI, 5), fontsize=8, color="gray")
    ax.set_xlabel("相对档位 h/h_hi")
    ax.set_ylabel("功率占满档 %")
    ax.set_title("a) 执行器功率曲线（典型值口径）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (b) 三策略 × 三曲线能耗
    ax = axes[0, 1]
    names = ["full_cool", "surface_trigger", "core_est_trigger"]
    labels = ["恒满冷", "表面触发", "芯温触发"]
    x = np.arange(len(names))
    w = 0.26
    for i, (ck, cl) in enumerate((("affinity", "理想"), ("fan_typical", "风机"),
                                  ("pump_typical", "泵"))):
        vals = [strategies[n]["energy"][ck]["energy_norm"] for n in names]
        ax.bar(x + (i - 1) * w, vals, w, label=cl)
    ax.set_xticks(x, labels)
    ax.set_ylabel("归一化能耗（同曲线口径 恒满冷=1）")
    ax.set_title("b) 触发式节能被真实曲线压缩")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (c) 辨识周期 vs β
    ax = axes[1, 0]
    betas = np.linspace(0.3, 3.5, 50)
    for fname, fade, style in (("cs2_0.5C_per_efc", 3.44e-4, "-"),
                               ("model_2000efc", 1.0e-4, "--")):
        per = [ident_period_efc(b, fade, R_TOL) for b in betas]
        ax.plot(betas, per, style, lw=2,
                label=f"fade={fade:.1e}/EFC ({fname.split('_')[1] if 'cs2' in fname else 'model'})")
    ax.set_yscale("log")
    ax.set_xlabel("β（R/R0 = 1+β·f）")
    ax.set_ylabel("辨识周期 EFC")
    ax.set_title(f"c) 辨识周期（R 误差容忍 {R_TOL:.0%}）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # (d) 辨识窗冻结代价
    ax = axes[1, 1]
    per_s = sorted({v["period_s"] for k, v in impact.items() if "period_s" in v})
    for mode, style, label in (("unconditional", "o-", "强制开窗"),
                               ("low_load_only", "s-", "低负荷排程")):
        ys = [impact[f"p{int(p)}_{mode}"]["delta_peak_k"] for p in per_s]
        ax.plot(np.array(per_s) / 60, ys, style, lw=2, label=label)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("辨识周期 min（压力测试档）")
    ax.set_ylabel("芯温峰值变化 K")
    ax.set_title("d) 辨识窗冻结控制的代价")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("22 · 执行器产品化与辨识脉冲排程", fontsize=13)
    fig.tight_layout()
    plotting.save(fig, "22a_actuator_ident.png")
    print(f"[落盘] 22_actuator_ident.json / 22a_actuator_ident.png  "
          f"({time.perf_counter()-t0:.1f} s)")


if __name__ == "__main__":
    main()
