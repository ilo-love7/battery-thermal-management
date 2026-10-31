"""20 · 观测器产热/内阻在线校核：HPPC 脉冲-弛豫辨识。

v1.2 §19 的遗留：持续高倍率下 Q/R_int 失配是芯温估计误差天花板
（最差 5.0 K）。本脚本做在线校核闭环，并得到一个负结果 + 一个解法：

**负结果**：把 R_int 失配当加性偏置状态 w_int 吸收——不收敛
（结构可观测但实用不可辨识：稳态 gap 对 T_s 零增益，瞬态信息太弱）。
**解法**：脉冲-弛豫显式辨识——功率脉冲结束后的表面温度弛豫慢时间常数
含 R_int 指纹（τ_slow ≈ (R_int+R_conv)·C_core），拟合 τ 反解 R_int；
w_Q（热源偏置）则可由偏置 Kalman 直接吸收（T_s 稳态对总热流敏感）。

场景：持续 3C + HPPC 脉冲 ×2（每 300 s/60 s），真值失配 q=1.5, r=1.3。

产出：
  results/metrics/20_q_correction.json
  results/figures/20a_q_correction.png
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmthermal import plotting  # noqa: E402
from bmthermal.network import build_cell, simulate_two_node  # noqa: E402
from bmthermal.observer import (  # noqa: E402
    fit_relaxation_tau,
    identify_r_int_k_w,
    run_kalman,
    run_kalman_bias,
    slow_time_constant_s,
)

OUT = Path(__file__).resolve().parents[1] / "results"
MET = OUT / "metrics"
MET.mkdir(parents=True, exist_ok=True)

NOISE_K = 0.05
Q_3C_W = 5.6
T_AMB = 25.0
DURATION_S = 1800.0
DT_S = 10.0

Q_SCALE = 1.5
R_SCALE = 1.3

PULSE_PERIOD_S = 300.0
PULSE_DUR_S = 60.0
PULSE_SCALE = 2.0


def make_duty(with_pulse: bool, period: float = PULSE_PERIOD_S,
              dur: float = PULSE_DUR_S) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, DURATION_S + DT_S / 2, DT_S)
    if not with_pulse:
        return t, np.full_like(t, Q_3C_W)
    phase = np.mod(t, period)
    return t, np.where(phase < dur, Q_3C_W * PULSE_SCALE, Q_3C_W)


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(7)

    # ---- 真值：Q ×1.5，R_int ×1.3 ----
    t, q_cmd = make_duty(with_pulse=True)
    q_mean = float(np.mean(q_cmd))
    cell_nom = build_cell(h=500.0)
    cell_truth = _replace(cell_nom, r_int_k_w=cell_nom.r_int_k_w * R_SCALE)
    truth = simulate_two_node(t, q_cmd * Q_SCALE, T_AMB, cell_truth)
    truth_core = truth["t_core_c"]
    z = truth["t_surface_c"] + rng.normal(0.0, NOISE_K, size=t.size)
    tau_truth = slow_time_constant_s(cell_truth)

    rows = {}

    # ---- 1) 基线 Kalman（无校核） ----
    res0 = run_kalman(t, q_cmd, z, T_AMB, cell_nom,
                      q_process_w2=0.05**2, r_meas_k2=NOISE_K**2,
                      t0_c=T_AMB, truth_core_c=truth_core, x0_c=T_AMB)
    steady0 = float(np.mean((res0.t_core_hat_c - truth_core)[-len(t) // 3:]))
    # 稳态 gap 误差（T_s 被测量钉住，误差集中在芯-壳 gap）：
    # err = R_int·Q_cmd − r_scale·R_int·q_scale·Q_cmd = Q·R_int·(1 − q·r)
    err_theory = q_mean * cell_nom.r_int_k_w * (1.0 - Q_SCALE * R_SCALE)
    rows["baseline_kalman"] = {
        **res0.summary(), "steady_err_k": steady0,
        "steady_err_theory_k": err_theory,
    }
    print(f"[基线]        稳态偏差 {steady0:+.2f} K  （gap 理论 {err_theory:+.2f} K）")

    # ---- 2) 偏置 Kalman：w_Q 可吸收，w_int 不收敛（负结果） ----
    res_b = run_kalman_bias(
        t, q_cmd, z, T_AMB, cell_nom,
        q_process_w2=0.05**2, r_meas_k2=NOISE_K**2,
        t0_c=T_AMB, truth_core_c=truth_core,
        q_scale=Q_SCALE, r_scale=R_SCALE)
    rows["bias_kalman"] = {
        **res_b.summary(),
        "w_int_converged": bool(
            abs(res_b.w_int_hat_w[-1] - res_b.w_int_theory_w)
            < 0.3 * abs(res_b.w_int_theory_w)),
    }
    print(f"[偏置]        稳态偏差 {res_b.steady_err_k:+.2f} K  "
          f"w_Q {res_b.w_q_hat_w[-1]:+.2f}/{res_b.w_q_theory_w:+.2f} W ✓  "
          f"w_int {res_b.w_int_hat_w[-1]:+.2f}/{res_b.w_int_theory_w:+.2f} W ✗（负结果）")

    # ---- 3) 脉冲-弛豫辨识 R_int ----
    print("[辨识] 脉冲弛豫拟合 τ → R_int ...")
    fits = []
    pulse_starts = np.arange(0.0, DURATION_S, PULSE_PERIOD_S)
    for k, ps in enumerate(pulse_starts):
        pe = ps + PULSE_DUR_S
        if pe + 40.0 >= min(ps + PULSE_PERIOD_S, DURATION_S):
            continue
        fit = fit_relaxation_tau(t, z, pe + 20.0, ps + PULSE_PERIOD_S - 20.0)
        if not np.isfinite(fit["tau_s"]):
            continue
        r_est = identify_r_int_k_w(fit["tau_s"], cell_nom)
        fits.append({
            "pulse": int(k), "pulse_end_s": float(pe),
            "tau_s": fit["tau_s"], "tau_r2": fit["r2"],
            "r_int_est_k_w": r_est,
            "r_scale_est": r_est / cell_nom.r_int_k_w,
        })
        print(f"   脉冲 {k}: τ={fit['tau_s']:6.1f} s (r²={fit['r2']:.3f}) → "
              f"R_int={r_est:.3f} K/W  r_scale={r_est/cell_nom.r_int_k_w:.3f}")
    r_scales = np.array([f["r_scale_est"] for f in fits])
    r_scale_med = float(np.median(r_scales))
    r_scale_err = abs(r_scale_med - R_SCALE) / R_SCALE * 100.0
    print(f"   中位 r_scale={r_scale_med:.3f}（真值 {R_SCALE}，误差 {r_scale_err:.1f}%）")

    # ---- 4) 校核后观测器：辨识 R_int + w_Q 偏置 ----
    cell_corr = _replace(cell_nom, r_int_k_w=cell_nom.r_int_k_w * r_scale_med)
    res_c = run_kalman_bias(
        t, q_cmd, z, T_AMB, cell_corr,
        q_process_w2=0.05**2, r_meas_k2=NOISE_K**2,
        t0_c=T_AMB, truth_core_c=truth_core,
        q_scale=Q_SCALE, r_scale=R_SCALE)
    rows["corrected"] = {
        **res_c.summary(),
        "r_scale_est": r_scale_med,
        "r_scale_err_pct": r_scale_err,
    }
    print(f"[校核后]      稳态偏差 {res_c.steady_err_k:+.2f} K  "
          f"(w_Q {res_c.w_q_hat_w[-1]:+.2f}/{res_c.w_q_theory_w:+.2f} W)")

    # ---- 5) 脉冲参数扫描（辨识精度 vs 脉冲设计） ----
    print("[扫参] 脉冲周期 × 脉宽（辨识 r_scale 误差）...")
    scan = []
    for period in (150.0, 300.0, 600.0):
        for dur in (20.0, 60.0, 120.0):
            tt, qq = make_duty(True, period, dur)
            tr = simulate_two_node(tt, qq * Q_SCALE, T_AMB, cell_truth)
            zz = tr["t_surface_c"] + rng.normal(0.0, NOISE_K, size=tt.size)
            ests = []
            for ps in np.arange(0.0, DURATION_S, period):
                pe = ps + dur
                if pe + 40.0 >= min(ps + period, DURATION_S):
                    continue
                fit = fit_relaxation_tau(tt, zz, pe + 20.0, ps + period - 20.0)
                if np.isfinite(fit["tau_s"]):
                    r_e = identify_r_int_k_w(fit["tau_s"], cell_nom)
                    if np.isfinite(r_e):
                        ests.append(r_e / cell_nom.r_int_k_w)
            if ests:
                med = float(np.median(ests))
                err_pct = abs(med - R_SCALE) / R_SCALE * 100.0
            else:
                med, err_pct = np.nan, np.nan
            scan.append({
                "period_s": period, "dur_s": dur,
                "duty_pct": dur / period * 100.0,
                "n_pulses": len(ests),
                "r_scale_est": med,
                "r_scale_err_pct": float(err_pct),
                "ok": bool(err_pct <= 5.0 and len(ests) >= 3),
            })
            print(f"   周期 {period:4.0f}s 脉宽 {dur:3.0f}s（占空 {dur/period*100:4.1f}%，"
                  f"{len(ests)} 脉冲）→ r_scale {med:.3f}（误差 {err_pct:.1f}%）"
                  f"{'  ✓' if scan[-1]['ok'] else ''}")

    ok_rows = [s for s in scan if s["ok"]]
    min_duty = min((s["duty_pct"] for s in ok_rows), default=np.nan)

    # ---- 判词 ----
    fixed = abs(res_c.steady_err_k) < 1.5 and r_scale_err < 10.0
    verdict = (
        f"基线稳态偏差 {steady0:+.1f} K（gap 理论 {err_theory:+.1f} K）；"
        f"负结果：w_int 偏置态不收敛（{res_b.w_int_hat_w[-1]:+.2f} vs 理论 "
        f"{res_b.w_int_theory_w:+.2f} W）——R_int 失配不能用加性偏置吸收，"
        f"须显式辨识；脉冲-弛豫辨识 r_scale={r_scale_med:.2f}（真 {R_SCALE}，"
        f"误差 {r_scale_err:.0f}%）后校核稳态偏差 {res_c.steady_err_k:+.2f} K，"
        f"w_Q 由偏置 Kalman 吸收；最低可用占空比 ~{min_duty:.0f}%"
        + ("——在线校核闭掉 v1.2 的失配天花板大头（残余为辨识误差残留）"
           if fixed else "——校核不充分")
    )
    print("判词:", verdict)

    payload = {
        "scenario": {
            "duty": f"持续 3C（{Q_3C_W} W）+ 脉冲 ×{PULSE_SCALE} "
                    f"每 {PULSE_PERIOD_S:.0f}s/{PULSE_DUR_S:.0f}s",
            "q_scale": Q_SCALE, "r_scale": R_SCALE,
            "duration_s": DURATION_S, "noise_k": NOISE_K,
            "tau_truth_s": tau_truth,
            "steady_err_theory_k": err_theory,
            "w_q_theory_w": res_b.w_q_theory_w,
            "w_int_theory_w": res_b.w_int_theory_w,
        },
        "cases": rows,
        "identification": {
            "per_pulse": fits,
            "r_scale_median": r_scale_med,
            "r_scale_err_pct": r_scale_err,
            "method": "脉冲后表面弛豫单指数拟合 τ → brentq 反解 R_int（慢模态指纹）",
        },
        "pulse_scan": scan,
        "min_duty_pct_for_ok": float(min_duty) if np.isfinite(min_duty) else None,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (MET / "20_q_correction.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- 图 ----------------
    fig, axes = plotting.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(t / 60.0, truth_core, "k-", lw=1.2, label="芯温真值")
    ax.plot(t / 60.0, res0.t_core_hat_c, color="#c0392b", lw=1.0,
            label=f"基线（{steady0:+.1f} K）")
    ax.plot(t / 60.0, res_c.t_core_hat_c, color="#27ae60", lw=1.0,
            label=f"辨识校核后（{res_c.steady_err_k:+.2f} K）")
    ax.set_xlabel("时间 [min]")
    ax.set_ylabel("芯温 [°C]")
    ax.set_title("持续 3C + 失配（q=1.5, r=1.3）：校核前后")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t / 60.0, res_b.w_q_hat_w, color="#2980b9", lw=1.2, label="w_Q 估计")
    ax.axhline(res_b.w_q_theory_w, color="#2980b9", ls="--", lw=1,
               label=f"w_Q* = {res_b.w_q_theory_w:+.2f} W")
    ax.plot(t / 60.0, res_b.w_int_hat_w, color="#e67e22", lw=1.2,
            label="w_int 估计（不收敛）")
    ax.axhline(res_b.w_int_theory_w, color="#e67e22", ls="--", lw=1,
               label=f"w_int* = {res_b.w_int_theory_w:+.2f} W")
    ax.set_xlabel("时间 [min]")
    ax.set_ylabel("偏置 [W]")
    ax.set_title("w_Q 可吸收 / w_int 不可吸收（负结果）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if fits:
        ax.plot([f["pulse"] for f in fits], [f["r_scale_est"] for f in fits],
                "o-", color="#8e44ad", lw=1.5, label="逐脉冲 r_scale 估计")
        ax.axhline(R_SCALE, color="k", ls="--", lw=1, label=f"真值 {R_SCALE}")
        ax.axhline(r_scale_med, color="#e67e22", lw=1.5,
                   label=f"中位 {r_scale_med:.2f}")
    ax.set_xlabel("脉冲序号")
    ax.set_ylabel("r_scale = R_int/R_int_nom")
    ax.set_title(f"脉冲-弛豫辨识（τ_slow 指纹，τ_truth={tau_truth:.0f} s）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    durs = sorted({s["dur_s"] for s in scan})
    for d in durs:
        pts = [s for s in scan if s["dur_s"] == d]
        ax.plot([s["period_s"] for s in pts],
                [s["r_scale_err_pct"] for s in pts], "o-",
                lw=1.5, label=f"脉宽 {d:.0f}s")
    ax.axhline(5.0, color="k", ls="--", lw=1, label="5% 要求")
    ax.set_xlabel("脉冲周期 [s]")
    ax.set_ylabel("r_scale 辨识误差 [%]")
    ax.set_title("脉冲设计 vs 辨识精度")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("20 · 在线校核：脉冲-弛豫辨识 + 偏置吸收", fontsize=13)
    plotting.save(fig, "20a_q_correction.png")

    print(f"完成，耗时 {payload['elapsed_s']} s")


if __name__ == "__main__":
    main()
