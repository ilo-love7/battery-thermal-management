"""21 · BMS 热管理控制闭环：滞环控制 × 芯温估计 vs 表面触发。

v1.2 §16 的"表面温度掩蔽芯部"风险，在这里是**闭环后果**而不只是估计误差：
表面触发的滞环控制器要等表面升温才开冷，而此时芯温已越过表面读数一个
芯-壳 gap（准稳态 ~Q·R_int，本场景均值 ~4 K）；芯温估计触发（观测器兑现
价值）直接按芯温干预，不背这个 gap。

工况（场景化，非产品设定值）
--------------------------
**持续 2C 基线 + US06 归一纹波** 的复合高负荷段（2 h）：
纯 US06 的均值热负荷（~0.4 W）不足以形成控制问题（开环 36 °C，探针证实
问题退化），故叠加 2C 持续基线使低档开环芯温越过判据区；US06 纹波保留
动态冲击（Q 峰值 ~7 W）。产热按 §1 标度律 q_from_crate（k·|C|^α）瞬时口径。

对比策略：
1. 恒满冷（h_hi 常开）——能耗上限基准（归一化 = 1.0）；
2. 表面温度触发 bang-bang——§16 掩蔽风险路径；
3. **芯温估计触发** bang-bang——观测器价值的直接兑现；
4.（鲁棒性）芯温估计触发 + 植株失配（Q×1.3, R_int×1.2，模型名义）。

阈值口径：固定 t_on = 判据 − 2 K = 43 °C，滞环带 3 K（同一组设定施加在
表面或芯温估计上——"同一 BMS 阈值设定"是公平对比）。
执行器约束：最短保持 20 s（表面触发在快表面节点+强档位下会以采样率抖振，
无保持时实测 800+ 次/2 h；芯温触发天然无抖振）。
另做两族阈值扫描：能耗-峰值 Pareto + 匹配峰值温度的能耗/损伤代价。

能耗口径：P ∝ h³（相似定律），归一化到恒满冷 = 1.0。

产出：
  results/metrics/21_thermal_control.json
  results/figures/21a_thermal_control.png
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
from bmthermal.control import (  # noqa: E402
    always_on,
    bang_bang,
    simulate_control_loop,
)
from bmthermal.network import build_cell  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MET = OUT / "metrics"
MET.mkdir(parents=True, exist_ok=True)

# ---- 场景 ----
T_AMB = 30.0            # 热环境
BASE_C = 2.0            # 持续基线倍率（高负荷段）
RIPPLE_SCALE = 1.0      # US06 归一纹波幅值（峰值 +1.46C）
CYCLES = 12             # US06 纹波周期数（600 s × 12 = 2 h）
H_LO = 25.0             # 低档（微风/泵低转，自然对流口径）
H_HI = 500.0            # 高档（液冷板中档，cold_plate h_mid）
NOISE_K = 0.05          # 表面热电偶
T_LIMIT_C = 45.0        # 项目判据（芯温）
T_ON = 43.0             # 阈值：判据 − 2 K（示例设定）
BAND_K = 3.0            # 滞环带宽
T_OFF = T_ON - BAND_K

Q_MISMATCH = 1.3        # 鲁棒性案例：植株产热失配
R_MISMATCH = 1.2        # 鲁棒性案例：植株 R_int 失配
MIN_DWELL_S = 20.0      # 执行器最短保持时间（泵/阀寿命；表面触发防抖振）


def q_from_crate(c_rate: np.ndarray) -> np.ndarray:
    """BMS 侧产热估计：Q = k·|C|^α（§1 标度律，k/α 从 01 指标读取）。"""
    metrics = Path(__file__).resolve().parents[1] / "results" / "metrics" / "01_heat_sweep.json"
    try:
        law = json.loads(metrics.read_text(encoding="utf-8"))["power_law"]
        k, alpha = law["k"], law["alpha"]
    except Exception:
        k, alpha = 0.7821775390520517, 1.7909636972736782
    return k * np.abs(np.asarray(c_rate, float)) ** alpha


def make_duty():
    """持续 BASE_C 基线 + US06 归一纹波 × CYCLES 的 Q(t)。"""
    from bmthermal.duty import load_us06

    t_p, c_p = load_us06(scale=RIPPLE_SCALE)
    dt = float(np.median(np.diff(t_p)))
    t_list, c_list = [], []
    t_off = 0.0
    for _ in range(CYCLES):
        t_list.append(t_p + t_off)
        c_list.append(c_p)
        t_off += float(t_p[-1]) + dt
    t = np.concatenate(t_list)
    crate = BASE_C + np.concatenate(c_list)   # 基线 + 纹波（全程放电口径）
    q = q_from_crate(crate)
    return t, q, crate


def summarize(res, t_on: float | None = None, t_off: float | None = None) -> dict:
    over = res.t_core_true_c > T_LIMIT_C
    dt_arr = np.diff(res.t_s, prepend=res.t_s[0])
    dt_arr[0] = 0.0
    return {
        "energy_norm": res.energy_norm,
        "duty_high": res.duty_high,
        "n_switch": res.n_switch,
        "t_max_core_c": res.t_max_core_c,
        "t_mean_core_c": float(res.t_core_true_c.mean()),
        "t_max_surface_c": float(res.t_surface_true_c.max()),
        "damage_s": res.damage_s,
        "core_err_rmse_k": res.core_err_rmse_k,
        "time_over_limit_s": float(dt_arr[over].sum()) if over.any() else 0.0,
        "t_on_c": t_on, "t_off_c": t_off,
        "meta": res.meta,
    }


def run_case(t, q, law, cell_true, cell_model, initial_mode=0, seed=0,
             q_true_scale=1.0, min_dwell_s=MIN_DWELL_S):
    return simulate_control_loop(
        t, q, T_AMB, cell_true, cell_model,
        h_lo=H_LO, h_hi=H_HI, law=law,
        noise_k=NOISE_K, seed=seed, initial_mode=initial_mode,
        q_true_scale=q_true_scale, min_dwell_s=min_dwell_s,
    )


def main() -> None:
    t0 = time.time()
    t, q, crate = make_duty()
    q_peak, q_mean = float(q.max()), float(q.mean())
    print(f"工况：{BASE_C}C 基线 + US06×{CYCLES}，{t[-1]/60:.0f} min，"
          f"Q 峰值 {q_peak:.2f} W / 均值 {q_mean:.2f} W，T_amb={T_AMB:.0f} °C")

    cell_model = build_cell(h=H_HI)                       # BMS 名义模型（高档）
    cell_true = cell_model                                # 主对比：植株=模型
    cell_true_bad = _replace(cell_true, r_int_k_w=cell_true.r_int_k_w * R_MISMATCH)
    gap_mean = q_mean * cell_true.r_int_k_w

    strategies = {}

    # ---- 1) 恒满冷 ----
    res_full = run_case(t, q, always_on(1), cell_true, cell_model, initial_mode=1)
    strategies["always_on"] = summarize(res_full)
    print(f"[恒满冷]   能耗 {res_full.energy_norm:.3f}  峰值 {res_full.t_max_core_c:.2f} °C  "
          f"损伤 {res_full.damage_s:.0f} s")

    # ---- 2) 表面温度触发（同一组阈值施加在表面上） ----
    res_surf = run_case(t, q, bang_bang(T_ON, T_OFF, use_estimate=False),
                        cell_true, cell_model, seed=1)
    strategies["surface_trigger"] = summarize(res_surf, T_ON, T_OFF)
    print(f"[表面触发] 能耗 {res_surf.energy_norm:.3f}  峰值 {res_surf.t_max_core_c:.2f} °C  "
          f"超限 {strategies['surface_trigger']['time_over_limit_s']:.0f} s  "
          f"切换 {res_surf.n_switch} 次")

    # ---- 2b) 无最短保持的表面触发（抖振诊断，不进主对比） ----
    res_surf_nd = run_case(t, q, bang_bang(T_ON, T_OFF, use_estimate=False),
                           cell_true, cell_model, seed=1, min_dwell_s=0.0)
    strategies["surface_trigger_no_dwell"] = summarize(res_surf_nd, T_ON, T_OFF)
    print(f"[表面·无保持] 切换 {res_surf_nd.n_switch} 次（抖振：快表面节点+强档位打穿滞环带）")

    # ---- 3) 芯温估计触发 ----
    res_core = run_case(t, q, bang_bang(T_ON, T_OFF, use_estimate=True),
                        cell_true, cell_model, seed=1)
    strategies["core_estimate_trigger"] = summarize(res_core, T_ON, T_OFF)
    print(f"[芯温触发] 能耗 {res_core.energy_norm:.3f}  峰值 {res_core.t_max_core_c:.2f} °C  "
          f"超限 {strategies['core_estimate_trigger']['time_over_limit_s']:.0f} s  "
          f"切换 {res_core.n_switch} 次")

    # ---- 4) 芯温触发 + 植株失配（鲁棒性） ----
    res_bad = run_case(t, q, bang_bang(T_ON, T_OFF, use_estimate=True),
                       cell_true_bad, cell_model, seed=1, q_true_scale=Q_MISMATCH)
    strategies["core_trigger_mismatch"] = summarize(res_bad, T_ON, T_OFF)
    strategies["core_trigger_mismatch"]["mismatch"] = {
        "q_scale": Q_MISMATCH, "r_scale": R_MISMATCH}
    print(f"[芯温+失配] 能耗 {res_bad.energy_norm:.3f}  峰值 {res_bad.t_max_core_c:.2f} °C  "
          f"估计 RMSE {res_bad.core_err_rmse_k:.2f} K")

    # ---- 两族阈值扫描：能耗-峰值 Pareto ----
    print("[扫参] 表面/芯温触发阈值扫描（能耗-峰值 Pareto）...")
    sweeps = {"surface": [], "core": []}
    for fam, use_est in (("surface", False), ("core", True)):
        for on in np.arange(T_ON - 3.0, T_ON + 2.01, 0.5):
            off = on - BAND_K
            r = run_case(t, q, bang_bang(float(on), float(off), use_est),
                         cell_true, cell_model, seed=1)
            sweeps[fam].append({
                "t_on_c": float(on), "t_off_c": float(off),
                "energy_norm": r.energy_norm,
                "t_max_core_c": r.t_max_core_c,
                "t_mean_core_c": float(r.t_core_true_c.mean()),
                "damage_s": r.damage_s,
                "duty_high": r.duty_high,
            })
        fam_s = sweeps[fam]
        print(f"   {fam}: 能耗 {min(s['energy_norm'] for s in fam_s):.3f}–"
              f"{max(s['energy_norm'] for s in fam_s):.3f}，"
              f"峰值 {min(s['t_max_core_c'] for s in fam_s):.2f}–"
              f"{max(s['t_max_core_c'] for s in fam_s):.2f} °C")

    # 匹配峰值温度：表面族插值到芯温触发主案例的芯温峰值
    target_tmax = res_core.t_max_core_c
    fam_s = sorted(sweeps["surface"], key=lambda s: s["t_max_core_c"])
    xs = np.array([s["t_max_core_c"] for s in fam_s])
    ys_e = np.array([s["energy_norm"] for s in fam_s])
    ys_d = np.array([s["damage_s"] for s in fam_s])
    if xs.min() <= target_tmax <= xs.max() and np.all(np.diff(xs) > 0):
        e_surf_m = float(np.interp(target_tmax, xs, ys_e))
        d_surf_m = float(np.interp(target_tmax, xs, ys_d))
    else:
        i = int(np.argmin(np.abs(xs - target_tmax)))
        e_surf_m, d_surf_m = float(ys_e[i]), float(ys_d[i])
    energy_penalty_pct = (e_surf_m / res_core.energy_norm - 1.0) * 100.0 \
        if res_core.energy_norm > 0 else float("nan")
    damage_penalty_pct = (d_surf_m / res_core.damage_s - 1.0) * 100.0 \
        if res_core.damage_s > 0 else float("nan")
    print(f"   匹配峰值 {target_tmax:.2f} °C：表面触发 能耗 {e_surf_m:.3f} / 损伤 {d_surf_m:.0f} s"
          f" vs 芯温触发 {res_core.energy_norm:.3f} / {res_core.damage_s:.0f} s"
          f" → 能耗代价 {energy_penalty_pct:+.0f}%，损伤代价 {damage_penalty_pct:+.0f}%")

    # ---- 判词 ----
    same_setting_gap = res_surf.t_max_core_c - res_core.t_max_core_c
    surf_viol = strategies["surface_trigger"]["time_over_limit_s"]
    core_viol = strategies["core_estimate_trigger"]["time_over_limit_s"]
    robust = res_bad.t_max_core_c <= T_LIMIT_C + 1.0
    verdict = (
        f"同一阈值设定（t_on={T_ON:.0f}°C）：表面触发芯温峰值 {res_surf.t_max_core_c:.1f} °C"
        f"（超限 {surf_viol:.0f} s），芯温估计触发 {res_core.t_max_core_c:.1f} °C"
        f"（超限 {core_viol:.0f} s），峰值差 {same_setting_gap:.1f} K ≈ 掩蔽 gap（均值 {gap_mean:.1f} K）；"
        f"表面触发在快表面节点+强档位下抖振（无最短保持时 {res_surf_nd.n_switch} 次/"
        f"{t[-1]/60:.0f} min），加最短保持 {MIN_DWELL_S:.0f} s 后 {res_surf.n_switch} 次，"
        f"芯温触发天然无抖振（{res_core.n_switch} 次）；"
        f"匹配峰值温度下表面触发能耗代价 {energy_penalty_pct:+.0f}%、损伤代价 {damage_penalty_pct:+.0f}%"
        f"（能耗非主要差异，合规与温度水平才是）；"
        f"失配 Q×{Q_MISMATCH}/R×{R_MISMATCH} 下芯温触发峰值 {res_bad.t_max_core_c:.1f} °C"
        + ("——观测器接控制闭环兑现了 §16 的芯温估计价值：同设定合规，"
           "阈值不必为 gap 预留过度余量" if (same_setting_gap > 0.5 or core_viol < surf_viol) and robust
           else "——优势不显著，需再调场景")
    )
    print("判词:", verdict)

    payload = {
        "scenario": {
            "duty": f"持续 {BASE_C}C 基线 + US06×{CYCLES} 归一纹波（q_from_crate 标度律）",
            "t_amb_c": T_AMB, "h_lo": H_LO, "h_hi": H_HI,
            "noise_k": NOISE_K, "duration_s": float(t[-1]),
            "c_rate_range": [float(crate.min()), float(crate.max())],
            "q_peak_w": q_peak, "q_mean_w": q_mean,
            "gap_mean_k": gap_mean,
            "gap_peak_quasisteady_k": q_peak * cell_true.r_int_k_w,
            "t_limit_c": T_LIMIT_C,
            "threshold_note": f"t_on = 判据 − 2 K = {T_ON} °C，滞环带 {BAND_K} K，"
                              "同一组设定施加在表面或芯温估计上",
            "energy_model": "P ∝ h³（相似定律），恒满冷=1.0",
            "min_dwell_s": MIN_DWELL_S,
            "chatter_note": (
                f"表面触发无最短保持时抖振 {res_surf_nd.n_switch} 次/"
                f"{t[-1]/60:.0f} min（壳节点 ~9 J/K，高档降温 ~5 K/s 打穿 3 K 滞环带）；"
                f"最短保持 {MIN_DWELL_S:.0f} s 后 {res_surf.n_switch} 次"
            ),
        },
        "thresholds": {"t_on_c": T_ON, "t_off_c": T_OFF},
        "strategies": strategies,
        "sweeps": sweeps,
        "matched_temperature": {
            "target_t_max_core_c": target_tmax,
            "surface_energy_norm": e_surf_m,
            "core_energy_norm": res_core.energy_norm,
            "energy_penalty_pct": energy_penalty_pct,
            "surface_damage_s": d_surf_m,
            "core_damage_s": res_core.damage_s,
            "damage_penalty_pct": damage_penalty_pct,
        },
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (MET / "21_thermal_control.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- 图 ----------------
    fig, axes = plotting.subplots(2, 2, figsize=(12, 8))
    t_min = t / 60.0

    ax = axes[0, 0]
    ax.plot(t_min, res_full.t_core_true_c, color="#7f8c8d", lw=1.0, label="恒满冷")
    ax.plot(t_min, res_surf.t_core_true_c, color="#c0392b", lw=1.0,
            label=f"表面触发（{res_surf.t_max_core_c:.1f} °C）")
    ax.plot(t_min, res_core.t_core_true_c, color="#27ae60", lw=1.0,
            label=f"芯温估计触发（{res_core.t_max_core_c:.1f} °C）")
    ax.axhline(T_LIMIT_C, color="k", ls="--", lw=1, label=f"判据 {T_LIMIT_C:.0f} °C")
    ax.axhline(T_ON, color="#2980b9", ls=":", lw=1, label=f"t_on={T_ON:.0f}")
    ax.set_xlabel("时间 [min]")
    ax.set_ylabel("芯温真值 [°C]")
    ax.set_title(f"控制闭环芯温：表面触发被 gap（均值 {gap_mean:.1f} K）掩蔽")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_min, res_surf.t_surface_true_c, color="#c0392b", lw=1.0,
            label="表面触发：表面真值")
    ax.plot(t_min, res_core.t_core_hat_c, color="#27ae60", lw=1.0,
            label="芯温触发：芯温估计")
    ax.step(t_min, res_core.h_cmd / H_HI * 20 + 30, where="post", color="#27ae60",
            lw=0.8, alpha=0.5, label="芯温触发档位（缩放）")
    ax.axhline(T_ON, color="#2980b9", ls=":", lw=1, label=f"t_on={T_ON:.0f}")
    ax.axhline(T_LIMIT_C, color="k", ls="--", lw=1, label=f"判据 {T_LIMIT_C:.0f}")
    ax.set_xlabel("时间 [min]")
    ax.set_ylabel("温度 [°C]")
    ax.set_title("触发信号：表面滞后 vs 芯温估计（同阈值）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for fam, color, lab in (("surface", "#c0392b", "表面触发"),
                            ("core", "#27ae60", "芯温估计触发")):
        pts = sorted(sweeps[fam], key=lambda s: s["t_max_core_c"])
        ax.plot([s["t_max_core_c"] for s in pts],
                [s["energy_norm"] for s in pts], "o-", color=color, lw=1.5, label=lab)
    ax.plot(res_full.t_max_core_c, res_full.energy_norm, "s", color="#7f8c8d",
            ms=9, label="恒满冷")
    ax.axvline(T_LIMIT_C, color="k", ls="--", lw=1, label=f"判据 {T_LIMIT_C:.0f} °C")
    ax.set_xlabel("芯温峰值 [°C]")
    ax.set_ylabel("归一化冷却能耗（恒满冷=1）")
    ax.set_title(f"能耗-峰值 Pareto（同峰值能耗代价 {energy_penalty_pct:+.0f}%）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    names = ["恒满冷", "表面触发", "芯温估计", "芯温+失配"]
    keys = ["always_on", "surface_trigger", "core_estimate_trigger",
            "core_trigger_mismatch"]
    damage = [strategies[k]["damage_s"] for k in keys]
    over = [strategies[k]["time_over_limit_s"] for k in keys]
    x = np.arange(len(names))
    ax.bar(x - 0.2, damage, width=0.4, color="#8e44ad", label="损伤积分 [s]")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, over, width=0.4, color="#e67e22", label="超 45 °C 时长 [s]")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("损伤积分（25 °C 等效秒）")
    ax2.set_ylabel("超限时长 [s]")
    ax.set_title("损伤与超限：策略代价")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle("21 · BMS 控制闭环：芯温估计触发 vs 表面触发", fontsize=13)
    plotting.save(fig, "21a_thermal_control.png")

    print(f"完成，耗时 {payload['elapsed_s']} s")


if __name__ == "__main__":
    main()
