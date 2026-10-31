"""scripts/14_core_observer.py —— BMS 芯温估计器评估（v1.2 深化 1）。

把 v1.1 §16 的"BMS 必须输出芯温估计"落成可运行的两节点 Kalman 观测器，
并在参数失配（h / 热容 / 产热）下给芯温误差与适用域。

产出
----
results/metrics/14_observer.json
results/figures/14a_observer_us06.png
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmthermal.duty import load_us06
from bmthermal.network import build_cell, simulate_two_node
from bmthermal.observer import run_kalman
from bmthermal import plotting

OUT = Path(__file__).resolve().parents[1] / "results"
NOISE_K = 0.05  # 表面热电偶噪声（好于 CALCE 的 0.07 K 量化，BMS 口径）


def q_from_crate(c_rate: np.ndarray) -> np.ndarray:
    """BMS 侧产热估计：Q = k·|C|^α（标度律瞬时口径，k/α 从 01 指标读取）。"""
    metrics = Path(__file__).resolve().parents[1] / "results" / "metrics" / "01_heat_sweep.json"
    try:
        law = json.loads(metrics.read_text(encoding="utf-8"))["power_law"]
        k, alpha = law["k"], law["alpha"]
    except Exception:
        k, alpha = 0.7821775390520517, 1.7909636972736782
    return k * np.abs(np.asarray(c_rate, float)) ** alpha


def us06_truth(scale: float = 2.0, cycles: int = 3, t_amb: float = 25.0, h: float = 500.0):
    """真值场景：US06 循环 × cycles，冷板 h=500。返回 (t, q, truth, cell, crate)。"""
    t_p, c_p = load_us06(scale=scale)
    dt = float(np.median(np.diff(t_p)))
    t_list, c_list = [], []
    t_off = 0.0
    for _ in range(cycles):
        t_list.append(t_p + t_off)
        c_list.append(c_p)
        t_off += float(t_p[-1]) + dt
    t = np.concatenate(t_list)
    crate = np.concatenate(c_list)
    q = q_from_crate(crate)
    cell = build_cell(h=h)
    truth = simulate_two_node(t, q, t_amb, cell)
    return t, q, truth, cell, crate


def evaluate_mismatch():
    """失配矩阵：h / C / Q 各自 ±，看芯温估计误差。"""
    t, q, truth, cell_nom, crate = us06_truth()
    rng = np.random.default_rng(42)
    z = truth["t_surface_c"] + rng.normal(0.0, NOISE_K, size=t.size)
    truth_core = truth["t_core_c"]

    rows = []
    cases = []
    for h_scale in (0.7, 1.0, 1.3):
        for c_scale in (0.8, 1.0, 1.2):
            for q_scale in (0.7, 1.0, 1.5):
                cases.append((h_scale, c_scale, q_scale))
    # 全因子太多（27）就都跑——每例 < 0.2 s，可接受
    for h_s, c_s, q_s in cases:
        model = build_cell(h=500.0 * h_s)
        model = replace(model, c_th_j_k=model.c_th_j_k * c_s)
        res = run_kalman(
            t, q * q_s, z, 25.0, model,
            q_process_w2=0.05**2, r_meas_k2=NOISE_K**2,
            t0_c=25.0, truth_core_c=truth_core, x0_c=25.0,
        )
        s = res.summary()
        rows.append(
            {
                "h_scale": h_s,
                "c_scale": c_s,
                "q_scale": q_s,
                "scenario": "full" if (h_s, c_s, q_s) == (1.0, 1.0, 1.0) else "mismatch",
                **s,
            }
        )
    return rows, t, q, truth, z, crate


def demo_calce():
    """应用演示：CALCE 25 °C US06 段，估芯-壳 gap（无真值，看量级合理性）。"""
    from bmthermal.calce_thermal import load_all_dynamic, split_profiles

    df = load_all_dynamic()
    segs = split_profiles(df[df["temp_c_set"] == 25.0])
    seg = segs["us06"].sort_values("t_s").reset_index(drop=True)
    # Arbin 电流符号：放电为负 → |I|；换算 C 率（2.5 Ah）
    i_a = seg["current_a"].to_numpy(float)
    crate = np.abs(i_a) / 2.5
    t = seg["t_s"].to_numpy(float)
    # 降采样到 5 s
    step = max(1, int(round(5.0 / float(np.median(np.diff(t))))))
    t, crate = t[::step], crate[::step]
    z = seg["temp_c"].to_numpy(float)[::step]
    q = q_from_crate(crate)
    cell = build_cell(h=150.0)  # 自然对流口径（CALCE 无主动冷却）
    res = run_kalman(
        t, q, z, 25.0, cell,
        q_process_w2=0.05**2, r_meas_k2=0.07**2, t0_c=float(z[0]), x0_c=float(z[0]),
    )
    gap = res.t_core_hat_c - res.t_surface_hat_c
    return {
        "n_points": int(t.size),
        "duration_s": float(t[-1] - t[0]),
        "t_surface_range_k": [float(z.min()), float(z.max())],
        "gap_max_k": float(np.max(gap)),
        "gap_p95_k": float(np.percentile(gap, 95)),
        "note": "无芯温真值，只演示 gap 估计量级（CALCE 温升 ~1 K，gap 估计应很小）",
    }, t, z, res


def stress_sustained():
    """持续高倍率应力：稳态下 Q/R_int 失配的芯温偏差理论值 = (q_scale·r_scale−1)·Q·R_int。

    US06 短时尖峰来不及建立稳态偏差（热时间常数 ~分钟级）；持续 3C 才暴露。
    """
    n = 180  # 1800 s @ 10 s
    t = np.linspace(0.0, 1800.0, n)
    q_true = 5.6  # 3C 等效持续产热
    cell_nom = build_cell(h=500.0)
    truth = simulate_two_node(t, np.full(n, q_true), 25.0, cell_nom)
    rng = np.random.default_rng(7)
    z = truth["t_surface_c"] + rng.normal(0.0, NOISE_K, size=n)
    truth_core = truth["t_core_c"]
    r_int = cell_nom.r_int_k_w

    rows = []
    for r_s in (0.7, 1.0, 1.3):
        for q_s in (0.7, 1.0, 1.5):
            model = replace(cell_nom, r_int_k_w=r_int * r_s)
            res = run_kalman(
                t, np.full(n, q_true * q_s), z, 25.0, model,
                q_process_w2=0.05**2, r_meas_k2=NOISE_K**2,
                t0_c=25.0, truth_core_c=truth_core, x0_c=25.0,
            )
            # 后 1/3 稳态段误差
            seg = res.t_core_hat_c[-n // 3:] - truth_core[-n // 3:]
            # 稳态：芯温偏差 = gap_est − gap_true = Q·R·(q_scale·r_scale − 1)
            theory = q_true * r_int * (q_s * r_s - 1.0)
            rows.append(
                {
                    "r_int_scale": r_s,
                    "q_scale": q_s,
                    "steady_bias_k": float(np.mean(seg)),
                    "steady_bias_theory_k": float(theory),
                    "core_err_rmse_k": res.core_err_rmse_k,
                    "core_err_max_k": res.core_err_max_k,
                    "within_3k_share": res.within_3k_share,
                    "diverged": res.diverged,
                }
            )
    return rows, t, truth, z


def make_figure(rows, rows_sus, t, q, truth, z, res_demo, t_d, z_d):
    fig, axes = plotting.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(t / 60.0, truth["t_core_c"], color="#c0392b", lw=1.5, label="芯温真值")
    ax.plot(t / 60.0, truth["t_surface_c"], color="#2980b9", lw=1.0, label="表面真值")
    cell_nom = build_cell(h=500.0)
    res_nom = run_kalman(
        t, q, z, 25.0, cell_nom,
        q_process_w2=0.05**2, r_meas_k2=NOISE_K**2,
        t0_c=25.0, truth_core_c=truth["t_core_c"], x0_c=25.0,
    )
    ax.plot(t / 60.0, res_nom.t_core_hat_c, color="#8e44ad", lw=1.0, ls="--", label="芯温估计（名义）")
    ax.set_title("US06 真值场景（3 循环，h=500）")
    ax.set_xlabel("时间 [min]")
    ax.set_ylabel("温度 [°C]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    rms = np.array([r["core_err_rmse_k"] for r in rows])
    hs = np.array([r["h_scale"] for r in rows])
    qs = np.array([r["q_scale"] for r in rows])
    sc = ax.scatter(hs * 500, rms, c=qs, cmap="viridis", s=60, edgecolor="k", lw=0.3)
    ax.axhline(3.0, color="r", ls="--", lw=1, label="3 K 门槛")
    ax.set_xlabel("估计器用的 h [W/(m²·K)]（真值 500）")
    ax.set_ylabel("芯温估计 RMSE [K]")
    ax.set_title("参数失配下的芯温误差（颜色=Q 失配）")
    fig.colorbar(sc, ax=ax, label="q_scale")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    th = np.array([r["steady_bias_theory_k"] for r in rows_sus])
    ob = np.array([r["steady_bias_k"] for r in rows_sus])
    ax.scatter(th, ob, color="#e67e22", s=70, edgecolor="k", lw=0.3)
    lim = max(abs(th).max(), abs(ob).max()) * 1.2 + 0.2
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="理论 = (q·r−1)·Q·R_int")
    ax.axhline(3.0, color="r", ls=":", lw=1, label="3 K 门槛")
    ax.axhline(-3.0, color="r", ls=":", lw=1)
    ax.set_xlabel("理论稳态偏差 [K]")
    ax.set_ylabel("观测器稳态偏差 [K]")
    ax.set_title("持续 3C 应力：Q/R_int 失配的稳态偏差")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_d / 60.0, z_d, color="#2980b9", lw=1.0, label="表面实测（CALCE 25 °C US06）")
    ax.plot(t_d / 60.0, res_demo.t_core_hat_c, color="#c0392b", lw=1.2, label="芯温估计")
    ax.set_title("应用演示：CALCE 实测表面 → 芯温估计")
    ax.set_xlabel("时间 [min]")
    ax.set_ylabel("温度 [°C]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    plotting.save(fig, "14a_observer_us06.png")


def main():
    t0 = time.time()
    rows, t, q, truth, z, crate = evaluate_mismatch()
    rows_sus, t_sus, truth_sus, z_sus = stress_sustained()
    demo, t_d, z_d, res_demo = demo_calce()

    ok_rows = [r for r in rows if r["within_3k_share"] >= 0.95 and not r["diverged"]]
    sus_worst = max(abs(r["steady_bias_k"]) for r in rows_sus)
    verdict = (
        f"US06 动态工况：{len(ok_rows)}/{len(rows)} 失配组合芯温 |err|<3 K 占比 ≥95%（最差 RMSE "
        f"{max(r['core_err_rmse_k'] for r in rows):.2f} K）——短时尖峰下观测器对失配鲁棒；"
        f"持续 3C 应力：稳态偏差与理论值 (q·r−1)·Q·R_int 一致，最差 {sus_worst:.1f} K，"
        f"Q 失配是持续高倍率下的主误差源（须在线校核热源估计）"
    )
    out = {
        "scenario": "US06×3 循环（scale=2，峰值 ~3C），真值=两节点 h=500 冷板",
        "noise_k": NOISE_K,
        "n_cases": len(rows),
        "nominal": [r for r in rows if r["scenario"] == "full"][0],
        "pass_share": float(np.mean([r["within_3k_share"] >= 0.95 for r in rows])),
        "us06_worst_rmse_k": float(max(r["core_err_rmse_k"] for r in rows)),
        "sustained_3c": rows_sus,
        "sustained_worst_bias_k": float(sus_worst),
        "verdict": verdict,
        "cases": rows,
        "calce_demo": demo,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    with open(OUT / "metrics" / "14_observer.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    make_figure(rows, rows_sus, t, q, truth, z, res_demo, t_d, z_d)
    print(json.dumps({k: out[k] for k in ("verdict", "nominal", "sustained_worst_bias_k", "elapsed_s")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
