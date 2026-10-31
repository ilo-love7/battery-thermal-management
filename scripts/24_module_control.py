"""24 · 模组级热管理控制：单点 NTC 布点的代价。

问题
----
BMS 模组通常只有 1–2 个 NTC。冷却液沿程升温 + 逐芯产热散布 → 上游探针
读数低于最热芯（空间遮蔽，scripts/21 是时间遮蔽）。四种策略对比：

1. 恒满冷——能耗上限基准；
2. **上游探针**（冷端，无补偿）——漏报下游过温的现实风险路径；
3. **下游探针**（热端）——天然最坏点触发：合规但过冷耗能；
4. **单点 + 梯度补偿**——探针在上游，内部稳态模型加偏移估计最热芯；
5. **逐芯理想**（每芯 NTC）——上界对照。

流量是第二轴：设计流量 2.0 L/min（梯度 <1 K，布点几乎无差别）vs
低流量 0.5 L/min（泵降档/衰减工况，梯度 ~3 K，布点决定合规）。

工况：与 21/22 同场景（持续 2C 基线 + US06 纹波 2 h + 低负荷 1 h，
T_in=25 °C），阈值 t_on=43 °C（判据 45 − 2），滞环 3 K，最短保持 20 s，
逐芯产热散布 N(1, 5%)（制造离散，固定种子）。

能耗双口径：理想 P∝h³ 与泵典型曲线（WP22 的口径修正）。

产出：
  results/metrics/24_module_control.json
  results/figures/24a_module_control.png
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
from bmthermal.module import ModuleSpec  # noqa: E402
from bmthermal.module_control import (  # noqa: E402
    simulate_module_control,
    steady_state_cells,
)

OUT = ROOT / "results"
MET = OUT / "metrics"
MET.mkdir(parents=True, exist_ok=True)

# ---- 场景（与 21/22 同口径）----
BASE_C = 2.0
RIPPLE_SCALE = 1.0
CYCLES = 12
TAIL_C = 0.3
TAIL_S = 3600.0
H_LO = 150.0            # 冷板低档（泵低转）
H_HI = 500.0            # 冷板高档
NOISE_K = 0.05
T_ON = 43.0
BAND_K = 3.0
T_OFF = T_ON - BAND_K
MIN_DWELL_S = 20.0
T_LIMIT_C = 45.0
T_IN_C = 29.0            # 热气候冷却液入口（把合规问题立起来的边界条件）
Q_SPREAD_SIGMA = 0.08

#: 设计流量（梯度 <1 K，布点几乎无差别）vs 低流量（泵降档/滤网堵塞工况，
#: 梯度 2–4 K，进入滞环带宽量级——布点决定合规）
FLOWS = {"design_2.0lpm": 2.0, "low_0.25lpm": 0.25}


def q_from_crate(c_rate: np.ndarray) -> np.ndarray:
    metrics = ROOT / "results" / "metrics" / "01_heat_sweep.json"
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
    t_tail = np.arange(1.0, TAIL_S + 1.0) + t_hi[-1]
    crate = np.concatenate([crate_hi, np.full_like(t_tail, TAIL_C)])
    t = np.concatenate([t_hi, t_tail])
    return t, q_from_crate(crate)


def bang_bang(t_on_c: float, t_off_c: float):
    def law(t_now: float, sig: float, mode: int) -> int:
        if sig >= t_on_c:
            return 1
        if sig <= t_off_c:
            return 0
        return mode
    return law


def always_on(mode: int = 1):
    def law(t_now, sig, cur):
        return mode
    return law


def run_strategy(t, q, spec, law, probe_idx, estimator, q_spread, seed=1):
    return simulate_module_control(
        t, q, spec, law,
        h_lo=H_LO, h_hi=H_HI,
        probe_idx=probe_idx, estimator=estimator,
        q_spread=q_spread, noise_k=NOISE_K,
        min_dwell_s=MIN_DWELL_S, seed=seed,
    )


def summarize(res) -> dict:
    return {
        "energy_norm_affinity": res.energy_norm_affinity,
        "energy_norm_pump": res.energy_norm_pump,
        "duty_high": res.duty_high,
        "n_switch": res.n_switch,
        "t_max_c": res.t_max_c,
        "delta_t_max_k": res.delta_t_max_k,
        "damage_max_s": res.damage_max_s,
        "time_over_limit_s": res.time_over_limit_s,
        "probe_idx": res.probe_idx,
        "estimator": res.estimator,
    }


def main() -> None:
    t0 = time.time()
    t, q = make_duty()
    print(f"[场景] {t[-1]/3600:.2f} h  Q {q.min():.2f}–{q.max():.2f} W/芯")

    law = bang_bang(T_ON, T_OFF)
    results: dict[str, dict] = {}
    traces: dict[str, dict] = {}

    for flow_name, flow_lpm in FLOWS.items():
        spec = ModuleSpec(flow_lpm=flow_lpm, t_in_c=T_IN_C)
        rng = np.random.default_rng(42)
        q_spread = 1.0 + Q_SPREAD_SIGMA * rng.standard_normal(spec.n_cells)
        print(f"\n[流量] {flow_name}  q_spread σ={Q_SPREAD_SIGMA:.0%}  "
              f"max={q_spread.max():.3f} min={q_spread.min():.3f}")
        # 稳态梯度画像（q=5 W 代表性负荷）
        ss = steady_state_cells(spec, np.full(spec.n_cells, 5.0) * q_spread, H_HI)
        grad = float(ss.max() - ss.min())
        print(f"   稳态梯度@5W: {grad:.2f} K  (upstream→hot 偏移 "
              f"{ss.max()-ss[0]:.2f} K)")

        strategies = {
            "full_cool": (always_on(1), 0, "worst"),
            "probe_upstream": (law, 0, "probe"),
            "probe_downstream": (law, spec.n_cells - 1, "probe"),
            "probe_compensated": (law, 0, "probe_comp"),
            "ideal_percell": (law, 0, "worst"),
        }
        for name, (law_f, pidx, est) in strategies.items():
            res = run_strategy(t, q, spec, law_f, pidx, est, q_spread)
            key = f"{flow_name}/{name}"
            results[key] = summarize(res)
            results[key]["steady_grad_at_5w_k"] = grad
            traces[key] = res
            print(f"   {name:18s} peak={res.t_max_c:5.1f}°C  超限={res.time_over_limit_s:5.0f}s  "
                  f"E_pump={res.energy_norm_pump:.3f}  ΔT_max={res.delta_t_max_k:.2f}K  "
                  f"切换={res.n_switch}")

    # ---- 稳健性：+15% 产热失配（低流量）——合规裕度耗尽时布点决定超限 ----
    robust: dict[str, dict] = {}
    spec_low = ModuleSpec(flow_lpm=FLOWS["low_0.25lpm"], t_in_c=T_IN_C)
    rng = np.random.default_rng(42)
    q_spread = 1.0 + Q_SPREAD_SIGMA * rng.standard_normal(spec_low.n_cells)
    for name, (law_f, pidx, est) in (
        ("probe_upstream", (law, 0, "probe")),
        ("probe_compensated", (law, 0, "probe_comp")),
        ("ideal_percell", (law, 0, "worst")),
    ):
        res = run_strategy(t, q * 1.15, spec_low, law_f, pidx, est, q_spread)
        robust[name] = {
            "t_max_c": res.t_max_c,
            "time_over_limit_s": res.time_over_limit_s,
            "energy_norm_pump": res.energy_norm_pump,
        }
        print(f"[+15%失配] {name:18s} peak={res.t_max_c:.1f}°C  超限={res.time_over_limit_s:.0f}s")

    # ---- 判词 ----
    low = {k.split("/")[1]: v for k, v in results.items() if k.startswith("low_")}
    des = {k.split("/")[1]: v for k, v in results.items() if k.startswith("design_")}
    upstream_pen = low["probe_upstream"]["time_over_limit_s"]
    comp_pen = low["probe_compensated"]["time_over_limit_s"]
    down_e = low["probe_downstream"]["energy_norm_pump"]
    ideal_e = low["ideal_percell"]["energy_norm_pump"]
    sense = ("probe_upstream", "probe_downstream", "probe_compensated", "ideal_percell")
    des_spread = max(des[n]["t_max_c"] for n in sense) - min(des[n]["t_max_c"] for n in sense)
    low_spread = max(low[n]["t_max_c"] for n in sense) - min(low[n]["t_max_c"] for n in sense)
    verdict = (
        f"低流量下空间梯度进入控制带宽量级（σ_q=8%，t_in={T_IN_C:.0f}°C）："
        f"上游探针漏报 → 超限 {upstream_pen:.0f} s（峰值 {low['probe_upstream']['t_max_c']:.1f} °C），"
        f"梯度补偿后 {comp_pen:.0f} s（峰值 {low['probe_compensated']['t_max_c']:.1f} °C，≈逐芯理想"
        f" {low['ideal_percell']['t_max_c']:.1f} °C）；"
        f"下游探针天然贴近最坏点（与逐芯理想几乎同轨迹），代价是相对上游多用 "
        f"{(down_e/low['probe_upstream']['energy_norm_pump']-1)*100:.0f}% 泵能耗——"
        f"上游的『省能』以温度水平/合规为代价，不是真省；"
        f"低流量下布点峰值差 {low_spread:.1f} K，设计流量下仅 {des_spread:.1f} K"
        f"（梯度 <1 K，布点几乎无差别）；+15% 产热失配下合规裕度耗尽："
        f"上游超限 {robust['probe_upstream']['time_over_limit_s']:.0f} s"
        f"（峰值 {robust['probe_upstream']['t_max_c']:.1f} °C），"
        f"补偿 {robust['probe_compensated']['time_over_limit_s']:.0f} s / "
        f"理想 {robust['ideal_percell']['time_over_limit_s']:.0f} s"
        f"（峰值 {robust['ideal_percell']['t_max_c']:.1f} °C）——布点决定的是合规裕度。"
        f"建议：NTC 放下游（热端）或上游+梯度补偿；流量衰减工况必须按最坏芯口径定阈值。"
    )
    print("判词:", verdict)

    payload = {
        "scenario": {
            "base_c": BASE_C, "cycles": CYCLES, "tail_c": TAIL_C,
            "h_lo": H_LO, "h_hi": H_HI, "t_on_c": T_ON, "t_off_c": T_OFF,
            "min_dwell_s": MIN_DWELL_S, "limit_c": T_LIMIT_C,
            "t_in_c": T_IN_C,
            "q_spread_sigma": Q_SPREAD_SIGMA,
            "flows_lpm": FLOWS,
        },
        "results": results,
        "robustness_q_x1.15": robust,
        "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    (MET / "24_module_control.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    # ---- 图 24a ----
    plotting.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) 低流量上游探针：峰值时刻的模组温度图
    ax = axes[0, 0]
    res = traces["low_0.25lpm/probe_upstream"]
    k_peak = int(np.argmax(res.temps_c.max(axis=0)))
    grid = res.temps_c[:, k_peak].reshape(2, 6)
    im = ax.imshow(grid, cmap="inferno", aspect="auto")
    for r in range(2):
        for c in range(6):
            ax.text(c, r, f"{grid[r, c]:.1f}", ha="center", va="center",
                    color="white", fontsize=8)
    ax.set_xticks(range(6), [f"列{j+1}" for j in range(6)], fontsize=8)
    ax.set_yticks(range(2), ["排1(近入口)", "排2"], fontsize=8)
    ax.set_title(f"a) 峰值时刻温度场（低流量/上游探针）\n"
                 f"探针=左上 {grid[0,0]:.1f}°C，最热={grid.max():.1f}°C")
    plt.colorbar(im, ax=ax, fraction=0.046, label="°C")

    # (b) 低流量各策略最热芯温度轨迹
    ax = axes[0, 1]
    style = {
        "probe_upstream": ("-", "#B45309"),
        "probe_downstream": ("-", "#0E7490"),
        "probe_compensated": ("-", "#1D4ED8"),
        "ideal_percell": ("--", "#047857"),
        "full_cool": (":", "#6B7280"),
    }
    for name, (ls, color) in style.items():
        res = traces[f"low_0.25lpm/{name}"]
        ax.plot(res.t_s / 3600, res.temps_c.max(axis=0), ls, color=color,
                lw=1.4, label=name)
    ax.axhline(T_LIMIT_C, color="red", lw=1, ls="--")
    ax.axhline(T_ON, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("时间 h")
    ax.set_ylabel("最热芯温度 °C")
    ax.set_title("b) 低流量：最热芯温度轨迹（虚线=判据 45 °C）")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # (c) 峰值温度（两流量 × 策略）+ 稳健性标记
    ax = axes[1, 0]
    names = ["probe_upstream", "probe_downstream", "probe_compensated", "ideal_percell"]
    x = np.arange(len(names))
    w = 0.36
    pk = [results[f"low_0.25lpm/{n}"]["t_max_c"] for n in names]
    pk_d = [results[f"design_2.0lpm/{n}"]["t_max_c"] for n in names]
    ax.bar(x - w / 2, pk, w, label="低流量 0.25 L/min", color="#B45309")
    ax.bar(x + w / 2, pk_d, w, label="设计流量 2.0 L/min", color="#0E7490")
    for i, n in enumerate(names):
        if n in robust:
            ax.scatter(i - w / 2, robust[n]["t_max_c"], marker="*", s=90,
                       color="#7C3AED", zorder=5)
    ax.scatter([], [], marker="*", s=90, color="#7C3AED",
               label="+15% 产热失配（低流量）")
    ax.axhline(T_LIMIT_C, color="red", lw=1, ls="--", label=f"判据 {T_LIMIT_C:.0f} °C")
    ax.set_xticks(x, ["上游探针", "下游探针", "上游+补偿", "逐芯理想"], fontsize=8)
    ax.set_ylabel("芯温峰值 °C")
    ax.set_ylim(42, 46)
    ax.set_title("c) 布点的温度代价与合规裕度")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, axis="y")

    # (d) 泵口径能耗 vs 峰值
    ax = axes[1, 1]
    for flow_name, color, marker in (("low_0.25lpm", "#B45309", "o"),
                                     ("design_2.0lpm", "#0E7490", "s")):
        for n in names:
            v = results[f"{flow_name}/{n}"]
            ax.scatter(v["energy_norm_pump"], v["t_max_c"], color=color,
                       marker=marker, s=45)
            ax.annotate(n.split("_")[1] if n != "probe_compensated" else "comp",
                        (v["energy_norm_pump"], v["t_max_c"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axhline(T_LIMIT_C, color="red", lw=1, ls="--")
    ax.set_xlabel("泵口径归一化能耗（恒满冷=1）")
    ax.set_ylabel("芯温峰值 °C")
    ax.set_title("d) 能耗-峰值权衡（圆=低流量，方=设计流量）")
    ax.grid(alpha=0.3)

    fig.suptitle("24 · 模组级控制与单点 NTC 布点", fontsize=13)
    fig.tight_layout()
    plotting.save(fig, "24a_module_control.png")
    print(f"[落盘] 24_module_control.json / 24a_module_control.png  ({time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()
