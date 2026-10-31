"""03 · 内部温度梯度与热参数辨识（博远基础版阶段 3 + 定安 D3）。

三件事：
1. **穿厚温差**：x-lumped 一维热传导给出电芯厚度方向 ΔT，量化"只看表面温度"的风险。
2. **芯-壳温差**：两节点模型 vs 圆柱解析解（同一物理、独立实现），互相验证。
3. **参数辨识闭环**：在已知真值的合成阶跃响应上加噪声，检验最小二乘能否把
   R_int / R_conv / C_core / C_shell 捞回来（方法与 A4 的 ECM 辨识同框架）。

产出：
  results/metrics/03_gradient_ident.json
  results/figures/03a_gradient.png        温度梯度
  results/figures/03b_identification.png  辨识拟合与参数回收
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from bmthermal import plotting  # noqa: E402
from bmthermal.heat import run_dfn_heat  # noqa: E402
from bmthermal.network import (  # noqa: E402
    analytic_core_surface_gap,
    build_cell,
    identify_two_node,
    simulate_two_node,
)
from bmthermal.params import CELL  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
MET = OUT / "metrics"
MET.mkdir(parents=True, exist_ok=True)


def thickness_gradient(c_rate: float = 2.0, h: float = 10.0) -> dict:
    """x-full：叠片厚度方向温度分布（分辨到 µm 级的层间温差）。

    口径声明：PyBaMM 的 x 是**叠片方向**。21700 的卷芯径向（cm 级）
    梯度不在这个模型里——那部分由两节点/解析解给出，两者要分开报。
    """
    import warnings

    import pybamm

    from bmthermal.params import apply_x_full_extras, load_parameter_values

    pv = load_parameter_values("Chen2020", t_amb_c=25.0, soc0=1.0, h=h)
    apply_x_full_extras(pv, h=h)
    mdl = pybamm.lithium_ion.DFN(options={"thermal": "x-full"})
    exp = pybamm.Experiment([f"Discharge at {c_rate:g}C until 3.0 V"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = pybamm.Simulation(mdl, parameter_values=pv, experiment=exp).solve()
    t = np.asarray(sol["Time [s]"].entries, dtype=float)
    gaps, profiles = [], []
    idxs = np.unique(np.linspace(0, t.size - 1, 30).astype(int))
    for k in idxs:
        y = np.asarray(sol["Cell temperature [C]"](float(t[k])), dtype=float).ravel()
        gaps.append(float(y.max() - y.min()))
        profiles.append(y)
    vol = np.asarray(sol["Volume-averaged cell temperature [C]"].entries, dtype=float)
    return {
        "c_rate": c_rate, "h": h, "option": "x-full",
        "t_s": t.tolist(),
        "dT_stack_max_k": float(np.max(gaps)),
        "note": "x=叠片厚度方向（~100 um 电极叠层），非卷芯径向",
        "dT_stack_series_k": gaps,
        "sample_times_s": t[idxs].tolist(),
        "profiles": [p.tolist() for p in profiles],
        "t_volume_max_c": float(np.max(vol)),
    }


def main() -> None:
    t0 = time.time()
    payload: dict = {}

    # ---- 1. 穿厚（叠片）温差 ----
    print("[1/3] 叠片方向温度梯度（x-full）...")
    grad = thickness_gradient(2.0, 10.0)
    payload["stack_gradient"] = grad
    print(f"   ΔT_stack max = {grad['dT_stack_max_k']:.6f} K "
          f"(T_vol_max={grad['t_volume_max_c']:.2f} C)  —— 叠片方向可忽略")

    # ---- 2. 芯-壳温差：两节点 vs 解析解 ----
    print("[2/3] 芯-壳温差：两节点 vs 圆柱解析解 ...")
    q_w = 3.0  # 代表产热（约 2–3C 量级）
    comp = []
    for h in (8.0, 50.0, 500.0):
        net = build_cell(h=h)
        steady_gap = q_w * net.r_int_k_w
        analytic_gap = analytic_core_surface_gap(q_w)
        rise = q_w * (net.r_int_k_w + net.r_conv_k_w)
        comp.append({
            "h": h, "q_w": q_w,
            "two_node_steady_gap_k": steady_gap,
            "analytic_gap_k": analytic_gap,
            "abs_diff_k": abs(steady_gap - analytic_gap),
            "steady_rise_total_k": rise,
        })
        print(f"   h={h:6.1f}  两节点 {steady_gap:.4f} K  解析 {analytic_gap:.4f} K  "
              f"差 {abs(steady_gap - analytic_gap):.2e} K")
    payload["core_surface_check"] = comp

    # 瞬态：两节点的芯壳分离过程（用于图）
    net = build_cell(h=10.0)
    r = run_dfn_heat(2.0, isothermal=True)
    sim = simulate_two_node(r.time_s, r.q_total_w, 25.0, net)
    payload["transient_two_node_h10"] = {
        "t_core_max_c": float(sim["t_core_c"].max()),
        "t_surface_max_c": float(sim["t_surface_c"].max()),
        "gap_max_k": float((sim["t_core_c"] - sim["t_surface_c"]).max()),
    }

    # ---- 3. 辨识闭环 ----
    print("[3/3] 参数辨识闭环（真值 + 噪声）...")
    rng = np.random.default_rng(42)
    t_arr = np.arange(0.0, 1200.0, 5.0)
    q_arr = np.full_like(t_arr, 2.5)
    truth = build_cell(h=10.0)
    out = simulate_two_node(t_arr, q_arr, 25.0, truth)
    ident_runs = []
    for noise_k in (0.0, 0.1, 0.5):
        fit = identify_two_node(t_arr, q_arr, out["t_surface_c"], 25.0, h=10.0, noise_k=noise_k)
        ident_runs.append({
            "noise_k": noise_k,
            "truth": {"r_int": truth.r_int_k_w, "r_conv": truth.r_conv_k_w,
                      "c_core": truth.c_core, "c_shell": truth.c_shell},
            "fit": fit,
            "err_pct": {
                "r_int": (fit["r_int_k_w"] / truth.r_int_k_w - 1) * 100,
                "r_conv": (fit["r_conv_k_w"] / truth.r_conv_k_w - 1) * 100,
                "c_core": (fit["c_core_j_k"] / truth.c_core - 1) * 100,
                "c_shell": (fit["c_shell_j_k"] / truth.c_shell - 1) * 100,
            },
        })
        print(f"   noise={noise_k:g} K  RMSE={fit['rmse_k']:.4f} K  "
              f"R_int 误差 {ident_runs[-1]['err_pct']['r_int']:+.1f}%  "
              f"h_eq={fit['h_equivalent']:.1f}")
    payload["identification"] = ident_runs

    payload["elapsed_s"] = round(time.time() - t0, 1)
    (MET / "03_gradient_ident.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # ---------------- 图 ----------------
    plotting.setup_style()

    # 3a: 两种内部梯度的对比（本项目的核心澄清）
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for tt, prof in list(zip(grad["sample_times_s"], grad["profiles"]))[::6]:
        ax[0].plot(np.linspace(0, 1, len(prof)), prof, label=f"{tt:.0f} s", alpha=0.8)
    ax[0].set_xlabel("叠片方向相对位置"); ax[0].set_ylabel("温度 (°C)")
    ax[0].set_title(f"叠片方向温度分布（ΔT={grad['dT_stack_max_k']*1e3:.2f} mK，可忽略）")
    ax[0].legend(fontsize=7)
    # 卷芯径向：芯-壳温差随产热
    q_grid = np.linspace(0.2, 15.0, 50)
    from bmthermal.network import build_cell as _bc
    net10 = _bc(h=10.0)
    gap_q = [q * net10.r_int_k_w for q in q_grid]
    ax[1].plot(q_grid, gap_q, color="#B45309", label="芯-壳温差（卷芯径向）")
    for qq, lab in [(2.72, "2C"), (5.6, "3C*"), (14.0, "5C*")]:
        ax[1].plot(qq, qq * net10.r_int_k_w, "o", color="#111827")
        ax[1].annotate(f"{lab}\n{qq*net10.r_int_k_w:.1f} K", (qq, qq * net10.r_int_k_w),
                       fontsize=7, ha="center", va="bottom")
    ax[1].set_xlabel("产热功率 Q (W)"); ax[1].set_ylabel("温差 (K)")
    ax[1].set_title("卷芯径向芯-壳温差（R_int = 1/(4πkL)）")
    ax[1].legend(fontsize=8)
    plotting.save(fig, "03a_gradient.png")

    # 3b: 辨识
    fit0 = identify_two_node(t_arr, q_arr, out["t_surface_c"], 25.0, h=10.0, noise_k=0.1)
    net_fit = build_cell(h=fit0["h_equivalent"])
    from bmthermal.network import TwoNodeCell
    net_refit = TwoNodeCell(
        c_th_j_k=fit0["c_th_j_k"], frac_core=fit0["c_core_j_k"] / fit0["c_th_j_k"],
        r_int_k_w=fit0["r_int_k_w"], area_m2=CELL.surface_area_m2,
        h_w_m2k=fit0["h_equivalent"],
    )
    out_fit = simulate_two_node(t_arr, q_arr, 25.0, net_refit)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(t_arr, out["t_surface_c"], label="真值（表面）", color="#111827")
    ax[0].plot(t_arr, out["t_core_c"], label="真值（芯）", color="#B45309", ls="--")
    ax[0].plot(t_arr, out_fit["t_surface_c"], label="辨识拟合（表面）", color="#1D4ED8", ls=":")
    ax[0].set_xlabel("时间 (s)"); ax[0].set_ylabel("温度 (°C)")
    ax[0].set_title("辨识拟合（噪声 0.1 K）"); ax[0].legend(fontsize=8)
    names = ["R_int", "R_conv", "C_core", "C_shell"]
    err0 = ident_runs[1]["err_pct"]
    vals = [err0["r_int"], err0["r_conv"], err0["c_core"], err0["c_shell"]]
    ax[1].bar(names, vals, color="#1D4ED8")
    ax[1].axhline(0, color="#111827", lw=0.8)
    ax[1].set_ylabel("相对真值偏差 (%)")
    ax[1].set_title("参数回收精度（0.1 K 噪声）")
    plotting.save(fig, "03b_identification.png")

    print(f"完成，用时 {payload['elapsed_s']} s")


if __name__ == "__main__":
    main()
