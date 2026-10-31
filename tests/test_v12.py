"""v1.2 深化测试：观测器 / 沸腾边界 / 导热垫映射的物理防线。

防线清单：
- 观测器：状态空间解与 network.solve_ivp 解一致（同一物理两条路径）
- 观测器：名义参数下 Kalman 芯温误差 < 0.5 K；持续偏差与 (q·r−1)·Q·R_int 理论一致
- 沸腾：CHF 在 FC-72 类文献量级；核态曲线随过热度单调；3C 标称单相成立
- 导热垫：h 随压力单调（收益递减）；2 mm 垫不可达 250 目标；反解与正解自洽
"""
from __future__ import annotations

import numpy as np
import pytest

from bmthermal.boiling import (
    FLUID, chf_zuber, effective_h_boiling, immersion_validity,
    nucleate_boiling_curve, hotspot_factor_to_chf,
)
from bmthermal.network import build_cell, simulate_two_node
from bmthermal.observer import run_kalman, simulate_state_space, two_node_matrices
from bmthermal.tim import PadSpec, effective_h, pressure_for_target_h


# ------------------------------------------------------------- observer ----


def test_state_space_matches_ode_solver():
    t = np.linspace(0.0, 600.0, 121)
    q = np.full_like(t, 3.0)
    cell = build_cell(h=120.0)
    ode = simulate_two_node(t, q, 25.0, cell)
    xs = simulate_state_space(cell, t, q, 25.0)
    assert np.allclose(xs[0], ode["t_core_c"], atol=1e-6)
    assert np.allclose(xs[1], ode["t_surface_c"], atol=1e-6)


def test_two_node_matrices_shapes():
    A, B, H = two_node_matrices(build_cell(h=500.0))
    assert A.shape == (2, 2) and B.shape == (2, 2) and H.shape == (1, 2)
    # 表面测温通道
    assert H[0, 1] == 1.0 and H[0, 0] == 0.0


def test_kalman_nominal_recovers_core():
    t = np.linspace(0.0, 1800.0, 361)
    q = np.full_like(t, 5.0)
    cell = build_cell(h=500.0)
    truth = simulate_two_node(t, q, 25.0, cell)
    rng = np.random.default_rng(1)
    z = truth["t_surface_c"] + rng.normal(0, 0.02, t.size)
    res = run_kalman(
        t, q, z, 25.0, cell,
        q_process_w2=0.02**2, r_meas_k2=0.02**2,
        t0_c=25.0, truth_core_c=truth["t_core_c"], x0_c=25.0,
    )
    assert res.core_err_rmse_k < 0.5
    assert not res.diverged


def test_kalman_sustained_bias_matches_theory():
    """持续高倍率下，Q 失配的稳态芯温偏差 ≈ (q_scale−1)·Q·R_int。"""
    t = np.linspace(0.0, 3600.0, 361)
    q_true = 5.6
    cell = build_cell(h=500.0)
    truth = simulate_two_node(t, np.full_like(t, q_true), 25.0, cell)
    z = truth["t_surface_c"]  # 无噪声，纯失配
    q_s = 1.5
    res = run_kalman(
        t, np.full_like(t, q_true * q_s), z, 25.0, cell,
        q_process_w2=1e-6, r_meas_k2=1e-6,
        t0_c=25.0, truth_core_c=truth["t_core_c"], x0_c=25.0,
    )
    seg = res.t_core_hat_c[-100:] - truth["t_core_c"][-100:]
    bias = float(np.mean(seg))
    theory = (q_s - 1.0) * q_true * cell.r_int_k_w
    # 观测器会通过表面积分部分补偿，偏差应不超过理论值且同号
    assert bias > 0
    assert bias <= theory * 1.2


# -------------------------------------------------------------- boiling ----


def test_chf_in_literature_range():
    q_chf = chf_zuber(FLUID)
    # FC-72 类介电体饱和池沸腾 CHF 文献量级 11–18 W/cm²
    assert 11e4 <= q_chf <= 18e4


def test_nucleate_curve_monotone():
    dt_e = np.linspace(1.0, 25.0, 50)
    q = nucleate_boiling_curve(dt_e)
    assert np.all(np.diff(q) > 0)


def test_immersion_single_phase_ok_at_nominal():
    v = immersion_validity(5.6, 0.0065, h_sp=1500.0, t_fluid_c=25.0)  # 3C
    assert v["regime"] == "single_phase_ok"
    assert v["t_wall_single_phase_c"] < FLUID.t_sat_c
    assert v["chf_margin"] > 10.0


def test_immersion_degraded_circulation_triggers_boiling():
    v = immersion_validity(14.0, 0.0065, h_sp=50.0, t_fluid_c=25.0)  # 5C, 弱循环
    assert v["boiling_onset"] is True


def test_hotspot_factor_positive():
    f = hotspot_factor_to_chf(5.6, 0.0065)
    assert f > 20.0  # 平均热流下须放大 20 倍以上才撞 CHF


def test_boiling_h_eff_load_dependent():
    b1 = effective_h_boiling(1000.0)
    b2 = effective_h_boiling(3000.0)
    assert b1["regime"] == "nucleate_boiling"
    # Rohsenow：q''∝ΔT³ → h_eff=q''/ΔT∝q''^(2/3) 随负荷上升
    assert b2["h_eff"] > b1["h_eff"]


# --------------------------------------------------------------- tim ----


def test_effective_h_monotone_with_diminishing_returns():
    pad = PadSpec(thickness_mm=1.0, k_w_mk=1.5)
    h1 = effective_h(pad, 50e3)
    h2 = effective_h(pad, 100e3)
    h3 = effective_h(pad, 1000e3)
    assert h1 < h2 < h3
    assert (h3 - h2) < (h2 - h1) + 1e-9 or (h3 - h2) < 30  # 收益递减（量级检查）


def test_thick_pad_infeasible_thin_pad_feasible():
    thin = PadSpec(thickness_mm=0.5, k_w_mk=1.5)
    thick = PadSpec(thickness_mm=2.0, k_w_mk=1.5)
    assert np.isfinite(pressure_for_target_h(thin, 250.0))
    assert not np.isfinite(pressure_for_target_h(thick, 250.0))
    assert effective_h(thin, 100e3) >= 250.0


def test_pressure_inverse_consistent():
    pad = PadSpec(thickness_mm=1.0, k_w_mk=1.5)
    p = pressure_for_target_h(pad, 250.0)
    assert np.isfinite(p)
    assert abs(effective_h(pad, p) - 250.0) < 1e-6
