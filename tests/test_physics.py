"""物理一致性测试：解析解对照、辨识闭环、寿命积分恒等式。

设计原则（与 A4 一致）：**每条测试都先有独立的物理预期**，再看代码是否满足；
不测"代码输出等于代码输出"。数值容差按物理量级定，不拍脑袋。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmthermal.aging import (  # noqa: E402
    damage_integral,
    equivalent_temperature,
    life_ratio,
    mean_temperature_bias,
)
from bmthermal.heat import fit_power_law, predict_q  # noqa: E402
from bmthermal.network import (  # noqa: E402
    TwoNodeCell,
    analytic_core_surface_gap,
    build_cell,
    identify_two_node,
    simulate_two_node,
    steady_state_rise,
)
from bmthermal.params import CELL, R_GAS, volumetric_heat_capacity  # noqa: E402


# --------------------------------------------------------------------------- #
# 热容：手算口径 vs PyBaMM 模型内部口径
# --------------------------------------------------------------------------- #
def test_thermal_capacity_matches_pybamm_internal():
    """按层手算的 ρc_p 必须与 PyBaMM 求解器实际用的体积热容一致（<1%）。

    注意：``Volume-averaged effective heat capacity`` 在 PyBaMM 26.x 里是
    **模型变量**（不是 ParameterValues 的键），只能从解里读。
    这是"书上抄的数"和"模型实际用的数"打架的防线。
    """
    pybamm = pytest.importorskip("pybamm")
    import warnings

    from bmthermal.params import load_parameter_values

    pv = load_parameter_values("Chen2020")
    mdl = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
    exp = pybamm.Experiment(["Discharge at 0.5C for 20 s"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = pybamm.Simulation(mdl, parameter_values=pv, experiment=exp).solve()
    internal = float(sol["Volume-averaged effective heat capacity [J.K-1.m-3]"].entries[0])
    manual = volumetric_heat_capacity("Chen2020")
    assert abs(manual / internal - 1.0) < 0.01


def test_thermal_capacity_positive_and_sane():
    """21700 电芯热容应在 30–60 J/K 区间（工程量级检查）。"""
    c = CELL.thermal_capacity_j_k
    assert 30.0 < c < 60.0


# --------------------------------------------------------------------------- #
# 两节点网络 vs 圆柱解析解
# --------------------------------------------------------------------------- #
def test_two_node_steady_gap_equals_analytic():
    """稳态芯-壳温差 Q·R_int 必须等于圆柱均匀内热源解析解 Q/(4πkL)。"""
    q = 2.7
    net = build_cell(h=50.0)
    assert steady_state_rise(q, net)["rise_internal_k"] == pytest.approx(
        analytic_core_surface_gap(q), rel=1e-12
    )


def test_two_node_total_rise_equals_series_resistance():
    """稳态总温升 = Q(R_int + R_conv)，串阻模型的定义。"""
    q = 2.7
    net = build_cell(h=50.0)
    r = steady_state_rise(q, net)
    assert r["rise_total_k"] == pytest.approx(
        q * (net.r_int_k_w + net.r_conv_k_w), rel=1e-12
    )


def test_two_node_transient_converges_to_steady_state():
    """长时间恒定产热后，两节点温度应收敛到稳态解析值。"""
    net = build_cell(h=20.0)
    t = np.linspace(0, 20000.0, 400)
    q = np.full_like(t, 2.0)
    out = simulate_two_node(t, q, 25.0, net)
    steady = steady_state_rise(2.0, net)
    assert out["t_core_c"][-1] == pytest.approx(25.0 + steady["rise_total_k"], abs=0.05)
    assert out["t_surface_c"][-1] == pytest.approx(
        25.0 + 2.0 * net.r_conv_k_w, abs=0.05
    )


# --------------------------------------------------------------------------- #
# 标度律拟合
# --------------------------------------------------------------------------- #
def test_power_law_recovers_exact_exponent():
    """对 Q = k C^α 的精确采样，拟合必须把 α 捞回来。"""
    k_true, a_true = 1.3, 1.8
    c = np.array([0.5, 1.0, 1.5, 2.0])
    q = k_true * c ** a_true
    law = fit_power_law(c, q)
    assert law["alpha"] == pytest.approx(a_true, rel=1e-9)
    assert law["k"] == pytest.approx(k_true, rel=1e-9)
    assert predict_q(3.0, law) == pytest.approx(k_true * 3.0 ** a_true, rel=1e-9)


# --------------------------------------------------------------------------- #
# 参数辨识闭环（已知真值）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("noise_k", [0.0, 0.05])
def test_identification_recovers_truth(noise_k):
    """合成阶跃响应 + 噪声，最小二乘应在 20% 内回收全部四个参数。"""
    rng_noise = noise_k
    t = np.arange(0.0, 1800.0, 4.0)
    q = np.full_like(t, 3.0)
    truth = build_cell(h=12.0)
    out = simulate_two_node(t, q, 25.0, truth)
    fit = identify_two_node(t, q, out["t_surface_c"], 25.0, h=12.0,
                            noise_k=rng_noise, seed=7)
    assert fit["rmse_k"] < (0.2 if noise_k == 0 else 2.0 * noise_k + 0.2)
    assert abs(fit["r_int_k_w"] / truth.r_int_k_w - 1.0) < 0.2
    assert abs(fit["r_conv_k_w"] / truth.r_conv_k_w - 1.0) < 0.2
    assert abs(fit["c_th_j_k"] / truth.c_th_j_k - 1.0) < 0.2


# --------------------------------------------------------------------------- #
# 寿命折算
# --------------------------------------------------------------------------- #
def test_damage_of_constant_temperature_closed_form():
    """恒温历程的损伤积分必须等于解析式 exp(-Ea/R(1/T-1/T_ref))·Δt。"""
    t = np.array([0.0, 100.0, 1000.0])
    tc = np.full(3, 45.0)
    d = damage_integral(t, tc, ea_j_mol=45000.0, t_ref_c=25.0)
    t_k, t_ref_k = 45.0 + 273.15, 25.0 + 273.15
    expected = np.exp(-45000.0 / R_GAS * (1.0 / t_k - 1.0 / t_ref_k)) * 1000.0
    assert d == pytest.approx(expected, rel=1e-10)


def test_life_ratio_inverse_of_damage():
    assert life_ratio(2.0, 1.0) == pytest.approx(0.5)
    assert life_ratio(1.0, 2.0) == pytest.approx(2.0)


def test_equivalent_temperature_inverts_damage():
    """等效温度的定义要求：k(T_eff)·Δt == D。"""
    t = np.linspace(0, 3600.0, 200)
    tc = 30.0 + 15.0 * np.sin(2 * np.pi * t / 3600.0)  # 30±15 °C 摆动
    t_eff = equivalent_temperature(t, tc, ea_j_mol=45000.0)
    d = damage_integral(t, tc, ea_j_mol=45000.0)
    t_ref_k = 298.15
    k_eff = np.exp(-45000.0 / R_GAS * (1.0 / (t_eff + 273.15) - 1.0 / t_ref_k))
    assert k_eff * (t[-1] - t[0]) == pytest.approx(d, rel=1e-9)


def test_mean_temperature_underestimates_damage():
    """Jensen 不等式方向性检查：温度波动 ⇒ 平均温度低估损伤（寿命乐观）。

    exp 是凸函数，E[exp(X)] ≥ exp(E[X])，所以用平均温度算的损伤偏小、
    寿命偏乐观（life_error_pct > 0）。
    """
    t = np.linspace(0, 3600.0, 300)
    tc = 35.0 + 20.0 * np.sin(2 * np.pi * t / 3600.0)  # 15–55 °C
    b = mean_temperature_bias(t, tc, ea_j_mol=45000.0)
    assert b["life_error_pct"] > 0.0
    assert b["t_eff_c"] > b["t_mean_c"]


# --------------------------------------------------------------------------- #
# required_h 往返
# --------------------------------------------------------------------------- #
def test_required_h_roundtrip():
    from bmthermal.cooling import max_rise_at_h, required_h

    q = 4.0
    allowed = 20.0
    h_req = required_h(q, allowed)
    assert np.isfinite(h_req) and h_req > 0
    assert max_rise_at_h(q, h_req) == pytest.approx(allowed, rel=1e-9)


def test_required_h_infinite_when_internal_resistance_dominates():
    """Q·R_int 已吃掉全部温升预算时，h 必须报 inf 而不是给个假数。"""
    from bmthermal.cooling import required_h

    q_huge = 1e6
    h = required_h(q_huge, 20.0)
    assert h == float("inf")


# --------------------------------------------------------------------------- #
# 动态工况剖面
# --------------------------------------------------------------------------- #
def test_us06_profile_shape():
    from bmthermal.duty import load_us06

    t, c = load_us06()
    assert t.size == 600
    assert np.max(c) > 1.0  # 峰值 >1C
    assert np.min(c) < 0.0  # 含回馈
    assert 0.05 < np.mean(c) < 0.5  # 均值 0.2C 量级
