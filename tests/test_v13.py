"""v1.3 深化测试：老化-热反馈 / CS2 拟合 / 脉冲辨识 / 控制闭环的物理防线。

防线清单：
- feedback：无反馈时放大系数 A=1；有增长反馈 A>1；闭环比方向正确（倒数坑）
- feedback：p 权重与 D_EOL 无关性（寿命比不依赖绝对标定）
- CS2 拟合：合成幂律/线性内阻数据能捞回真值；窗口隔离深衰减尾段
- 辨识：τ_slow ↔ R_int 往返自洽；弛豫拟合捞回合成 τ；越界返回 nan
- 控制：bang-bang 滞环/最短保持；恒满冷能耗=1；芯温触发峰值被阈值管控
"""
from __future__ import annotations

import numpy as np
import pytest

from bmthermal.control import always_on, bang_bang, simulate_control_loop
from bmthermal.cs2_aging import fit_fade_power, fit_resistance_growth, implied_delta_t_k
from bmthermal.feedback import (
    FADE_EOL,
    FeedbackParams,
    heat_multiplier,
    life_amplification,
    pair_life_ratio,
    resistance_ratio,
)
from bmthermal.network import build_cell
from bmthermal.observer import (
    identify_r_int_k_w,
    fit_relaxation_tau,
    slow_time_constant_s,
)


# --------------------------------------------------------------- feedback ----


def test_heat_multiplier_endpoints():
    p = FeedbackParams(beta=2.5, eta=0.4, fade_exponent=1.0)
    assert heat_multiplier(0.0, p) == pytest.approx(1.0)
    assert heat_multiplier(1.0, p) == pytest.approx(1.0 + p.q_mult_slope)
    assert heat_multiplier(1.0, p) > heat_multiplier(0.5, p) > 1.0


def test_resistance_ratio_linear():
    assert resistance_ratio(0.0, beta=2.5) == pytest.approx(1.0)
    assert resistance_ratio(FADE_EOL, beta=2.5) == pytest.approx(1.5)


def test_no_feedback_gives_unit_amplification():
    """d_cycle 恒定（不随衰减变化）→ 反馈放大 A=1（数值恒等）。"""
    for p in (0.5, 1.0):
        r = life_amplification(lambda u: 1e-6, FeedbackParams(fade_exponent=p))
        assert r["amplification"] == pytest.approx(1.0, rel=1e-9)


def test_feedback_amplifies_life_loss():
    """d_cycle 随 u 线性增长（产热-内阻正反馈）→ A>1 且寿命损失为正。"""
    slope = 0.3
    r = life_amplification(lambda u: 1e-6 * (1.0 + slope * u))
    assert r["amplification"] > 1.0
    assert r["life_loss_pct"] > 0.0


def test_pair_life_ratio_direction():
    """反馈更陡的方案 b 吃亏 → 闭环比 > 开环比（漂移非负）。

    同时守住 pair_life_ratio 曾经的倒数坑：闭环比必须 = I_a/I_b
    （方案 a 若每 EFC 损伤恒小一半，寿命比恒 2，漂移 0）。
    """
    # 恒定比例：闭环比 = 开环比 = 2，漂移 0
    r0 = pair_life_ratio(lambda u: 1e-6, lambda u: 2e-6)
    assert r0["open_loop_ratio"] == pytest.approx(2.0)
    assert r0["closed_loop_ratio"] == pytest.approx(2.0, rel=1e-6)
    assert abs(r0["ratio_drift_pct"]) < 1e-6

    # b 的反馈更陡（s_b > s_a）：b 的闭环寿命缩水更多 → a 相对更优
    s_a, s_b = 0.1, 0.5
    r1 = pair_life_ratio(
        lambda u: 1e-6 * (1.0 + s_a * u),
        lambda u: 1e-6 * (1.0 + s_b * u),
    )
    assert r1["ratio_drift_pct"] > 0.0
    assert r1["closed_loop_ratio"] > r1["open_loop_ratio"]


def test_amplification_independent_of_d_scale():
    """d_cycle 整体缩放不改变 A（D_EOL 无关性的另一面）。"""
    r1 = life_amplification(lambda u: 1e-6 * (1 + 0.3 * u))
    r2 = life_amplification(lambda u: 2e-6 * (1 + 0.3 * u))
    assert r1["amplification"] == pytest.approx(r2["amplification"], rel=1e-9)


# ------------------------------------------------------------- cs2 fitting ----


def test_fit_fade_power_recovers_exponent():
    rng = np.random.default_rng(0)
    n = np.arange(1.0, 800.0)
    a_true, b_true = 0.02, 0.72
    f = a_true * n**b_true
    f_noisy = f * (1.0 + rng.normal(0, 0.01, n.size))
    r = fit_fade_power(n, f_noisy)
    assert r["b"] == pytest.approx(b_true, abs=0.03)
    assert r["a"] == pytest.approx(a_true, rel=0.1)
    assert r["r2"] > 0.99


def test_fit_fade_power_window_excludes_tail():
    """尾段（f>25 %）换成另一个指数，b 不应被污染。"""
    n = np.arange(1.0, 1000.0)
    f = 0.03 * n**0.6
    tail = f > 0.25
    f2 = f.copy()
    f2[tail] = 0.25 * n[tail] ** 1.5   # 畸变尾段
    r = fit_fade_power(n, f2)
    assert r["b"] == pytest.approx(0.6, abs=0.05)


def test_fit_resistance_growth_recovers_beta():
    rng = np.random.default_rng(1)
    f = np.linspace(0.0, 0.35, 200)
    beta_true = 2.2
    r_ratio = 1.0 + beta_true * f + rng.normal(0, 0.005, f.size)
    r = fit_resistance_growth(f, r_ratio)
    assert r["beta"] == pytest.approx(beta_true, abs=0.15)
    assert r["r2"] > 0.9
    assert r["r_ratio_at_eol"] == pytest.approx(1.0 + beta_true * 0.2, abs=0.05)


def test_implied_delta_t_sane():
    assert implied_delta_t_k(1.0) == pytest.approx(0.0, abs=1e-9)
    # 损伤翻倍 ≈ +7~8 K（Ea=45 kJ/mol, 25 °C 基准）
    d2 = implied_delta_t_k(2.0)
    assert 5.0 < d2 < 12.0
    # 单调
    assert implied_delta_t_k(3.0) > d2 > implied_delta_t_k(1.5) > 0
    assert np.isnan(implied_delta_t_k(-1.0))


# ---------------------------------------------------------- identification ----


def test_slow_time_constant_scale():
    cell = build_cell(h=500.0)
    tau = slow_time_constant_s(cell)
    approx = (cell.r_int_k_w + cell.r_conv_k_w) * cell.c_core
    assert tau > 0
    assert tau == pytest.approx(approx, rel=0.25)


def test_identify_r_int_roundtrip():
    """已知 R_int 放大 1.7× → 由 τ 反解必须捞回同一值。"""
    from dataclasses import replace as _replace

    cell_nom = build_cell(h=500.0)
    cell_bad = _replace(cell_nom, r_int_k_w=cell_nom.r_int_k_w * 1.7)
    tau = slow_time_constant_s(cell_bad)
    r_est = identify_r_int_k_w(tau, cell_nom)
    assert r_est == pytest.approx(cell_nom.r_int_k_w * 1.7, rel=1e-6)


def test_identify_r_int_out_of_range_nan():
    cell = build_cell(h=500.0)
    assert np.isnan(identify_r_int_k_w(slow_time_constant_s(cell) * 100.0, cell))
    assert np.isnan(identify_r_int_k_w(-1.0, cell))


def test_fit_relaxation_tau_recovers_tau():
    t = np.arange(0.0, 600.0, 2.0)
    tau_true = 120.0
    y = 25.0 + 3.0 * np.exp(-(t - 100.0) / tau_true)
    r = fit_relaxation_tau(t, y, 100.0, 500.0)
    assert r["tau_s"] == pytest.approx(tau_true, rel=0.05)
    assert r["r2"] > 0.999


# -------------------------------------------------------------- control ----


def test_bang_bang_hysteresis():
    law = bang_bang(43.0, 40.0, use_estimate=False)
    xh = np.array([45.0, 43.0])
    assert law(0, 42.9, xh, 0) == 0
    assert law(0, 43.0, xh, 0) == 1
    assert law(0, 42.0, xh, 1) == 1      # 滞环保持
    assert law(0, 40.0, xh, 1) == 0
    assert law(0, 41.5, xh, 0) == 0      # 滞环保持
    with pytest.raises(ValueError):
        bang_bang(40.0, 43.0)


def test_control_loop_always_on_baseline():
    t = np.arange(0.0, 600.0, 2.0)
    q = np.full_like(t, 3.0)
    cell = build_cell(h=500.0)
    res = simulate_control_loop(
        t, q, 25.0, cell, cell, h_lo=25.0, h_hi=500.0,
        law=always_on(1), noise_k=0.0, initial_mode=1)
    assert res.energy_norm == pytest.approx(1.0, abs=1e-9)
    assert res.n_switch == 0
    assert res.duty_high == pytest.approx(1.0)
    assert np.isfinite(res.damage_s) and res.damage_s > 0
    assert res.t_max_core_c > 25.0


def test_control_loop_min_dwell_blocks_chatter():
    """交替指令律：无最短保持时逐步切换；加保持后切换数骤降。"""
    t = np.arange(0.0, 1200.0, 1.0)
    q = np.full_like(t, 3.0)
    cell = build_cell(h=500.0)

    def flip(t_, z, xh, mode):
        return 1 - mode

    res_free = simulate_control_loop(
        t, q, 25.0, cell, cell, h_lo=25.0, h_hi=500.0, law=flip,
        noise_k=0.0, min_dwell_s=0.0)
    res_dwell = simulate_control_loop(
        t, q, 25.0, cell, cell, h_lo=25.0, h_hi=500.0, law=flip,
        noise_k=0.0, min_dwell_s=60.0)
    assert res_free.n_switch >= 1000            # 抖振
    assert res_dwell.n_switch <= 25             # 1200 s / 60 s
    # 保持约束：任意两次切换间隔 ≥ 60 s
    sw = np.where(np.diff(res_dwell.mode_trace) != 0)[0]
    if sw.size >= 2:
        assert np.min(np.diff(res_dwell.t_s[sw])) >= 60.0 - 1e-9


def test_control_loop_core_trigger_bounds_peak():
    """芯温估计触发在阶跃热负荷下把芯温峰值管在 t_on 附近（超调 < 带宽）。"""
    t = np.arange(0.0, 3600.0, 2.0)
    q = np.full_like(t, 5.0)
    cell = build_cell(h=500.0)
    t_on, t_off = 40.0, 37.0
    res = simulate_control_loop(
        t, q, 25.0, cell, cell, h_lo=25.0, h_hi=500.0,
        law=bang_bang(t_on, t_off, use_estimate=True),
        noise_k=0.02, seed=0, min_dwell_s=30.0)
    assert res.t_max_core_c < t_on + 2.5        # 无失控超调
    assert res.t_max_core_c > t_on - 1.0        # 确实管到了阈值
    assert res.core_err_rmse_k < 1.0
