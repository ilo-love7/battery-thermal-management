"""深化轮（v1.1）物理一致性测试。

防线清单：
- PCM 焓法：熔化分数有界、有效热容峰在 Tm、等效热容法在熔化段高估峰值
- 标定：ZOH 精确离散与解析解一致；(C,β,G)→(αC,αβ,αG) 轨迹不变（结构不可辨识）
- 标定：固定 C 后 β/G 与 τ 可复现；ambient_from_rest 有界
"""
from __future__ import annotations

import numpy as np
import pytest

from bmthermal.calib import (
    _prepare, _simulate_lti, ambient_from_rest, fit_lumped,
)
from bmthermal.pcm import (
    PCMProps, c_eff_enthalpy, divergence, melt_fraction,
    simulate_enthalpy, simulate_equiv_capacity,
)


# ---------------------------------------------------------------- PCM ----


def test_melt_fraction_bounded_and_monotone():
    p = PCMProps(t_melt_c=45.0, width_k=3.0)
    T = np.linspace(20.0, 70.0, 201)
    f = melt_fraction(T, p)
    assert np.all(f >= 0.0) and np.all(f <= 1.0)
    assert np.all(np.diff(f) >= -1e-12)          # 单调不减
    assert f[0] < 0.01 and f[-1] > 0.99          # 两端饱和


def test_c_eff_peak_at_t_melt():
    p = PCMProps(t_melt_c=45.0, width_k=3.0, latent_j=360.0)
    T = np.linspace(30.0, 60.0, 301)
    c = c_eff_enthalpy(T, p)
    assert np.argmax(c) == pytest.approx(np.argmin(np.abs(T - 45.0)), abs=2)
    # 峰值 = C_base + L·max(df/dT)；tanh 形状下 max(df/dT) ≈ 0.375/width
    assert c.max() > p.c_base_j_k + 0.5 * p.latent_j / p.width_k * 0.8


def test_enthalpy_pins_temperature_near_t_melt():
    """熔化进行中温度应被钉在 Tm 附近（而不是线性爬升）。"""
    p = PCMProps(t_melt_c=45.0, width_k=2.0, latent_j=240.0, g_w_k=0.05)
    out = simulate_enthalpy(q_w=5.6, p=p, duration_s=3600.0)
    # 熔化中段（0.2 < f < 0.8）的温度跨度应远小于无 PCM 的爬升
    mid = (out["melt_fraction"] > 0.2) & (out["melt_fraction"] < 0.8)
    assert mid.sum() > 10
    span = float(out["t_c"][mid].max() - out["t_c"][mid].min())
    assert span < 3.0, f"钉扎失效：熔化段温跨 {span:.1f} K"


def test_equiv_capacity_underestimates_post_melt_peak():
    """等效热容法在熔完后仍带额外热容 → **低估**峰值（方向性检验）。

    （数据否掉了"等效热容法高估"的直觉：它把潜热永久摊进热容，
    熔完后还在保护；焓法熔完保护消失、更悲观也更真实。）
    """
    p = PCMProps(t_melt_c=45.0, width_k=3.0, latent_j=240.0)
    d = divergence(q_w=5.6, p=p, duration_s=3600.0)
    assert d["melt_final"] > 0.99        # 1 小时内熔完
    assert d["peak_diff_k"] < 0.0        # 等效 − 焓法 < 0


# -------------------------------------------------------------- 标定 ----


def test_zoh_matches_analytic_decay():
    """一阶系统 ZOH 精确离散 = 解析解 exp(-dt/τ)。"""
    tau = 300.0
    A = np.array([[-1.0 / tau]])
    B = np.array([[1.0 / tau]])
    dt = np.full(50, 10.0)
    u = np.zeros((1, 51))                  # 零输入 → 纯衰减
    y = _simulate_lti(A, B, np.array([1.0]), np.array([1.0]), u, dt)
    t = np.concatenate(([0.0], np.cumsum(dt)))
    assert y == pytest.approx(np.exp(-t / tau), rel=1e-10)


def test_scale_invariance_structural_unidentifiability():
    """(C, β, G) → (αC, αβ, αG) 输出轨迹完全不变。"""
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 3600.0, 30.0)
    I = rng.normal(-0.5, 0.4, t.size)
    T = np.full_like(t, 25.0)
    seg = [(t, T, I)]

    def traj(c, beta, g):
        A = np.array([[-g / c]])
        B = np.array([[beta / c, g / c]])
        tt, TT, II, dt = _prepare(t, T, I)
        uu = np.stack([II ** 2, np.full_like(II, 25.0)])
        return _simulate_lti(A, B, np.array([1.0]),
                             np.array([25.0]), uu, dt)

    y1 = traj(60.0, 0.03, 0.015)
    y2 = traj(180.0, 0.09, 0.045)          # α = 3
    assert np.allclose(y1, y2, rtol=1e-10)


def test_fit_lumped_fixed_c_recovers_tau():
    """合成数据（已知 τ）上，固定 C 的拟合应恢复 G=C/τ。"""
    c_true, beta_true, g_true = 60.0, 0.03, 0.015
    tau = c_true / g_true
    rng = np.random.default_rng(11)
    t = np.arange(0.0, 12000.0, 30.0)
    I = rng.normal(-0.5, 0.5, t.size)
    A = np.array([[-g_true / c_true]])
    B = np.array([[beta_true / c_true, g_true / c_true]])
    tt, TT, II, dt = _prepare(t, np.zeros_like(t), I)
    uu = np.stack([II ** 2, np.full_like(II, 25.0)])
    y = _simulate_lti(A, B, np.array([1.0]), np.array([25.0]), uu, dt)
    T_meas = y + rng.normal(0, 0.02, y.size)     # 加 0.02 K 噪声

    f = fit_lumped([(tt, T_meas, II)], t_amb_c=25.0, c_th_j_k=c_true)
    assert f.rmse_k < 0.05
    assert f.g_w_k == pytest.approx(g_true, rel=0.15)
    assert f.beta_w_a2 == pytest.approx(beta_true, rel=0.15)
    assert (c_true / f.g_w_k) == pytest.approx(tau, rel=0.15)


def test_ambient_from_rest_bounded():
    t = np.arange(0.0, 3600.0, 10.0)
    T = 25.0 + 0.0002 * t                    # 缓慢漂移
    I = np.where((t % 600) < 60, 0.0, -1.0)  # 周期性静置
    ta = ambient_from_rest(t, T, I)
    assert ta.shape == t.shape
    assert np.all(ta >= T.min() - 1.0) and np.all(ta <= T.max() + 1.0)
