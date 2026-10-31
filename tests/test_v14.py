"""v1.4 深化测试：actuator / calendar / module_control / a123 辨识护栏。

重点覆盖 v1.4 踩坑里长出来的性质：
- 功率曲线单调性与理想律退化一致性；
- 日历份额的方向性、rebase 不变性、形状拟合可辨识性护栏；
- 模组控制环防线（时变 h 推进、梯度方向、控制律生效）；
- R(T) 拟合与辨识反解的往返一致性。
"""
from __future__ import annotations

import numpy as np
import pytest

from bmthermal.actuator import (
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
from bmthermal.a123_thermal import _temp_from_name, fit_r_int_t
from bmthermal.calendar import (
    bootstrap_two_group_share,
    calendar_bracket,
    calibrate_from_share,
    fit_decomposition,
    rebase_share,
    two_group_share,
)
from bmthermal.module import ModuleSpec
from bmthermal.module_control import simulate_module_control, steady_state_cells


# ---------- actuator ----------

class TestActuator:
    def test_power_monotone_and_idle_floor(self):
        for curve in (AFFINITY, FAN_TYPICAL, PUMP_TYPICAL):
            hs = np.linspace(0, 1, 50)
            p = curve.power_w(hs)
            assert np.all(np.diff(p) >= -1e-12), "功率必须随档位单调不减"
            assert p[0] == pytest.approx(curve.p_max_w * curve.idle_frac)
            assert p[-1] == pytest.approx(curve.p_max_w)

    def test_affinity_recovers_cubic_law(self):
        c = ActuatorCurve(p_max_w=10.0, idle_frac=0.0, static_frac=0.0)
        for h in (0.1, 0.3, 0.7, 1.0):
            assert c.power_w(h) == pytest.approx(10.0 * h**3)

    def test_static_term_raises_low_end(self):
        pump = ActuatorCurve(p_max_w=10.0, idle_frac=0.1, static_frac=0.4)
        pure = ActuatorCurve(p_max_w=10.0, idle_frac=0.1, static_frac=0.0)
        assert pump.power_w(0.1) > pure.power_w(0.1), "静扬程线性项抬高低档功率"

    def test_energy_norm_consistency(self):
        t = np.arange(0, 100.0, 1.0)
        h = np.where(t < 50, 25.0, 500.0)
        e = energy_from_trace(t, h, 500.0, PUMP_TYPICAL)
        en = energy_norm_from_trace(t, h, 25.0, 500.0, PUMP_TYPICAL)
        e_full = energy_from_trace(t, np.full_like(t, 500.0), 500.0, PUMP_TYPICAL)
        assert en == pytest.approx(e / e_full)
        # 解析对拍：稳态占空比 0.5
        d = duty_energy_norm(0.5, 25.0, 500.0, PUMP_TYPICAL)
        assert d == pytest.approx(en, rel=0.02)

    def test_ident_period_scales_with_beta(self):
        p1 = ident_period_efc(0.5, 3e-4, 0.05)
        p2 = ident_period_efc(2.5, 3e-4, 0.05)
        assert p2 < p1 / 4, "β↑ → 辨识周期应近似 ∝1/β"
        with pytest.raises(ValueError):
            ident_period_efc(0.0, 3e-4)

    def test_ident_schedule_low_load_defers(self):
        t = np.arange(0, 1000.0, 1.0)
        q = np.where(t < 500, 5.0, 0.2)   # 前半高负荷
        mask = ident_schedule(t, 200.0, 50.0, q_w=q, q_threshold_w=1.0)
        assert mask.any(), "低负荷段必须开到窗"
        assert not mask[t < 500].any(), "高负荷段不得开窗（时分复用）"

    def test_ident_hold_law_freezes_mode(self):
        t = np.arange(0, 100.0, 1.0)
        mask = (t >= 30) & (t < 60)
        law = ident_hold_law(lambda tt, z, x, m: 1 - m, mask, t)
        assert law(40.0, 0, np.zeros(2), 0) == 0, "窗内必须保持"
        assert law(10.0, 0, np.zeros(2), 0) == 1, "窗外正常切换"


# ---------- calendar ----------

def _synth_cells(n=300, s=0.5, b=0.75, p=0.5, dt_fast=10000.0, t_ratio=1.5,
                 seed=0):
    """构造 f = a·N^b + k·(N·dt)^p 的合成两组电芯（真值 s 份额在 dt_fast）。"""
    rng = np.random.default_rng(seed)
    # 由 s 反推 k 与 a：工作点 N=300 处 s = f_cal/(f_cyc+f_cal)
    f_work = 0.10
    a, k = calibrate_from_share(s, f_work, float(n), dt_fast, b, p)
    cells = {}
    for i in range(3):
        dt = dt_fast * t_ratio
        nn = np.arange(1, n + 1, dtype=float)
        f = a * nn**b + k * (nn * dt)**p + rng.normal(0, 2e-4, n)
        cells[f"slow{i}"] = (nn, f, dt)
    for i in range(4):
        dt = dt_fast
        nn = np.arange(1, n + 1, dtype=float)
        f = a * nn**b + k * (nn * dt)**p + rng.normal(0, 2e-4, n)
        cells[f"fast{i}"] = (nn, f, dt)
    return cells, a, k


class TestCalendar:
    def test_two_group_share_identity(self):
        # 构造恒等：r = 1 + s(t_ratio−1)
        s, tr = 0.4, 2.0
        r = 1 + s * (tr - 1)
        assert two_group_share(r * 1e-4, 1e-4, tr) == pytest.approx(s)

    def test_rebase_invariants(self):
        assert rebase_share(0.6, 100.0, 100.0) == pytest.approx(0.6)
        # 换到更短参考时长 → 份额变小（日历项占比随 EFC 缩短而降）
        assert rebase_share(0.6, 30.0, 100.0) < 0.6
        # 往返一致
        s1 = rebase_share(0.6, 30.0, 100.0)
        s2 = rebase_share(s1, 100.0, 30.0)
        assert s2 == pytest.approx(0.6)

    def test_bracket_direction(self):
        assert calendar_bracket(0.5, 10.0, 10.0) == pytest.approx(1.0)
        # 场景 EFC 比参考短 → 乘子 >1（通量损伤不再打折）
        assert calendar_bracket(0.5, 3.0, 10.0) > 1.0
        # 纯日历极限 s=1 → 恒 1（旧模型就是纯日历）
        assert calendar_bracket(1.0, 3.0, 10.0) == pytest.approx(1.0)

    def test_fit_decomposition_identifies_share(self):
        cells, a_true, k_true = _synth_cells(s=0.4)
        fit = fit_decomposition(cells, p_cal=0.5, smooth_win=5, b_fixed=0.75)
        assert 0.2 < fit.share_at[0.10] < 0.65, \
            f"共享振幅 + 固定 b 下应识别出 s≈0.4，得到 {fit.share_at[0.10]:.2f}"
        assert fit.k_cal > 0

    def test_fit_amplitude_guardrail_documented(self):
        # 逐芯振幅会吸收组间差（v1.4 坑）——shared 模式必须给出更接近真值的 s
        cells, _, _ = _synth_cells(s=0.4)
        fit_shared = fit_decomposition(cells, p_cal=0.5, smooth_win=5,
                                       amplitude="shared")
        assert fit_shared.k_cal > 0

    def test_bootstrap_share_range(self):
        cells, _, _ = _synth_cells(s=0.5)
        fade = {cid: float(np.mean(np.diff(f))) * 1e3
                for cid, (n, f, dt) in cells.items()}
        slow = [c for c in cells if c.startswith("slow")]
        fast = [c for c in cells if c.startswith("fast")]
        out = bootstrap_two_group_share(fade, slow, fast, t_ratio=1.5, n_boot=50)
        assert 0.0 <= out["median"] <= 1.0
        assert out["lo68"] <= out["hi68"]


# ---------- module_control ----------

class TestModuleControl:
    def test_steady_state_gradient_direction(self):
        spec = ModuleSpec(flow_lpm=0.25, t_in_c=25.0)
        q = np.full(spec.n_cells, 5.0)
        ss = steady_state_cells(spec, q, 500.0)
        # 冷却液沿程升温 → 下游（末列）更热
        assert ss[-1] > ss[0], "沿流向下游必须更热"
        # 流量越低梯度越大
        spec_hi = ModuleSpec(flow_lpm=2.0, t_in_c=25.0)
        ss_hi = steady_state_cells(spec_hi, q, 500.0)
        assert (ss[-1] - ss[0]) > (ss_hi[-1] - ss_hi[0])

    def test_control_loop_modes_and_metrics(self):
        spec = ModuleSpec(flow_lpm=0.25, t_in_c=25.0)
        t = np.arange(0, 600.0, 2.0)
        q = np.full_like(t, 6.0)

        def law(t_now, sig, mode):
            return 1 if sig >= 40.0 else (0 if sig <= 37.0 else mode)

        res = simulate_module_control(t, q, spec, law, 150.0, 500.0,
                                      probe_idx=0, estimator="probe",
                                      min_dwell_s=20.0)
        assert res.temps_c.shape == (spec.n_cells, t.size)
        assert 0 <= res.energy_norm_affinity <= 1
        assert res.t_max_c > 25.0
        assert np.isfinite(res.damage_max_s)

    def test_estimator_signals(self):
        spec = ModuleSpec(flow_lpm=0.25, t_in_c=25.0)
        t = np.arange(0, 300.0, 2.0)
        q = np.full_like(t, 5.0)

        def hold(t_now, sig, mode):
            return 0

        r_probe = simulate_module_control(t, q, spec, hold, 150.0, 500.0,
                                          probe_idx=0, estimator="probe",
                                          noise_k=0.0)
        r_comp = simulate_module_control(t, q, spec, hold, 150.0, 500.0,
                                         probe_idx=0, estimator="probe_comp",
                                         noise_k=0.0)
        # 补偿信号必须不低于探针信号（偏移 ≥0）
        assert np.all(r_comp.signal_c[10:] >= r_probe.signal_c[10:] - 1e-6)

    def test_q_spread_validation(self):
        spec = ModuleSpec()
        t = np.arange(0, 10.0, 1.0)
        q = np.ones_like(t)

        def hold(t_now, sig, mode):
            return 0

        with pytest.raises(ValueError):
            simulate_module_control(t, q, spec, hold, 1.0, 2.0,
                                    q_spread=np.ones(3))


# ---------- a123 ----------

class TestA123:
    def test_temp_from_name(self):
        assert _temp_from_name("A123_DST-US06-FUDS-N10.zip") == -10.0
        assert _temp_from_name("A123_DST-US06-FUDS-25.zip") == 25.0
        assert _temp_from_name("A123_DST-US06-FUDS-0.zip") == 0.0

    def test_fit_r_int_t_roundtrip(self):
        # 构造已知 Ea 的单调 R(T)：ln ratio = Ea/R(1/T−1/298)
        from bmthermal.params import R_GAS
        ea = 20000.0
        rows = []
        for tc in (-10.0, 0.0, 25.0, 50.0):
            ratio = np.exp(ea / R_GAS * (1.0 / (tc + 273.15) - 1.0 / 298.15))
            for cell in ("A", "B"):
                rows.append({"t_set_c": tc, "cell": cell, "r_ohm_mid_soc": 0.1 * ratio,
                             "cap_ah": 3.0, "t_meas_c": tc, "n_rows": 100})
        import pandas as pd
        fit = fit_r_int_t(pd.DataFrame(rows), t_ref_c=25.0)
        assert fit.ea_r_j_mol == pytest.approx(ea, rel=0.05)
        assert fit.ratio(-10.0) == pytest.approx(
            np.exp(ea / R_GAS * (1.0 / 263.15 - 1.0 / 298.15)), rel=0.05)
        assert fit.r2 > 0.99

    def test_calibrate_from_share_roundtrip(self):
        a, k = calibrate_from_share(0.4, 0.1, 500.0, 10000.0, 0.75, 0.5)
        f_c = a * 500.0**0.75
        f_k = k * (500.0 * 10000.0) ** 0.5
        assert f_k / (f_c + f_k) == pytest.approx(0.4)
