"""v1.7 测试：绝对寿命锚点 / 外推审计 / 表观 Ea / 口径换算。"""
from __future__ import annotations

import numpy as np
import pytest

from bmthermal.abs_life import (
    ChemAnchor,
    audit_extrapolation,
    n_eol_from_curve,
    n_eol_linear,
    n_eol_power,
)
from bmthermal.calendar import calendar_bracket
from bmthermal.snl_aging import first_crossing, fit_fade_power


class TestAbsLifeScale:
    def test_linear_and_power(self):
        assert abs(n_eol_linear(2.5e-5) - 8000.0) < 1e-6
        # f=a·N^b, a=1e-4, b=0.5 → N(0.2)=(0.2/1e-4)^2=4e6
        assert abs(n_eol_power(1e-4, 0.5) - 4e6) < 1.0

    def test_rejects_bad_inputs(self):
        assert np.isnan(n_eol_linear(0.0))
        assert np.isnan(n_eol_linear(float("nan")))
        assert np.isnan(n_eol_power(1e-4, 0.0))

    def test_anchor_scale_consistency(self):
        """参考工况 N_EOL 与 A_chem·d_ref 自洽。"""
        anc = ChemAnchor(chem="X", temp_c=25.0, c_dis=1.0,
                         fade_ref_per_efc=2.5e-5, n_cells=1,
                         fade_std_per_efc=0.0, d_ref_s=8000.0)
        assert abs(anc.n_eol_ref - 8000.0) < 1e-6
        # 场景损伤翻倍 → 寿命减半
        assert abs(anc.n_eol(16000.0, 1.0) - 4000.0) < 1e-6

    def test_shape_invariance_under_clock_scale(self):
        """幂律口径下 k=(d/d_ref)·m 也是 1/k 缩放（时钟解释）。"""
        anc = ChemAnchor(chem="X", temp_c=25.0, c_dis=1.0,
                         fade_ref_per_efc=1e-4, n_cells=1,
                         fade_std_per_efc=0.0, d_ref_s=1000.0,
                         a_fade=1e-4, b_fade=0.5)
        n0 = anc.n_eol(1000.0, 1.0, shape="power")
        n1 = anc.n_eol(2000.0, 1.0, shape="power")
        assert abs(n1 / n0 - 0.5) < 1e-9

    def test_m_c_linear_in_damage(self):
        anc = ChemAnchor(chem="X", temp_c=25.0, c_dis=1.0,
                         fade_ref_per_efc=1e-4, n_cells=1,
                         fade_std_per_efc=0.0, d_ref_s=1000.0)
        n_m1 = anc.n_eol(1000.0, 1.0)
        n_m2 = anc.n_eol(1000.0, 2.0)
        assert abs(n_m2 / n_m1 - 0.5) < 1e-9


class TestCrossingRobust:
    def _curve_glitch(self):
        n = np.arange(1, 501, dtype=float)
        f = 5e-4 * n
        f[100] = 0.95  # 单点毛刺
        return n, f

    def test_glitch_does_not_fake_eol(self):
        n, f = self._curve_glitch()
        cross = first_crossing(n, f, 0.2)
        assert np.isfinite(cross)
        # 真值 0.2/5e-4=400；毛刺在 n=101 不得被当作 EOL
        assert abs(cross - 400.0) < 15.0

    def test_monotonicity_guard(self):
        # 20% 交叉早于 10% 交叉 → 病理曲线拒收
        n = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f = np.array([0.0, 0.3, 0.05, 0.15, 0.35])
        assert np.isnan(first_crossing(n, f, 0.2))

    def test_not_reached_returns_nan(self):
        n = np.arange(1, 51, dtype=float)
        f = 1e-4 * n
        assert np.isnan(first_crossing(n, f, 0.2))

    def test_audit_recovers_power_curve(self):
        """纯净幂律曲线：线性偏乐观、幂律近真值。"""
        n = np.arange(1, 801, dtype=float)
        f = 0.01 * n ** 0.5  # N(0.2)=400
        audit = audit_extrapolation([("syn", n, f)])
        rows = audit.to_dict()["rows"]
        assert len(rows) == 1
        r = rows[0]
        assert abs(r["n_cross"] - 400.0) < 20.0
        # 幂律口径应收敛到 ~400（窗口内完美幂律）
        assert abs(r["n_power"] - 400.0) / 400.0 < 0.15
        # b 还原
        assert abs(r["b"] - 0.5) < 0.05


class TestFadePowerFit:
    def test_recovers_exponent(self):
        n = np.arange(1, 301, dtype=float)
        f = 0.005 * n ** 0.7
        pw = fit_fade_power(n, f)
        assert abs(pw["b"] - 0.7) < 0.02
        assert pw["r2"] > 0.99

    def test_window_excludes_deep_fade(self):
        n = np.arange(1, 301, dtype=float)
        f = 0.002 * n  # 线性到 0.6
        pw = fit_fade_power(n, f, f_lo=0.01, f_hi=0.25)
        # 拟合点全部落在窗口
        assert pw["n_pts"] < 250


class TestEaFormula:
    def test_ea_roundtrip(self):
        r_gas = 8.314462618
        ea = 30_000.0
        t1, t2 = 25.0 + 273.15, 35.0 + 273.15
        ratio = np.exp(-ea / r_gas * (1.0 / t2 - 1.0 / t1))
        # 反解
        ea_back = r_gas * np.log(ratio) / (1.0 / t1 - 1.0 / t2)
        assert abs(ea_back - ea) < 1.0

    def test_higher_temp_fades_faster_positive_ea(self):
        r_gas = 8.314462618
        f1, f2 = 1.0, 1.5
        t1, t2 = 298.15, 308.15
        ea = r_gas * np.log(f2 / f1) / (1.0 / t1 - 1.0 / t2)
        assert ea > 0


class TestMFluxConversion:
    def test_mflux_identity_when_no_calendar(self):
        """s=0 → m_flux=m_meas；日历混入为零时换算是恒等。"""
        s, r_t, m_meas = 0.0, 0.7, 2.3
        m_flux = (m_meas - s * r_t) / (1.0 - s)
        assert abs(m_flux - m_meas) < 1e-12

    def test_mflux_strips_calendar_share(self):
        """m_meas 全部来自日历时（m_flux→0 应被护栏托底）。"""
        s, r_t, m_meas = 0.5, 1.0, 0.5  # m_meas = s·r_t = 0.5 → m_flux=0
        m_flux = (m_meas - s * r_t) / (1.0 - s)
        assert abs(m_flux) < 1e-12

    def test_bracket_identity_at_ref(self):
        assert abs(calendar_bracket(0.4, 100.0, 100.0) - 1.0) < 1e-12

    def test_bracket_rises_when_faster(self):
        # 场景时长 < 参考 → 括号 > 1（通量损伤不再被打折）
        assert calendar_bracket(0.4, 50.0, 100.0) > 1.0
