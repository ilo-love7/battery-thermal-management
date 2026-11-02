# -*- coding: utf-8 -*-
"""v1.9 测试：机理点归因（mech_attribution）+ 温度交互统计（rate_effect 新增）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bmthermal.mech_attribution import (
    anchored_excess_decomp,
    fit_sei_amplitude,
    shape_fingerprint_group,
    smooth_fade,
    two_component_fit,
)
from bmthermal.rate_effect import (
    apparent_ea_from_fade,
    matched_rep_m,
    pair_m_ratio,
)


def _curve_sei(n_max=2000, a=1e-4, b=0.8):
    n = np.arange(1, n_max + 1, dtype=float)
    f = a * n ** b
    return n, f


def _curve_mixed(n_max=2000, a_s=1e-4, b=0.8, a_m=3e-5):
    n = np.arange(1, n_max + 1, dtype=float)
    f = a_s * n ** b + a_m * n
    return n, f


def _curve_linear(n_max=2000, a_m=8e-5):
    n = np.arange(1, n_max + 1, dtype=float)
    f = a_m * n
    return n, f


class TestTwoComponentFit:
    def test_pure_sei_goes_sei_only(self):
        n, f = _curve_sei()
        d = two_component_fit(n, f, b_sei=0.8, f_target=0.2)
        assert d.boundary == "sei_only"
        assert d.mech_share == pytest.approx(0.0, abs=1e-9)

    def test_pure_linear_goes_mech_side(self):
        n, f = _curve_linear()
        d = two_component_fit(n, f, b_sei=0.8, f_target=0.2)
        # 共线基下可能 interior（a_s 数值小正值）——线性侧判定看份额
        assert d.boundary in ("mech_only", "interior")
        assert d.mech_share > 0.95
        assert d.a_mech == pytest.approx(8e-5, rel=0.1)

    def test_boundary_refit_amplitude_not_joint_residual(self):
        """v1.9 坑回归：截断后沿用联合解幅度会虚高。"""
        n, f = _curve_sei()
        d = two_component_fit(n, f, b_sei=0.8, f_target=0.2)
        assert d.a_sei == pytest.approx(1e-4, rel=0.1)
        # 纯 SEI 数据上单分量 RSS 应≈混合 RSS（线性幅度已闭式归零）
        assert d.rss == pytest.approx(d.rss_sei_only, rel=1e-6)

    def test_mixed_share_between_extremes(self):
        n, f = _curve_mixed()
        d = two_component_fit(n, f, b_sei=0.8, f_target=0.2)
        assert 0.1 < d.mech_share < 0.9

    def test_basis_corr_reported(self):
        n, f = _curve_sei()
        d = two_component_fit(n, f, b_sei=0.8, f_target=0.2)
        assert d.basis_corr > 0.98  # 共线性警示数值化


class TestAnchoredExcess:
    def test_recovers_mech_share_synthetic(self):
        """锚定真 a_s：混合曲线的机械份额可回收。"""
        a_s_true, b, a_m_true = 1e-4, 0.8, 3e-5
        n, f = _curve_mixed(2000, a_s_true, b, a_m_true)
        d = anchored_excess_decomp(n, f, a_s_true, b, 0.2)
        # 在 f=0.2 处的模型份额
        n_t = d.n_at_target
        expected = (a_m_true * n_t) / (a_s_true * n_t ** b + a_m_true * n_t)
        assert d.mech_share == pytest.approx(expected, rel=0.05)
        assert not d.excess_negative

    def test_negative_excess_flagged(self):
        """实测低于锚定本底 → 超额为负，份额 0 + 标记。"""
        n, f = _curve_sei(2000, a=5e-5, b=0.8)   # 比锚小一半
        d = anchored_excess_decomp(n, f, 1e-4, 0.8, 0.2)
        assert d.excess_negative
        assert d.mech_share == pytest.approx(0.0, abs=1e-9)

    def test_fit_sei_amplitude_roundtrip(self):
        n, f = _curve_sei(2000, a=2e-4, b=0.75)
        a = fit_sei_amplitude(n, f, 0.75)
        assert a == pytest.approx(2e-4, rel=0.05)

    def test_smoothing_shared_convention(self):
        f = np.ones(50)
        f[20] = 0.95  # 单点毛刺
        fs = smooth_fade(f)
        assert abs(fs[20] - 1.0) < 0.02  # 毛刺被中位压掉


class TestShapeFingerprint:
    def test_fingerprint_from_summary(self):
        df = pd.DataFrame([
            dict(chem="LFP", temp_c=25.0, c_dis=3.0, b_fade=0.95,
                 fade_rate_per_efc=6e-5),
            dict(chem="LFP", temp_c=25.0, c_dis=3.0, b_fade=0.94,
                 fade_rate_per_efc=5e-5),
            dict(chem="LFP", temp_c=25.0, c_dis=1.0, b_fade=0.89,
                 fade_rate_per_efc=2.5e-5),
            dict(chem="LFP", temp_c=25.0, c_dis=1.0, b_fade=0.88,
                 fade_rate_per_efc=2.4e-5),
        ])
        fp = shape_fingerprint_group(df, "LFP", 25.0, 3.0)
        assert fp.n_cells == 2
        assert fp.db_vs_ref == pytest.approx(0.945 - 0.885, abs=1e-6)
        assert fp.cv_fade > 0


class TestPairRatio:
    def test_eligibility_needs_two_sides(self):
        r = pair_m_ratio([2.0, 2.1], [1.0])
        assert not r["eligible"]
        r2 = pair_m_ratio([2.0, 2.1], [1.0, 1.1])
        assert r2["eligible"]
        assert r2["m"] == pytest.approx(2.05 / 1.05, rel=1e-6)
        assert r2["ci95"][0] <= r2["m"] <= r2["ci95"][1]

    def test_matched_rep_controls_batch(self):
        rows_fast = [("a", 2.0), ("b", 4.0)]
        rows_ref = [("a", 1.0), ("b", 4.0), ("c", 1.0)]  # c 无配对
        r = matched_rep_m(rows_fast, rows_ref)
        assert r["eligible"]
        assert r["n_pairs"] == 2
        # 比值 [2.0, 1.0]，中位=1.5（两值中位=均值）——c 的 2.0 未混入
        assert r["m_matched"] == pytest.approx(1.5)

    def test_matched_needs_pairs(self):
        r = matched_rep_m([("z", 2.0)], [("a", 1.0)])
        assert not r["eligible"]


class TestApparentEa:
    def test_positive_ea_eligible(self):
        # fade 升 → Ea>0
        r = apparent_ea_from_fade(2.0, 1.0, 25.0, 35.0)
        assert r["eligible"]
        assert r["ea_kj_mol"] == pytest.approx(52.9, rel=0.02)  # 10K 翻倍

    def test_inverted_axis_disqualified(self):
        # fade 降（NMC 25→35 反向）→ 负 Ea → 资格否决
        r = apparent_ea_from_fade(0.6, 1.0, 25.0, 35.0)
        assert not r["eligible"]
        assert r["ea_kj_mol"] < 0

    def test_roundtrip_ea_value(self):
        r_gas = 8.314462618
        ea = 30_000.0
        t1, t2 = 25.0 + 273.15, 35.0 + 273.15
        ratio = np.exp(-ea / r_gas * (1.0 / t2 - 1.0 / t1))
        r = apparent_ea_from_fade(1.0 * ratio, 1.0, 25.0, 35.0)
        assert r["ea_kj_mol"] == pytest.approx(30.0, rel=0.01)
