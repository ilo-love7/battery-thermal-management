"""v1.11 测试：实验方案量化（exp_design）。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from bmthermal.exp_design import (
    M5_HI, M5_LO, M5_STAR,
    decision_5c, decision_postmortem, design_matrix, m_ratio_se,
    power_curve, power_two_group, required_sigma_q, sample_size_two_group,
    slope_se, z_quantile,
)


class TestZQuantile:
    def test_known_values(self):
        assert z_quantile(0.975) == pytest.approx(1.95996, abs=1e-4)
        assert z_quantile(0.5) == pytest.approx(0.0, abs=1e-9)
        assert z_quantile(0.8) == pytest.approx(0.8416, abs=1e-3)

    def test_symmetry(self):
        assert z_quantile(0.1) == pytest.approx(-z_quantile(0.9), abs=1e-8)

    def test_bad_p(self):
        with pytest.raises(ValueError):
            z_quantile(0.0)
        with pytest.raises(ValueError):
            z_quantile(1.0)


class TestPower:
    def test_large_effect_full_power(self):
        assert power_two_group(5.0, 1.0, 4) > 0.99

    def test_zero_effect_alpha(self):
        # δ=0 时方向性功效 = α/2（单侧约定；样本量公式用 z_a+z_b 同口径）
        assert power_two_group(0.0, 1.0, 8) == pytest.approx(0.025, abs=1e-3)

    def test_monotone_in_n(self):
        ps = [power_two_group(0.8, 1.0, n) for n in (2, 4, 8, 16)]
        assert all(ps[i] < ps[i + 1] for i in range(len(ps) - 1))

    def test_sample_size_roundtrip(self):
        n = sample_size_two_group(0.8, 1.0, power=0.80)
        assert power_two_group(0.8, 1.0, n) >= 0.80
        # n−1 一般达不到（边界情形容让）
        if n > 2:
            assert power_two_group(0.8, 1.0, n - 1) < 0.82

    def test_sample_size_scales(self):
        # δ 翻倍 → n 约 1/4
        n1 = sample_size_two_group(0.5, 1.0)
        n2 = sample_size_two_group(1.0, 1.0)
        assert n2 == pytest.approx(n1 / 4, rel=0.25)

    def test_power_curve_shape(self):
        pc = power_curve(1.0, 1.0, n_max=10)
        assert pc["n"] == list(range(2, 11))
        assert len(pc["power"]) == 9
        assert pc["power"][-1] > pc["power"][0]


class TestPrecisionBudget:
    def test_slope_se_zero_noise(self):
        assert slope_se(0.0, 500.0, 20) == 0.0

    def test_slope_se_shrinks_with_window(self):
        s1 = slope_se(0.002, 200.0, 20)
        s2 = slope_se(0.002, 400.0, 20)
        assert s2 == pytest.approx(s1 / 2, rel=1e-9)

    def test_required_sigma_q_roundtrip(self):
        sig = required_sigma_q(1e-5, 300.0, 15)
        assert slope_se(sig, 300.0, 15) == pytest.approx(1e-5, rel=1e-9)

    def test_m_ratio_se_scales(self):
        se1 = m_ratio_se(2e-5, 5e-5, 3e-6, 5e-6, 4, 4)
        se2 = m_ratio_se(2e-5, 5e-5, 3e-6, 5e-6, 16, 16)
        assert se2 == pytest.approx(se1 / 2, rel=1e-9)
        assert se1 > 0


class TestDecisionRules:
    def test_5c_positive(self):
        assert decision_5c(2.6, 2.3, 2.9) == "throughput_positive"

    def test_5c_negative(self):
        assert decision_5c(3.3, 3.15, 3.45) == "throughput_negative"

    def test_5c_inconclusive(self):
        assert decision_5c(3.0, 2.5, 3.4) == "inconclusive"

    def test_5c_bound_variant(self):
        assert decision_5c(2.35, 2.25, 2.35) == "throughput_positive_at_lower_bound"
        assert decision_5c(3.34, 3.31, 3.40) == "throughput_negative_at_upper_bound"

    def test_5c_ci_must_contain(self):
        with pytest.raises(ValueError):
            decision_5c(3.0, 3.2, 3.4)

    def test_postmortem_readings(self):
        assert decision_postmortem(0.6, 0.3, 2.0, 1.0) == "reading_b_confirmed"
        assert decision_postmortem(0.2, 0.4, 0.8, 1.2) == "reading_a_gains_weight"
        assert decision_postmortem(0.6, 0.3, 0.8, 1.2) == "mixed_evidence"


class TestDesignMatrix:
    def test_counts(self):
        rows = design_matrix()
        # 2 chem × (2 rates × 3 temps target + 3 temps ref) = 2×9 = 18
        assert len(rows) == 18
        assert sum(r["role"] == "reference" for r in rows) == 6

    def test_custom(self):
        rows = design_matrix(c_dis_list=(5.0,), temps=(25.0,),
                             chems=("LFP",), n_per_cell=6)
        assert len(rows) == 2          # 5C target + 1C ref
        assert all(r["n_cells"] == 6 for r in rows)
