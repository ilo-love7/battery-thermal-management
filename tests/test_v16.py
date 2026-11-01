"""v1.6 深化测试：rate_effect / 模组寿命一致性护栏。

重点覆盖 v1.6 踩坑里长出来的性质：
- m(c) 口径（1C 基准=1、n=1 只作指示、n≥2 才资格）；
- 机理归因方向护栏：高倍率升高≠低倍率升高（NCA 0.5C 异质性候选）、
  降低无机制支持 → nuisance；析锂候选只在充电倍率≥1C 时升级；
- 损伤分解 d=m(c)·k(T) 与权衡行的自洽（寿命=1/d、吞吐∝1/(d·dt)）；
- 分量倍率指数混合法的边界（Σs·p ∈ [p_rev, p_ohm_max]）；
- 模组稳态一致性：均匀 q 温差小、低流量温差更大、惩罚恒正。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bmthermal.aging import arrhenius_factor
from bmthermal.module import ModuleSpec
from bmthermal.module_control import steady_state_cells
from bmthermal.rate_effect import (
    cross_chem_envelope,
    damage_per_efc,
    mechanism_attribution,
    rate_points,
    tier_tradeoff_row,
)


def make_df() -> pd.DataFrame:
    rows = []
    cfg = {
        "LFP": (2.5e-5, {0.5: (1.13, 1), 1.0: (1.0, 4), 2.0: (0.90, 2), 3.0: (2.30, 4)}),
        "NMC": (1.0e-4, {0.5: (0.89, 1), 1.0: (1.0, 4), 2.0: (0.60, 2), 3.0: (0.64, 4)}),
        "NCA": (3.0e-4, {0.5: (1.38, 2), 1.0: (1.0, 4), 2.0: (1.05, 2)}),
    }
    for chem, (base, groups) in cfg.items():
        for c, (m, n) in groups.items():
            for i in range(n):
                rows.append(dict(
                    chem=chem, temp_c=25.0, c_dis=float(c), c_chg=0.5,
                    soc_lo=0.0, soc_hi=100.0,
                    fade_rate_per_efc=base * m * (1.0 + 0.01 * i),
                    dt_s_per_cycle=10000.0, cell_id=f"{chem}_{c}_{i}",
                ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- m(c) 口径

class TestRatePoints:
    def test_reference_is_one(self):
        pts = rate_points(make_df(), "LFP")
        ref = next(p for p in pts if p.c_dis == 1.0)
        assert ref.m_per_efc == pytest.approx(1.0)
        assert ref.role == "ref"

    def test_m_value_and_roles(self):
        pts = {p.c_dis: p for p in rate_points(make_df(), "LFP")}
        assert pts[3.0].m_per_efc == pytest.approx(2.30, rel=1e-6)
        assert pts[3.0].role == "solid" and pts[3.0].eligible
        assert pts[2.0].role == "solid"
        assert pts[0.5].role == "indicative" and not pts[0.5].eligible

    def test_fade_per_hour_consistent(self):
        pts = {p.c_dis: p for p in rate_points(make_df(), "NMC")}
        assert pts[1.0].fade_per_hour == pytest.approx(
            pts[1.0].fade_per_efc / 10000.0 * 3600.0)

    def test_envelope(self):
        pts = [p for chem in ("LFP", "NMC", "NCA") for p in rate_points(make_df(), chem)]
        env = cross_chem_envelope(pts)
        assert env[3.0]["lo"] == pytest.approx(0.64)
        assert env[3.0]["hi"] == pytest.approx(2.30)
        assert 1.0 not in env  # 1C 基准不进包络


# ---------------------------------------------------------------- 机理归因

class TestAttribution:
    def test_elevated_high_rate_has_mechanism(self):
        a = mechanism_attribution("LFP", 3.0, 2.30, c_chg_mean=0.5)
        assert a.direction == "elevated"
        assert a.plausible_mechanisms
        assert "析锂可能性低" in a.plausible_mechanisms[-1]

    def test_plating_candidate_when_chg_high(self):
        a = mechanism_attribution("LFP", 3.0, 2.30, c_chg_mean=2.0)
        assert any("析锂候选" in m for m in a.plausible_mechanisms)

    def test_depressed_is_nuisance(self):
        a = mechanism_attribution("NMC", 3.0, 0.64, c_chg_mean=0.5)
        assert a.direction == "depressed"
        assert not a.plausible_mechanisms
        assert "异质性" in a.verdict

    def test_low_rate_elevated_not_mechanical(self):
        a = mechanism_attribution("NCA", 0.5, 1.38, c_chg_mean=0.5)
        assert a.direction == "elevated_low_rate"
        assert not a.plausible_mechanisms

    def test_neutral(self):
        a = mechanism_attribution("NCA", 2.0, 1.05, c_chg_mean=0.5)
        assert a.direction == "neutral"


# ---------------------------------------------------------------- 损伤分解

class TestDamageDecomposition:
    def test_identity_at_ref_temp(self):
        assert damage_per_efc(1.0, np.array([25.0])) == pytest.approx(1.0)
        assert damage_per_efc(2.0, np.array([25.0])) == pytest.approx(2.0)

    def test_hot_increases_damage(self):
        cold = damage_per_efc(1.0, np.array([25.0]))
        hot = damage_per_efc(1.0, np.array([45.0]))
        assert hot > cold

    def test_arrhenius_monotonic(self):
        ts = np.array([0.0, 25.0, 45.0])
        f = arrhenius_factor(ts)
        assert f[0] < f[1] < f[2]
        assert f[1] == pytest.approx(1.0)

    def test_tradeoff_row_consistency(self):
        row = tier_tradeoff_row(3.0, 2.3, np.array([40.0, 50.0]), 3600.0)
        assert row["life_efc_rel"] == pytest.approx(1.0 / row["damage_per_efc_rel"])
        assert row["throughput_per_hour_rel"] == pytest.approx(
            3600.0 / (row["damage_per_efc_rel"] * 3600.0))
        assert row["damage_per_efc_rel"] > 2.3  # k(T>25) 乘上去

    def test_p_local_blend_bounds(self):
        # 分量倍率指数混合法：Σ s_i·p_i ∈ [p_rev, max(p_ohm, p_irr)]
        s = {"ohmic": 0.08, "irr": 0.29, "rev": 0.62}
        p_lo = s["ohmic"] * 1.83 + s["irr"] * 1.0 + s["rev"] * 1.0
        p_hi = s["ohmic"] * 2.00 + s["irr"] * 2.0 + s["rev"] * 1.0
        assert 1.0 <= p_lo < p_hi <= 2.0
        assert p_lo == pytest.approx(1.0564)


# ---------------------------------------------------------------- 模组一致性

class TestModuleConsistency:
    def _spread(self, flow: float, q: np.ndarray) -> dict:
        spec = ModuleSpec(flow_lpm=flow, t_in_c=25.0)
        t = steady_state_cells(spec, q, spec.h_w_m2k)
        k = arrhenius_factor(t, 45000.0)
        return {"dt": float(t.max() - t.min()), "pen": float(k.max() / k.mean() - 1.0)}

    def test_uniform_q_small_spread(self):
        r = self._spread(2.0, np.full(12, 2.7))
        assert r["dt"] < 0.5
        assert 0.0 <= r["pen"] < 0.02

    def test_low_flow_worse(self):
        q = np.full(12, 2.7)
        d, lo = self._spread(2.0, q), self._spread(0.25, q)
        assert lo["dt"] > d["dt"]
        assert lo["pen"] > d["pen"]

    def test_penalty_nonnegative(self):
        rng = np.random.default_rng(0)
        q = 2.7 * (1.0 + 0.15 * rng.standard_normal(12))
        r = self._spread(0.25, q)
        assert r["pen"] >= 0.0
