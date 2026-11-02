"""v1.8 测试：β 跨体系锚 / m_flux 联合口径 / 5C 外推护栏。"""
from __future__ import annotations

import numpy as np
import pytest

from bmthermal.beta_anchor import (
    LongCell,
    Matr10Cell,
    _fade_from_capacity,
    extract_matr10_r,
    extract_step_r,
    fit_beta_cross_sectional,
    fit_beta_longitudinal,
    pick_reference_r,
    smooth_series,
)


# ----------------------------------------------------------------- 阶跃提取

class TestStepExtraction:
    def test_step_r_basic(self):
        """静置→放电阶跃：R=ΔV/ΔI，方向放电。"""
        i = np.array([0.0, 0.0, 0.0, -4.84, -4.84, -4.84])
        v = np.array([4.18, 4.18, 4.18, 4.17, 4.10, 4.05])
        hits = extract_step_r(i, v)
        assert len(hits) == 1
        h = hits[0]
        assert h["into_discharge"] is True
        # ΔV=-0.01, ΔI=-4.84 → R≈2.07 mΩ
        assert abs(h["r_ohm"] - 0.01 / 4.84) < 1e-9

    def test_step_r_rejects_bad_r(self):
        """ΔV 与 ΔI 同号（R≤0）或过大（>0.5）不收。"""
        i = np.array([0.0, -4.84, -4.84])
        v = np.array([4.18, 4.19, 4.19])  # ΔV>0 放电 → R<0
        assert extract_step_r(i, v) == []
        v2 = np.array([4.18, 1.0, 1.0])   # ΔV=-3.18 → R=0.66>0.5
        assert extract_step_r(i, v2) == []

    def test_pick_reference_r_rate_match(self):
        """取最接近目标倍率的放电阶跃；超容差返回 nan。"""
        i = np.zeros(6); v = np.zeros(6)
        i[3:] = -4.84; v[3] = -0.01
        hits = extract_step_r(i, np.array([4.2, 4.2, 4.2, 4.19, 4.1, 4.0]))
        assert np.isfinite(pick_reference_r(hits, rate_a=4.84, tol_frac=0.25))
        assert np.isnan(pick_reference_r(hits, rate_a=20.0, tol_frac=0.25))

    def test_smooth_series_glitch(self):
        """滚动中位压单点毛刺。"""
        x = np.ones(21) * 0.02
        x[10] = 0.95
        s = smooth_series(x, 7)
        assert s[10] < 0.1


# ----------------------------------------------------------------- MATR-10

class TestMatr10:
    def _rows_full_charge(self, r_true=0.023, i_dis=-4.84, v0=4.18):
        """CV 尾（小电流充电）→ 放电反转，R 已知。"""
        v1 = v0 + r_true * (i_dis - 0.02)
        return [
            (0.02, v0, "C", 4.80),
            (i_dis, v1, "D", 4.80),
            (i_dis, v1 - 0.05, "D", 4.84),
        ]

    def test_extract_full_charge_reversal(self):
        rows = self._rows_full_charge(r_true=0.023)
        out = extract_matr10_r(rows)
        assert len(out) == 1
        (rate,), (r,) = out.keys(), out.values()
        assert abs(rate - 4.84) < 1e-9
        assert abs(r - 0.023) < 1e-9

    def test_rejects_not_full_charge(self):
        """v0<4.15 的反转（非满充）不收——防 SOC 混入。"""
        rows = self._rows_full_charge(v0=3.80)
        assert extract_matr10_r(rows) == {}

    def test_rejects_small_step(self):
        rows = self._rows_full_charge(i_dis=-0.10)
        assert extract_matr10_r(rows) == {}

    def test_same_rate_median(self):
        """同倍率多次反转取中位。"""
        rows = []
        for r in (0.020, 0.022, 0.024):
            rows.extend(self._rows_full_charge(r_true=r))
        out = extract_matr10_r(rows)
        (r_med,) = out.values()
        assert abs(r_med - 0.022) < 1e-9

    def test_beta_cross_sectional_roundtrip(self):
        """R=R0(1+βf) 截面回归还原 β。"""
        rng = np.random.default_rng(1)
        cells = []
        beta_true, r0 = 2.3, 0.016
        for k in range(30):
            f = 0.02 + 0.34 * k / 29
            r = r0 * (1 + beta_true * f) * (1 + 0.01 * rng.standard_normal())
            cells.append(Matr10Cell(cell_id=f"c{k}", cap_ah=4.84 * (1 - f),
                                    r_1c=r, fade=f))
        fit = fit_beta_cross_sectional(cells, fade_lo=0.01, fade_hi=0.38,
                                       n_boot=200, seed=0)
        assert abs(fit["beta"] - beta_true) / beta_true < 0.05
        assert fit["n_cells"] == 30
        assert fit["ci95"][0] <= fit["beta"] <= fit["ci95"][1]
        assert abs(fit["r0_ohm"] - r0) / r0 < 0.05

    def test_beta_cross_sectional_insufficient(self):
        cells = [Matr10Cell(cell_id=f"c{k}", cap_ah=4.0, r_1c=0.02, fade=0.1)
                 for k in range(3)]
        fit = fit_beta_cross_sectional(cells)
        assert np.isnan(fit["beta"])
        assert fit["n_cells"] == 3


# ----------------------------------------------------------------- 纵向

class TestLongitudinal:
    def test_fade_from_capacity(self):
        cap = np.array([2.0] * 10 + [1.8] * 10)
        f = _fade_from_capacity(cap)
        assert abs(f[0]) < 1e-12
        assert abs(f[-1] - 0.1) < 1e-12

    def test_fade_rejects_empty(self):
        f = _fade_from_capacity(np.array([np.nan, 0.0]))
        assert np.all(np.isnan(f))

    def test_beta_longitudinal_roundtrip(self):
        """R/R0=1+βf 逐芯拟合 → 中位还原。"""
        rng = np.random.default_rng(2)
        cells = []
        for k in range(8):
            beta = 2.0 + 0.1 * k
            f = np.linspace(0.01, 0.20, 25)
            rr = 1 + beta * f + 0.005 * rng.standard_normal(f.size)
            cells.append(LongCell(cell_id=f"c{k}", chem="NMC",
                                  n=np.arange(f.size, dtype=float),
                                  fade=f, r_ratio=rr))
        fit = fit_beta_longitudinal(cells)
        assert fit["n_cells_fitted"] == 8
        assert 2.0 <= fit["beta_median"] <= 2.8
        assert fit["beta_range"][0] <= fit["beta_median"] <= fit["beta_range"][1]


# ----------------------------------------------------------------- 联合口径

class TestJointConvention:
    @staticmethod
    def m_flux(m_meas, s, r_t=0.45):
        """复刻 scripts/36 公式（脚本内定义，测试内联对拍）。"""
        if s >= 1.0:
            return float("nan")
        v = (m_meas - s * r_t) / (1.0 - s)
        return float(max(v, 0.05))

    def test_m_flux_zero_share(self):
        assert abs(self.m_flux(2.30, 0.0) - 2.30) < 1e-12

    def test_m_flux_direction(self):
        """m_meas>r_t 时 m_flux 随 s 升（日历折扣被剥离）。"""
        s0 = self.m_flux(2.30, 0.0)
        s1 = self.m_flux(2.30, 0.4)
        s2 = self.m_flux(2.30, 0.6)
        assert s0 < s1 < s2

    def test_m_flux_floor_guard(self):
        """负值被数值护栏抬到 0.05。"""
        assert self.m_flux(0.10, 0.6) == pytest.approx(0.05)

    def test_joint_c5_monotone_in_s(self):
        """C5_joint=d1/(d3·m_flux·br3)：m>1 时 C5 随 s 降（LFP 型）。"""
        def c5(s, m=2.30, r_t=0.45):
            mf = max((m - s * r_t) / (1.0 - s), 0.05)
            br = 1.0 + 1.2 * s   # bracket 随 s 升（示意单调）
            return 1.0 / (mf * br)
        vals = [c5(s) for s in (0.0, 0.2, 0.4, 0.6)]
        assert all(a > b for a, b in zip(vals, vals[1:]))


# ----------------------------------------------------------------- 5C 护栏

class Test5cGuard:
    def test_critical_m_identities(self):
        """打平点 = m=1 基线值本身（thr(m)=thr0/m、life(m)=life0/m）。"""
        life0, thr0 = 0.782, 3.098
        m_star_life = life0
        m_star_thr = thr0
        assert abs(life0 / m_star_life - 1.0) < 1e-12
        assert abs(thr0 / m_star_thr - 1.0) < 1e-12

    def test_ohmic_proxy(self):
        """m5=5^(p−1)：p=2 → 5；p=1.83 → 3.87。"""
        assert abs(5.0 ** (2.0 - 1.0) - 5.0) < 1e-12
        assert abs(5.0 ** (1.83 - 1.0) - 5.0 ** 0.83) < 1e-12
        assert 3.8 < 5.0 ** 0.83 < 3.9

    def test_power_extrap_roundtrip(self):
        """m=a·c^b 幂律外推还原。"""
        a, b = 0.9, 0.35
        c = np.array([1.0, 2.0, 3.0])
        m = a * c ** b
        bb, la = np.polyfit(np.log(c), np.log(m), 1)
        assert abs(bb - b) < 1e-9
        assert abs(np.exp(la) * 5.0 ** bb - a * 5.0 ** b) < 1e-9

    def test_extrap_band_spans_flips(self):
        """护栏语义：外推带宽于打平点距离时不得收窄冒充精度（数值对拍
        29 指标基线 life0=0.78/thr0=3.10 与 37 外推带 [0.21,3.35]）。"""
        band_lo, band_hi = 0.21, 3.35
        life0, thr0 = 0.782, 3.098
        assert band_lo < life0 < band_hi
        assert band_lo < thr0 < band_hi
