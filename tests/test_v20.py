# -*- coding: utf-8 -*-
"""v1.10 测试：无损退化模式分解（degrade_modes）。

覆盖：RPT 识别/去重、纯 shift/纯 shape 分解、边界闭式重估
（v1.9 幅度虚高坑回归）、fade 锚点插值、dQ/dV 峰跟踪。
"""
from __future__ import annotations

import numpy as np
import pytest

from bmthermal.degrade_modes import (
    _nnls2,
    anchor_profile,
    cell_mode_trajectory,
    dqdv_peaks,
    identify_rpt_segments,
    mode_split,
    resample_discharge,
    track_peaks,
)


def _v_curve(q: np.ndarray, qn: float = 1.0) -> np.ndarray:
    """类石墨两平台放电曲线（单调降）。"""
    x = q / qn
    return (3.45 - 0.45 * x
            - 0.12 * np.tanh((x - 0.35) / 0.05)
            + 0.10 * np.tanh((x - 0.70) / 0.05))


def _seg(q_end: float, cycle: int, n: int = 120, qn: float = 1.0,
         shift: float = 0.0, scale: float = 1.0) -> dict:
    """构造一条 RPT 段（V(q)=f(q/scale+shift) 支持纯 shift/纯 scale）。"""
    q = np.linspace(0.0, q_end, n)
    v = _v_curve(q / scale + shift, qn)
    t = np.linspace(0.0, 3600.0 * q_end / (0.5 * qn), n)
    I = np.full(n, -0.5 * qn)
    return {
        "cycle_number": cycle, "q": q, "v": v, "t": t,
        "temp": np.full(n, 25.0), "q_end": q_end, "i_mean": 0.5 * qn,
        # identify_rpt_segments 输入形态（循环 dict）
        "current_in_A": I,
        "voltage_in_V": v,
        "time_in_s": t,
        "discharge_capacity_in_Ah": q,
        "temperature_in_C": np.full(n, 25.0),
    }


def _cycle(seg: dict) -> dict:
    """RPT 段 dict → 循环 dict（identify_rpt_segments 的输入）。"""
    n = len(seg["q"])
    return {
        "cycle_number": seg["cycle_number"],
        "current_in_A": seg["current_in_A"],
        "voltage_in_V": seg["voltage_in_V"],
        "time_in_s": seg["time_in_s"],
        "discharge_capacity_in_Ah": seg["discharge_capacity_in_Ah"],
        "temperature_in_C": seg["temperature_in_C"],
        "internal_resistance_in_ohm": None,
    }


class TestRptIdentify:
    def test_fine_halfc_accepted_coarse_3c_rejected(self):
        fine = _cycle(_seg(1.0, 10))
        # 3C 粗采样（120 s → 9 点）被拒
        coarse = _cycle(_seg(1.0, 11, n=9))
        coarse["current_in_A"] = np.full(9, -3.0)
        segs = identify_rpt_segments([fine, coarse], nominal_ah=1.0)
        assert [s["cycle_number"] for s in segs] == [10]

    def test_duplicate_entries_deduped(self):
        a = _cycle(_seg(1.0, 1))
        b = _cycle(_seg(1.0, 2))  # 同签名（同点数/同跨度/同容量）
        segs = identify_rpt_segments([a, b], nominal_ah=1.0)
        assert len(segs) == 1


class TestModeSplit:
    def test_pure_shift_is_shift_side(self):
        q_ref, v_ref = np.linspace(0, 1, 200), _v_curve(np.linspace(0, 1, 200))
        qk = np.linspace(0, 0.95, 200)
        vk = _v_curve(qk + 0.05)          # Q_k(V) = Q0(V) − 0.05
        d = mode_split(q_ref, v_ref, qk, vk, cycle_number=5)
        assert d.q_shift == pytest.approx(0.05, rel=0.1)
        assert d.share_shape < 0.05
        assert d.qsum_err < 0.1

    def test_pure_scale_is_shape_side(self):
        q_ref, v_ref = np.linspace(0, 1, 200), _v_curve(np.linspace(0, 1, 200))
        qk = np.linspace(0, 0.90, 200)
        vk = _v_curve(qk / 0.90)           # Q_k(V) = 0.9·Q0(V)
        d = mode_split(q_ref, v_ref, qk, vk, cycle_number=5)
        assert d.share_shape > 0.90
        assert d.q_shape == pytest.approx(0.10, rel=0.15)

    def test_boundary_refit_keeps_amplitude(self):
        """v1.9 坑回归：边界解必须闭式重估留下那支，幅度不虚高。"""
        q0 = np.linspace(0.1, 0.9, 11)
        y = 0.05 + 0.0 * q0                    # 纯 shift
        coef, name = _nnls2(np.column_stack([np.ones_like(q0), q0]), y)
        assert name in ("interior", "shift_only")
        assert coef[0] == pytest.approx(0.05, rel=1e-6)
        assert coef[1] == pytest.approx(0.0, abs=1e-9)

    def test_anchor_profile_loss_constant_under_shift(self):
        q_ref = np.linspace(0, 1, 200)
        v_ref = _v_curve(q_ref)
        qk = np.linspace(0, 0.93, 200)
        vk = _v_curve(qk + 0.07)
        prof = anchor_profile(q_ref, v_ref, qk, vk)
        assert np.allclose(prof["loss"], 0.07, atol=0.01)
        assert prof["loss_eod"] == pytest.approx(0.07, rel=0.05)


class TestPdqdvAndTrack:
    def test_flat_curve_yields_no_spurious_peaks(self):
        q = np.linspace(0, 1, 200)
        v = 3.3 - 0.2 * q                     # 近直线 → 无平台峰
        pk = dqdv_peaks(q, v)
        assert pk.size == 0 or len(pk) <= 1

    def test_track_peaks_shift_and_height(self):
        ref = np.array([[3.40, 10.0], [3.20, 8.0]])
        cur = np.array([[3.38, 9.0], [3.18, 4.0]])
        r = track_peaks(ref, cur, tol_v=0.05)
        assert r["matched"] == 2
        assert r["mean_shift_v"] == pytest.approx(-0.02, abs=1e-9)
        # 两值 median=均值（v1.9 numpy 坑）：(0.9+0.5)/2 = 0.7
        assert r["height_ratio"] == pytest.approx(0.7, rel=0.01)


class TestCellTrajectory:
    def test_scale_aging_yields_shape_dominant(self):
        segs = [
            _seg(1.00, 1, scale=1.00),
            _seg(0.95, 101, scale=0.95),
            _seg(0.90, 201, scale=0.90),
        ]
        cycles = [_cycle(s) for s in segs]
        cm = cell_mode_trajectory(cycles, nominal_ah=1.0, cell_id="t",
                                  meta={"chem": "LFP", "temp_c": 25.0,
                                        "c_dis": 3.0, "rep": "a"})
        assert cm.n_rpt == 3
        assert cm.share_shape_eol > 0.8
        assert cm.fade_at_eol == pytest.approx(0.10, rel=0.05)

    def test_shift_aging_yields_shift_dominant(self):
        segs = [
            _seg(1.00, 1, shift=0.0),
            _seg(0.96, 101, shift=0.04),
            _seg(0.92, 201, shift=0.08),
        ]
        cm = cell_mode_trajectory([_cycle(s) for s in segs], nominal_ah=1.0)
        assert cm.share_shape_eol < 0.2

    def test_fade_anchor_interpolation(self):
        segs = [
            _seg(1.00, 1),
            _seg(0.95, 101, scale=0.95),
            _seg(0.85, 201, scale=0.85),
        ]
        cm = cell_mode_trajectory([_cycle(s) for s in segs], nominal_ah=1.0,
                                  fade_anchor=0.10)
        assert np.isfinite(cm.share_shape_at_fade)


class TestResample:
    def test_resample_monotonic_grid(self):
        seg = _seg(1.0, 1)
        qg, vg = resample_discharge(seg)
        assert qg[0] == 0.0 and qg[-1] == pytest.approx(1.0)
        assert np.all(np.diff(qg) > 0)
        assert vg[-1] < vg[0]              # 放电曲线整体下降（平台特征允许局部起伏）
