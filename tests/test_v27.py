# -*- coding: utf-8 -*-
"""test_v27 — v1.16：V(q) 曲线提取 / band 分段速率 / mode_split 轨迹。"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from bmthermal.eis_features import (
    block_cc_discharge_capacity, block_cc_discharge_curve,
)
from bmthermal.degrade_modes import mode_split_trajectory
from bmthermal.efc_rate import segment_band_rates
import pandas as pd

OXFORD = Path("D:/项目/02_数据集/01_国际公开集/Oxford_PathDependent")


# ---------------------------------------------------------------------------
# block_cc_discharge_curve
# ---------------------------------------------------------------------------

class TestDischargeCurve:
    def _df(self, t, i, v):
        return pd.DataFrame({"TestTime": t, "Amps": i, "Volts": v})

    def test_basic_curve(self):
        """合成 CC 放电段：q_end 与 V(q) 单调性正确。"""
        n = 100
        t = np.arange(n, dtype=float)          # 分钟，1 min 间隔
        i = np.full(n, -1.5)                   # 恒流放电
        v = np.linspace(4.1, 3.5, n)
        out = block_cc_discharge_curve(self._df(t, i, v))
        assert out["n_pts"] == n
        assert out["i_med"] == pytest.approx(-1.5)
        # 1.5 A × 99 min / 60 ≈ 2.475 Ah
        assert out["q_end"] == pytest.approx(2.475, rel=1e-6)
        assert np.all(np.diff(out["q"]) >= -1e-12)   # 单调
        assert out["q"][0] == 0.0
        assert len(out["v"]) == n

    def test_picks_largest_segment(self):
        """两段放电（中间隔静置）取容量大的段。"""
        n1, n2, nr = 20, 80, 5
        t = np.arange(n1 + nr + n2, dtype=float)
        i = np.concatenate([np.full(n1, -1.5), np.zeros(nr),
                            np.full(n2, -1.5)])
        v = np.concatenate([np.linspace(4.1, 3.8, n1), np.full(nr, 3.8),
                            np.linspace(4.1, 3.2, n2)])
        out = block_cc_discharge_curve(self._df(t, i, v))
        assert out["n_pts"] == n2
        assert out["q_end"] == pytest.approx(1.5 * (n2 - 1) / 60.0, rel=1e-6)

    def test_v_sanitized_aligned(self):
        """电压列 NaN 样本被同步剔除（q/v 等长对齐）。"""
        n = 30
        t = np.arange(n, dtype=float)
        i = np.full(n, -1.5)
        v = np.linspace(4.1, 3.5, n)
        v[10] = np.nan
        out = block_cc_discharge_curve(self._df(t, i, v))
        assert out["n_pts"] == n - 1
        assert len(out["q"]) == len(out["v"])
        assert np.all(np.isfinite(out["v"]))

    def test_empty_no_discharge(self):
        """无放电段 → 空曲线。"""
        t = np.arange(10, dtype=float)
        i = np.full(10, 1.5)                   # 全充电
        v = np.full(10, 3.8)
        out = block_cc_discharge_curve(self._df(t, i, v))
        assert out["n_pts"] == 0
        assert out["q_end"] == 0.0

    def test_agrees_with_capacity(self):
        """q_end 与 block_cc_discharge_capacity 同口径。"""
        n = 60
        t = np.arange(n, dtype=float)
        i = np.full(n, -1.2)
        v = np.linspace(4.0, 3.3, n)
        df = self._df(t, i, v)
        out = block_cc_discharge_curve(df)
        assert out["q_end"] == pytest.approx(
            block_cc_discharge_capacity(df), rel=1e-9)


# ---------------------------------------------------------------------------
# segment_band_rates
# ---------------------------------------------------------------------------

class TestBandRates:
    def _traj(self, group, cell, efc, fade):
        return pd.DataFrame({
            "group": group, "cell": cell, "efc": efc, "fade_pct": fade,
        })

    def test_linear_uniform_bands(self):
        """线性 fade：各带速率一致。"""
        efc = np.arange(0.0, 401.0, 10.0)
        fade = 0.05 * efc                     # 5%/100 EFC → 20% @ 400
        traj = self._traj(1, 1, efc, fade)
        br = segment_band_rates(traj, bands=((0, 15), (15, 25)))
        r0 = br[(br.band_lo == 0)].rate_band.iloc[0]
        r1 = br[(br.band_lo == 15)].rate_band.iloc[0]
        assert r0 == pytest.approx(5.0, rel=1e-6)
        assert r1 == pytest.approx(5.0, rel=1e-6)

    def test_convex_deeper_faster(self):
        """凸轨迹：深带速率 > 浅带（加速）。"""
        efc = np.arange(0.0, 301.0, 10.0)
        fade = 0.02 * efc + 0.0005 * efc ** 2   # 末端 ≈ 51%
        traj = self._traj(5, 1, efc, fade)
        br = segment_band_rates(traj, bands=((0, 15), (15, 25), (25, 40),
                                             (40, 100)))
        r_early = br[(br.band_lo == 0)].rate_band.iloc[0]
        r_deep = br[(br.band_lo == 25)].rate_band.iloc[0]
        assert r_deep > r_early
        # 深带（fade 25–40 ≈ efc 205–264，带中点 ≈ 235）斜率 = b+2a·T
        assert r_deep == pytest.approx(25.5, rel=0.05)

    def test_thin_band_flagged(self):
        """带内点数 < min_pts → flag=1 且速率 NaN。"""
        efc = np.array([0.0, 50.0, 100.0])
        fade = np.array([1.0, 8.0, 30.0])
        traj = self._traj(5, 2, efc, fade)
        br = segment_band_rates(traj, bands=((0, 15), (15, 25), (25, 100)))
        thin = br[(br.band_lo == 15)].iloc[0]      # 只有 8% 一点不落带…
        # 15–25 带无点 → flag
        assert int(thin.flag) == 1 and np.isnan(thin.rate_band)
        deep = br[(br.band_lo == 25)].iloc[0]      # 30% 一点
        assert int(deep.flag) == 1

    def test_negative_fade_excluded(self):
        """负 fade 点（q_bol 噪声）不落带，不污染速率。"""
        efc = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
        fade = np.array([-2.0, 1.0, 4.0, 7.0, 10.0])
        traj = self._traj(1, 3, efc, fade)
        br = segment_band_rates(traj, bands=((0, 15),))
        r = br.rate_band.iloc[0]
        # 正 fade 4 点 (50,1)…(200,10) 线性斜率 = 9/150 → 6.0 %/100 EFC
        assert r == pytest.approx(6.0, rel=1e-6)


# ---------------------------------------------------------------------------
# mode_split_trajectory
# ---------------------------------------------------------------------------

class TestModeSplitTraj:
    @staticmethod
    def _ref_curve(q_end=3.0, n=150):
        """线性 V(q) 参考：v = 4.1 − 0.8·q/q_end。"""
        q = np.linspace(0.0, q_end, n)
        v = 4.1 - 0.8 * q / q_end
        return {"q": q, "v": v}

    @staticmethod
    def _shift_curve(delta=0.3, q_end=3.0, n=150):
        """纯 shift：V_k(q) = V_0(q+δ) → L(V)≡δ。"""
        q = np.linspace(0.0, q_end - delta, n)
        v = 4.1 - 0.8 * (q + delta) / q_end
        return {"q": q, "v": v}

    @staticmethod
    def _shape_curve(gamma=0.1, q_end=3.0, n=150):
        """纯 shape：q_k(V) = (1−γ)·q_0(V) → L(V)=γ·Q0(V)。"""
        q = np.linspace(0.0, (1.0 - gamma) * q_end, n)
        v = 4.1 - 0.8 * (q / (1.0 - gamma)) / q_end
        return {"q": q, "v": v}

    def test_pure_shift(self):
        """容量平移（LLI 型）→ share_shape ≈ 0、q_shift ≈ δ。"""
        df = mode_split_trajectory(
            [self._ref_curve(), self._shift_curve(0.3)], ref_idx=0)
        row = df.iloc[1]
        assert row.share_shape == pytest.approx(0.0, abs=0.05)
        assert row.q_shift == pytest.approx(0.3, rel=0.05)

    def test_pure_shape(self):
        """形变（LAM 型）→ share_shape ≈ 1。"""
        df = mode_split_trajectory(
            [self._ref_curve(), self._shape_curve(0.1)], ref_idx=0)
        row = df.iloc[1]
        assert row.share_shape == pytest.approx(1.0, abs=0.05)
        assert row.q_shape == pytest.approx(0.3, rel=0.1)

    def test_meta_passthrough(self):
        """元数据列透传。"""
        cur = self._shift_curve(0.2)
        cur.update({"group": 5, "cell": 5, "step": 10, "fade_pct": 20.0})
        df = mode_split_trajectory([self._ref_curve(), cur], ref_idx=0)
        assert df.iloc[1]["group"] == 5
        assert df.iloc[1]["step"] == 10
        assert df.iloc[1]["fade_pct"] == 20.0

    def test_empty_input(self):
        assert mode_split_trajectory([]).empty


# ---------------------------------------------------------------------------
# 真实数据集成（在机即跑）
# ---------------------------------------------------------------------------

class TestRealCurve:
    @pytest.mark.skipif(not (OXFORD / "Part2" / "Group_5.zip").exists(),
                        reason="Oxford Part2 Group5 不在机")
    def test_g5_block_curve_real(self):
        """G5 真实块：V(q) 提取与容量口径一致、物理合理。"""
        z = zipfile.ZipFile(OXFORD / "Part2" / "Group_5.zip")
        name = "Group 5/TPG5.1-Cell5.mat"
        from bmthermal.eis_features import load_maccor_block
        df = load_maccor_block(z.read(name))
        z.close()
        out = block_cc_discharge_curve(df)
        assert out["n_pts"] > 1000                 # C/25 RPT 细采样
        assert 2.5 < out["q_end"] < 3.5            # BoL ≈ 3.1 Ah
        assert out["i_med"] == pytest.approx(-0.125, abs=0.02)
        assert 2.4 < out["v"].min() < out["v"].max() < 4.3
        assert np.all(np.diff(out["q"]) >= -1e-9)

    @pytest.mark.skipif(not (OXFORD / "Group_1.zip").exists(),
                        reason="Oxford Part 1 不在机")
    def test_g1_g5_mode_split_direction(self):
        """G5 深段（fade≈57%）对 BoL 的 loss_eod ≈ q_end 差。"""
        from bmthermal.eis_features import load_maccor_block
        z5 = zipfile.ZipFile(OXFORD / "Part2" / "Group_5.zip")
        bol = block_cc_discharge_curve(
            load_maccor_block(z5.read("Group 5/TPG5.1-Cell5.mat")))
        eol = block_cc_discharge_curve(
            load_maccor_block(z5.read("Group 5/TPG5.11-Cell5.mat")))
        z5.close()
        df = mode_split_trajectory(
            [{**bol, "step": 1}, {**eol, "step": 11}], ref_idx=0)
        row = df.iloc[1]
        assert row.loss_eod == pytest.approx(bol["q_end"] - eol["q_end"],
                                             rel=0.02)
        assert row.q_shift + row.q_shape == pytest.approx(
            row.loss_eod, rel=0.15)                # qsum 闭合
