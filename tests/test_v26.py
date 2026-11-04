# -*- coding: utf-8 -*-
"""test_v26 — v1.15：脏数据清洗防线 / 静置窗 / matched-depth 速率。"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from bmthermal.eis_features import (
    DT_MAX_MIN, I_MAX_A, block_throughput_stats, rest_run_durations,
    _sanitize_mi,
)
from bmthermal.efc_rate import (
    early_rate_band, fade_at_efc, matched_depth_rates,
)
import pandas as pd

OXFORD = Path("D:/项目/02_数据集/01_国际公开集/Oxford_PathDependent")


# ---------------------------------------------------------------------------
# 清洗防线
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_sentinel_time_dropped(self):
        """哨兵 t（~1.3e15）剔除后积分不爆炸。"""
        t = np.array([0.0, 1.0, 1.3e15, 3.0, 4.0])
        i = np.array([-1.5, -1.5, -1.5, -1.5, -1.5])
        ts, is_ = _sanitize_mi(t, i)
        assert len(ts) == 4
        assert np.all(np.abs(ts) < 1e9)

    def test_garbage_current_dropped(self):
        """荒谬电流（>50 A）剔除。"""
        t = np.arange(6.0)
        i = np.array([1.5, 1.5, 999.0, 1.5, 1.5, 1.5])
        _, is_ = _sanitize_mi(t, i)
        assert len(is_) == 5
        assert np.all(np.abs(is_) <= I_MAX_A)

    def test_nan_dropped(self):
        t = np.array([0.0, np.nan, 2.0, 3.0])
        i = np.array([1.0, 1.0, np.nan, 1.0])
        ts, is_ = _sanitize_mi(t, i)
        assert len(ts) == 2

    def test_dt_clamped(self):
        """断点 dt 超上限按 0 计——吞吐不因时间跳变爆炸。"""
        # 合成块：正常 4 点 + 时间跳变 1e6 分钟 + 正常 2 点
        n = 7
        t = np.array([0.0, 1.0, 2.0, 3.0, 1.0e6, 1.0e6 + 1, 1.0e6 + 2])
        i = np.full(n, -1.5)
        ts, is_ = _sanitize_mi(t, i)
        dt = np.where(np.diff(ts) > DT_MAX_MIN, 0.0,
                      np.maximum(np.diff(ts), 0.0))
        q = float(np.sum(np.abs(0.5 * (is_[1:] + is_[:-1])) * dt) / 60.0)
        assert np.isfinite(q)
        assert q < 1.0     # 5 个 1-min 样本对×1.5 A/60 ≈ 0.125 Ah；跳变贡献 0


# ---------------------------------------------------------------------------
# 静置窗
# ---------------------------------------------------------------------------

class TestRestRuns:
    def test_basic_runs(self):
        """两段静置（5 min、10 min）被识别；通电段不计。"""
        # 1 s 采样：5 min 静置 + 5 min 放电 + 10 min 静置
        n1, n2, n3 = 300, 300, 600
        t = np.arange(n1 + n2 + n3, dtype=float)  # 秒
        i = np.concatenate([np.zeros(n1), np.full(n2, -1.5), np.zeros(n3)])
        durs = rest_run_durations(t, i / 60.0 * 0 + i,  # 同数组
                                  i_thresh=0.01, min_run_min=1.0)
        # 时长单位跟 t 走（这里 t 是秒——用分钟轴重测）
        t_min = t / 60.0
        durs = rest_run_durations(t_min, i, i_thresh=0.01, min_run_min=1.0)
        assert len(durs) == 2
        assert durs[0] == pytest.approx(5.0, rel=0.05)
        assert durs[1] == pytest.approx(10.0, rel=0.05)

    def test_min_run_filter(self):
        """单样本毛刺静置（<1 min）被滤除。"""
        t_min = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
        i = np.array([1.5, 0.0, 1.5, 1.5, 1.5])
        durs = rest_run_durations(t_min, i, min_run_min=1.0)
        assert len(durs) == 0

    def test_all_rest(self):
        t_min = np.arange(100.0)
        i = np.zeros(100)
        durs = rest_run_durations(t_min, i, min_run_min=1.0)
        assert len(durs) == 1
        assert durs[0] == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# matched-depth 速率
# ---------------------------------------------------------------------------

class TestMatchedDepth:
    def _traj(self, group, cell, efc, fade):
        return pd.DataFrame({
            "group": group, "cell": cell, "efc": efc, "fade_pct": fade,
        })

    def test_linear_early(self):
        """fade=0.05×efc 线性到 20%：rate_early=rate_to_cap=5 %/100 EFC。"""
        efc = np.arange(0.0, 401.0, 10.0)
        fade = 0.05 * efc          # 5%/100 EFC，末端 20%
        traj = self._traj(1, 1, efc, fade)
        md = matched_depth_rates(traj, fade_cap=15.0)
        r = md.iloc[0]
        assert r.rate_early == pytest.approx(5.0, rel=1e-6)
        assert r.efc_at_cap == pytest.approx(15.0 / 0.05, rel=1e-6)
        assert r.rate_to_cap == pytest.approx(5.0, rel=1e-6)

    def test_depth_mismatch_secant_vs_early(self):
        """凸轨迹：全深度割线 > early 窗 OLS——修正是方向对的。"""
        efc = np.arange(0.0, 201.0, 10.0)
        fade = 0.02 * efc + 0.0004 * efc ** 2   # 凸（加速），末端 20%
        traj = self._traj(5, 1, efc, fade)
        md = matched_depth_rates(traj, fade_cap=15.0)
        r = md.iloc[0]
        secant_full = fade[-1] / efc[-1] * 100
        assert secant_full > r.rate_early
        # fade≤15% 窗（efc≲170）OLS 斜率 ≈ b+a·T = 0.02+0.0004·170
        assert r.rate_early == pytest.approx(8.8, rel=0.08)

    def test_short_track_flagged(self):
        """fade 速达 cap（早段点 <3）→ 取前 3 点并 flag=1。"""
        efc = np.array([0.0, 50.0, 100.0])
        fade = np.array([5.0, 20.0, 30.0])   # fade≤15 只剩 1 点
        traj = self._traj(6, 1, efc, fade)
        md = matched_depth_rates(traj, fade_cap=15.0)
        r = md.iloc[0]
        assert int(r.flag) == 1
        assert np.isfinite(r.rate_early)

    def test_no_cap_no_efc_at_cap(self):
        """fade 从未达 15% → efc_at_cap NaN，rate_to_cap NaN。"""
        efc = np.arange(0.0, 101.0, 10.0)
        fade = np.full(11, 4.0)   # 恒 4%
        traj = self._traj(6, 2, efc, fade)
        md = matched_depth_rates(traj, fade_cap=15.0)
        r = md.iloc[0]
        assert np.isnan(r.efc_at_cap)
        assert np.isnan(r.rate_to_cap)


# ---------------------------------------------------------------------------
# 真实数据集成（在机即跑）
# ---------------------------------------------------------------------------

class TestRealBlocks:
    @pytest.mark.skipif(not (OXFORD / "Group_1.zip").exists(),
                        reason="Oxford Part 1 不在机")
    def test_rest_stats_g1_vs_g5(self):
        """G1（组合）延长静置剂量应高于 G5（连续循环）。"""
        z1 = zipfile.ZipFile(OXFORD / "Group_1.zip")
        z5 = zipfile.ZipFile(OXFORD / "Part2" / "Group_5.zip")
        name1 = next(n for n in z1.namelist()
                     if n.endswith(".mat") and "TPG1." in n)
        name5 = next(n for n in z5.namelist()
                     if n.endswith(".mat") and "TPG5." in n)
        s1 = block_throughput_stats(z1.read(name1))
        s5 = block_throughput_stats(z5.read(name5))
        for st in (s1, s5):
            assert np.isfinite(st["t_min"])
            assert abs(st["t_min"]) < 1e6          # 无哨兵
            assert st["q_cc"] < 5.0                # 容量物理合理
        assert s1["ext_rest_min"] >= 0 and s5["ext_rest_min"] >= 0
        z1.close()
        z5.close()


class TestAnchors:
    def _traj(self, group, cell, efc, fade):
        return pd.DataFrame({
            "group": group, "cell": cell, "efc": efc, "fade_pct": fade,
        })

    def test_fade_at_efc_interp(self):
        efc = np.array([0.0, 100.0, 200.0, 300.0])
        fade = np.array([0.0, 4.0, 8.0, 12.0])
        traj = self._traj(1, 1, efc, fade)
        f = fade_at_efc(traj, 150.0)
        assert f.iloc[0].fade_at_efc == pytest.approx(6.0)
        # 未达目标里程 → NaN
        f2 = fade_at_efc(traj, 500.0)
        assert np.isnan(f2.iloc[0].fade_at_efc)

    def test_early_rate_band_skip(self):
        """BoL 噪声点使 skip 档变化——带存在且有限。"""
        efc = np.array([0.0, 10.0, 100.0, 200.0, 300.0])
        fade = np.array([5.0, 0.0, 2.0, 6.0, 10.0])   # 首点噪声 5%
        traj = self._traj(1, 1, efc, fade)
        band = early_rate_band(traj, fade_cap=15.0, max_skip=2)
        med = band["group_med_band"]["g1"]["med_by_skip"]
        assert all(np.isfinite(med))
        lo, hi = band["group_med_band"]["g1"]["band"]
        assert lo <= hi
