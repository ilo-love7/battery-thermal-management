"""test_v25 — v1.14 EFC/吞吐轴与同里程速率（scripts/52）。"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bmthermal.eis_features import (
    block_cc_discharge_capacity, block_cc_discharge_segment,
    block_throughput_stats,
)
from bmthermal.efc_rate import build_trajectories, group_rates
from bmthermal.guide_map import is_rpt, load_guide_rows

ROOT = Path(__file__).resolve().parents[1]
OXFORD = Path("D:/项目/02_数据集/01_国际公开集/Oxford_PathDependent")


# ---------------------------------------------------------------------------
# block_cc_discharge_segment：定位 + 单位
# ---------------------------------------------------------------------------

def _seg_df(t, i):
    return pd.DataFrame({
        "TestTime": np.asarray(t, dtype=float),
        "Amps": np.asarray(i, dtype=float),
        "Volts": np.full(len(t), 3.8),
        "Amphr": np.zeros(len(t)),
        "Step": np.zeros(len(t)),
    })


class TestCCSegment:
    def test_matches_capacity_function(self):
        # 2 h @ −1.5 A（C/2 放 3 Ah）+ 静置 + 充电段
        t = np.concatenate([np.arange(0, 121), np.arange(121, 181),
                            np.arange(181, 301)])
        i = np.concatenate([np.full(121, -1.5), np.zeros(60),
                            np.full(120, 1.0)])
        df = _seg_df(t, i)
        seg = block_cc_discharge_segment(df)
        assert seg["q"] == pytest.approx(
            block_cc_discharge_capacity(df), rel=1e-9)
        assert seg["q"] == pytest.approx(3.0, rel=0.01)

    def test_end_index_and_cum_dis(self):
        t = np.arange(0, 121, dtype=float)      # 分钟制
        i = np.full(121, -1.5)
        seg = block_cc_discharge_segment(_seg_df(t, i))
        assert seg["end_idx"] == 120
        assert seg["t_end"] == pytest.approx(120.0)
        # ∫1.5A·120min/60 = 3 Ah（放电口径）
        assert seg["ah_dis_at_end"] == pytest.approx(3.0, rel=0.01)
        assert seg["ah_tp_at_end"] == pytest.approx(3.0, rel=0.01)

    def test_tiny_or_no_discharge(self):
        seg = block_cc_discharge_segment(_seg_df([0, 1, 2], [0, 0, 0]))
        assert seg["q"] == 0.0
        assert seg["end_idx"] == -1
        assert seg["ah_dis_at_end"] == 0.0


# ---------------------------------------------------------------------------
# guide_map：合成 xlsx 解析
# ---------------------------------------------------------------------------

class TestGuideMap:
    def test_synthetic_guide(self, tmp_path):
        rows = [
            ["Group 3 - test", "", ""],
            ["Cell", "Procedure", "RPT/Profile"],
            ["7", "TPG3", "BoL RPT"],
            ["", "TPG3.2", "Profile"],
            ["", "TPG3.3", "RPT1"],
            ["8", "TPG3", "BoL RPT"],
        ]
        p = tmp_path / "g.xlsx"
        pd.DataFrame(rows).to_excel(p, header=False, index=False)
        df = load_guide_rows([(p, 1, 4)])
        assert len(df) == 4
        c7 = df[df.cell == 7]
        assert list(c7.step) == [1, 2, 3]
        # 单元格号前向填充
        assert list(df[df.cell == 8].label) == ["BoL RPT"]
        assert is_rpt("Profile + RPT1-RPT4")
        assert not is_rpt("Profile")
        assert not is_rpt("EoL")

    @pytest.mark.skipif(not (OXFORD / "Part2" / "Guide_to_Datafiles_2.xlsx").exists(),
                        reason="Part 2 Guide 不在机")
    def test_real_guide_counts(self):
        df = load_guide_rows([(OXFORD / "Part2" / "Guide_to_Datafiles_2.xlsx", 1, 6)])
        assert len(df) == 187
        assert int(df.label.map(is_rpt).sum()) == 107
        assert set(int(x) for x in df.group.unique()) == {1, 2, 3, 4, 5, 6}
        # 三 Guide 合计与 50 的日志一致（603 块 / RPT 326）
        specs = [
            (OXFORD / "Guide_to_Datafiles.xlsx", 1, 4),
            (OXFORD / "Part2" / "Guide_to_Datafiles_2.xlsx", 1, 6),
            (OXFORD / "Part3" / "Guide_to_Datafiles_3.xlsx", 7, 10),
        ]
        if all(p.exists() for p, _, _ in specs):
            al = load_guide_rows(specs)
            assert len(al) == 603
            assert int(al.label.map(is_rpt).sum()) == 326


# ---------------------------------------------------------------------------
# efc_rate：合成轨迹（EFC/日历轴、速率）
# ---------------------------------------------------------------------------

def _tp_row(g, cell, step, label, ah_dis, t_min, q_cc,
            ah_dis_at_cc=None, t_cc_end=None):
    return {
        "group": g, "cell": cell, "step": step, "label": label,
        "ah_dis": ah_dis, "ah_tp": ah_dis * 2, "t_min": t_min,
        "q_cc": q_cc,
        "ah_dis_at_cc": ah_dis if ah_dis_at_cc is None else ah_dis_at_cc,
        "ah_tp_at_cc": (ah_dis if ah_dis_at_cc is None else ah_dis_at_cc) * 2,
        "t_cc_end_min": t_min if t_cc_end is None else t_cc_end,
        "n_rows": 10, "ah_chg": 0.0, "ah_amphr": ah_dis * 2,
    }


class TestTrajectories:
    def test_efc_axis_and_rates(self):
        # q_bol=3.0；三块各放电 3.0 Ah（1 EFC/块），q 点 3.0→2.7→2.4
        tp = pd.DataFrame([
            _tp_row(5, 1, 1, "BoL RPT", 3.0, 1440.0, 3.0),
            _tp_row(5, 1, 2, "Profile + RPT1", 3.0, 1440.0, 2.7),
            _tp_row(5, 1, 3, "Profile + RPT2", 3.0, 1440.0, 2.4),
        ])
        traj, rates = build_trajectories(tp)
        assert len(traj) == 3
        # q 点在块内段结束 → EFC 含块内已放电：0.5, 1.5, 2.5
        assert list(np.round(traj.efc, 3)) == [1.0, 2.0, 3.0]
        assert list(np.round(traj.fade_pct, 2)) == [0.0, 10.0, 20.0]
        r = rates.iloc[0]
        assert r.efc_total == pytest.approx(3.0)
        assert r.days_total == pytest.approx(3.0)       # 3×1440 min
        assert r.fade_eol_pct == pytest.approx(20.0)
        # 割线 %/100 EFC = fade/efc·100 = 666.7；OLS 斜率 10 %/EFC → 1000
        assert r.rate_efc_secant == pytest.approx(20.0 / 3.0 * 100, rel=1e-6)
        assert r.rate_efc_ols == pytest.approx(1000.0, rel=1e-6)
        assert r.rate_day_secant == pytest.approx(20.0 / 3.0 * 100, rel=1e-6)

    def test_bol_reference_and_threshold(self):
        # q_bol 取 BoL 标签最大；q<Q_MIN 的点被滤掉
        tp = pd.DataFrame([
            _tp_row(5, 1, 1, "BoL RPT", 3.0, 100.0, 3.1),
            _tp_row(5, 1, 2, "BoL RPT", 3.0, 100.0, 2.9),
            _tp_row(5, 1, 3, "Profile + RPT1", 3.0, 100.0, 2.8),
            _tp_row(5, 1, 4, "Profile + RPT2", 0.1, 10.0, 0.1),  # 碎段
            _tp_row(5, 1, 5, "Profile + RPT3", 3.0, 100.0, 2.7),
        ])
        traj, rates = build_trajectories(tp)
        assert len(traj) == 4          # 0.1 Ah 点被滤
        assert rates.iloc[0].q_bol == pytest.approx(3.1)
        # fade 用 3.1 为参考
        assert traj[traj.step == 5].fade_pct.iloc[0] == \
            pytest.approx((1 - 2.7 / 3.1) * 100, rel=1e-6)

    def test_short_series_skipped(self):
        tp = pd.DataFrame([
            _tp_row(5, 1, 1, "BoL RPT", 3.0, 100.0, 3.0),
            _tp_row(5, 1, 2, "RPT1", 3.0, 100.0, 2.9),
        ])
        traj, rates = build_trajectories(tp)
        assert traj.empty and rates.empty

    def test_group_rates_medians(self):
        rates = pd.DataFrame([
            {"group": 5, "cell": 1, "efc_total": 100.0, "days_total": 50.0,
             "fade_eol_pct": 60.0, "rate_efc_secant": 60.0,
             "rate_efc_ols": 55.0, "rate_day_secant": 120.0},
            {"group": 5, "cell": 2, "efc_total": 120.0, "days_total": 60.0,
             "fade_eol_pct": 50.0, "rate_efc_secant": 40.0,
             "rate_efc_ols": 45.0, "rate_day_secant": 80.0},
        ])
        gr = group_rates(rates, {5: {"kind": "cycle_only"}})
        assert gr["g5"]["n"] == 2
        assert gr["g5"]["efc_total_med"] == pytest.approx(110.0)
        assert gr["g5"]["rate_efc_secant_med"] == pytest.approx(50.0)
        assert gr["g5"]["kind"] == "cycle_only"


# ---------------------------------------------------------------------------
# 真实数据集成（在机即跑）
# ---------------------------------------------------------------------------

G5_ZIP = OXFORD / "Part2" / "Group_5.zip"


@pytest.mark.skipif(not G5_ZIP.exists(), reason="Group_5.zip 不在机")
class TestRealThroughput:
    def test_g5_block_dual_estimator(self):
        z = zipfile.ZipFile(G5_ZIP)
        names = [n for n in z.namelist()
                 if n.endswith("TPG5.2-Cell5.mat")]
        assert names, "G5 BoL 块不在 zip"
        st = block_throughput_stats(z.read(names[0]))
        assert st["n_rows"] > 1000
        # 双估计器：∫|I|dt vs 分步 Amphr 极差和
        rel = abs(st["ah_tp"] - st["ah_amphr"]) / max(st["ah_tp"], 1e-9)
        assert rel < 0.05
        # q_cc 在物理范围（NCA 18650 BoL ≈ 3 Ah）
        assert 2.0 < st["q_cc"] < 3.5
        # TestTime 分钟制：块时长 = t_min/1440 天
        assert st["t_min"] / 1440.0 < 30          # BoL 块不该几十天
