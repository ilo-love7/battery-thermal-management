"""v1.13 测试：三阶段配对 / 三通道分区 / EoL 正则 / 硬件实验管线。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bmthermal.eis_features import (
    EIS_N_POINTS, NyquistFeatures, chain_share, nyquist_features,
    parse_eis_member, stage_deltas,
)
from bmthermal.hwexp_pipeline import (
    AGING_SCHEMA, POSTMORTEM_SCHEMA, SIGMA_Q_REQ_AH,
    CellFade, analyze_postmortem, compare_arms, e1_end_to_end,
    e2_end_to_end, fit_arm, fit_cell_fade, sigma_q_check,
    synth_aging, synth_postmortem, validate_schema,
)


def _synth_spectrum(hook: float = -0.01, arc: float = 0.004,
                    tail: float = 0.006, r0: float = 0.045):
    """合成谱：钩 + 弧 + 尾（同 v1.12 夹具）。"""
    n = EIS_N_POINTS
    idx = np.arange(n, dtype=float)
    z_re = r0 + 0.0004 * idx + 0.012 * (idx / n) ** 3
    z_im = np.empty(n)
    hp = 5
    z_im[:hp] = hook * np.linspace(1.0, 0.0, hp)
    ap = 15
    ax = np.linspace(0, 1, ap)
    z_im[hp:hp + ap] = arc * np.sin(np.pi * ax) ** 2
    ts = hp + ap
    z_im[ts:] = np.linspace(0.001, tail, n - ts)
    return z_re, z_im


class TestEoLParsing:
    def test_eol_member(self):
        m = parse_eis_member("EIS/EoL/Group1_EoL_Cell15_SOC20.mat")
        assert m == {"stage": "EoL", "group": 1, "cell": 15, "soc": 20}

    def test_bol_mol_still_parse(self):
        assert parse_eis_member(
            "EIS/BoL/Group2_BoL_Cell3_SOC50.mat")["stage"] == "BoL"
        assert parse_eis_member(
            "EIS/MoL/Group3_MoL_Cell10_SOC80.mat")["stage"] == "MoL"

    def test_part3_group10(self):
        m = parse_eis_member("EIS/EoL/Group10_EoL_Cell22_SOC20.mat")
        assert m["group"] == 10 and m["stage"] == "EoL"

    def test_garbage_rejected(self):
        assert parse_eis_member("EIS/EoL/Garbage.mat") is None


class TestStageDeltas:
    def test_deltas_and_chain_additivity(self):
        z1 = _synth_spectrum()
        z2 = _synth_spectrum(arc=0.005, r0=0.047)
        z3 = _synth_spectrum(arc=0.007, r0=0.050)
        f1, f2, f3 = (nyquist_features(*z) for z in (z1, z2, z3))
        d12 = stage_deltas(f1, f2)
        d23 = stage_deltas(f2, f3)
        d13 = stage_deltas(f1, f3)
        for k in d13:
            assert d13[k] == pytest.approx(d12[k] + d23[k], abs=1e-12)

    def test_chain_share(self):
        s = chain_share({"d_arc_dre": 0.02}, {"d_arc_dre": 0.03})
        assert s == pytest.approx(0.6, rel=1e-9)

    def test_chain_share_nonpositive_denom(self):
        assert np.isnan(chain_share({"d_arc_dre": -0.02},
                                    {"d_arc_dre": -0.01}))

    def test_pair_deltas_compatible(self):
        from bmthermal.eis_features import pair_deltas
        f1 = nyquist_features(*_synth_spectrum())
        f2 = nyquist_features(*_synth_spectrum(arc=0.006))
        p = pair_deltas(f1, f2, group=1, cell=2, soc=50)
        assert p.deltas["d_arc_height"] == pytest.approx(
            stage_deltas(f1, f2)["d_arc_height"], abs=1e-15)


class TestThreeChannelPartition:
    def _feat(self, r_ohm, r_pol, arc_dre):
        return {"d_r_ohm": r_ohm, "d_r_pol": r_pol, "d_arc_dre": arc_dre}

    def test_tail_dominant_partition(self):
        """组 5 形态：r_pol 爆炸住尾段——两通道口径会误判，三通道不。"""
        # 用与 scripts/49 同公式的轻量复算
        rows = [self._feat(0.067, 0.527, 0.006),
                self._feat(0.060, 0.500, 0.005)]
        df = pd.DataFrame(rows)
        tail = (df.d_r_pol - df.d_arc_dre).clip(lower=0)
        ohm = df.d_r_ohm.clip(lower=0)
        arc = df.d_arc_dre.clip(lower=0)
        total = float(ohm.sum() + arc.sum() + tail.sum())
        share_tail = float(tail.sum() / total)
        assert share_tail > 0.8                 # 尾段主导
        share_arc = float(arc.sum() / total)
        assert share_arc < 0.05                 # 弧份额极小

    def test_arc_dominant_partition(self):
        rows = [self._feat(0.008, 0.067, 0.064)]
        df = pd.DataFrame(rows)
        tail = (df.d_r_pol - df.d_arc_dre).clip(lower=0)
        total = float(df.d_r_ohm.sum() + df.d_arc_dre.sum() + tail.sum())
        assert float(df.d_arc_dre.sum() / total) > 0.7


class TestSchemaValidation:
    def _good_aging(self):
        return pd.DataFrame([
            {"cell_id": "c1", "chem": "LFP", "c_dis": 1.0, "temp_c": 25.0,
             "efc": 0.0, "q_rpt_ah": 1.03},
            {"cell_id": "c1", "chem": "LFP", "c_dis": 1.0, "temp_c": 25.0,
             "efc": 100.0, "q_rpt_ah": 1.02},
            {"cell_id": "c1", "chem": "LFP", "c_dis": 1.0, "temp_c": 25.0,
             "efc": 200.0, "q_rpt_ah": 1.01},
        ])

    def test_good_passes(self):
        out = validate_schema(self._good_aging(), AGING_SCHEMA, "g")
        assert len(out) == 3

    def test_missing_column(self):
        df = self._good_aging().drop(columns=["q_rpt_ah"])
        with pytest.raises(ValueError, match="missing columns"):
            validate_schema(df, AGING_SCHEMA, "g")

    def test_bad_dtype(self):
        df = self._good_aging()
        df["efc"] = "abc"
        with pytest.raises(ValueError):
            validate_schema(df, AGING_SCHEMA, "g")

    def test_nan_rejected(self):
        df = self._good_aging()
        df.loc[0, "q_rpt_ah"] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            validate_schema(df, AGING_SCHEMA, "g")


class TestSigmaGate:
    def test_budget_boundary(self):
        assert sigma_q_check(SIGMA_Q_REQ_AH)["adequate"] is True
        assert sigma_q_check(SIGMA_Q_REQ_AH * 1.01)["adequate"] is False

    def test_0p2pct_meter_fails(self):
        g = sigma_q_check(0.002 * 1.039)
        assert g["adequate"] is False
        assert g["sigma_q_meas_ah"] > 3 * SIGMA_Q_REQ_AH

    def test_nonpositive_rejected(self):
        with pytest.raises(ValueError):
            sigma_q_check(0.0)


class TestFadeFit:
    def _df(self, rate=2e-5, noise=0.0, n=6, cid="c1"):
        efcs = np.linspace(0, 1200, n)
        q0 = 1.039
        return pd.DataFrame([
            {"cell_id": cid, "chem": "LFP", "c_dis": 1.0, "temp_c": 25.0,
             "efc": float(e), "q_rpt_ah": q0 * (1 - rate * e + noise)}
            for e in efcs])

    def test_linear_fade_rate_recovered(self):
        f = fit_cell_fade(self._df(), "c1")
        assert f.rate_per_efc == pytest.approx(2e-5, rel=1e-6)
        assert f.n_obs == 6

    def test_too_few_points(self):
        with pytest.raises(ValueError, match="≥3"):
            fit_cell_fade(self._df(n=2), "c1")

    def test_nonpositive_q(self):
        df = self._df()
        df.loc[0, "q_rpt_ah"] = -0.5
        with pytest.raises(ValueError, match="non-positive"):
            fit_cell_fade(df, "c1")


class TestDecisionWiring:
    def test_m_ratio_thresholds(self):
        """5 芯/臂紧散布、m=2.0 → CI 全 < 2.35 → throughput_positive*。

        （n=1/臂时散布代理 20% 会让 CI 跨阈值 → inconclusive——
        统计诚实行为，见 test_single_cell_inconclusive。）
        """
        ref = [CellFade(f"r{k}", "LFP", 1.0, 25.0, 2.5e-5, 6, 1.039)
               for k in range(5)]
        tgt = [CellFade(f"t{k}", "LFP", 5.0, 25.0, 5.0e-5, 6, 1.039)
               for k in range(5)]
        v = compare_arms(ref, tgt)
        assert v.m_est == pytest.approx(2.0, rel=1e-9)
        assert v.decision.startswith("throughput_positive")

    def test_single_cell_inconclusive(self):
        """n=1/臂 → 散布代理 20% → CI 宽 → 不可判（如实）。"""
        ref = [CellFade("r", "LFP", 1.0, 25.0, 2.5e-5, 6, 1.039)]
        tgt = [CellFade("t", "LFP", 5.0, 25.0, 5.0e-5, 6, 1.039)]
        v = compare_arms(ref, tgt)
        assert v.decision == "inconclusive"
        assert v.ci_low < 2.35 < v.ci_high

    def test_ci_straddling_star(self):
        ref = [CellFade("r1", "LFP", 1.0, 25.0, 2.5e-5, 6, 1.039),
               CellFade("r2", "LFP", 1.0, 25.0, 3.5e-5, 6, 1.039),
               CellFade("r3", "LFP", 1.0, 25.0, 1.8e-5, 6, 1.039),
               CellFade("r4", "LFP", 1.0, 25.0, 3.0e-5, 6, 1.039)]
        tgt = [CellFade("t1", "LFP", 5.0, 25.0, 7.8e-5, 6, 1.039),
               CellFade("t2", "LFP", 5.0, 25.0, 6.2e-5, 6, 1.039),
               CellFade("t3", "LFP", 5.0, 25.0, 9.1e-5, 6, 1.039),
               CellFade("t4", "LFP", 5.0, 25.0, 8.3e-5, 6, 1.039)]
        v = compare_arms(ref, tgt)     # m̂≈2.6，散布大 → 可能跨阈值
        assert v.ci_low < v.m_est < v.ci_high
        assert v.decision in ("inconclusive", "throughput_positive",
                              "throughput_negative",
                              "throughput_positive_at_lower_bound",
                              "throughput_negative_at_upper_bound")

    def test_postmortem_directions(self):
        for lam35, lam25, c35, c25, expect in [
                (0.35, 0.20, 0.40, 0.22, "reading_b_confirmed"),
                (0.20, 0.35, 0.22, 0.40, "reading_a_gains_weight"),
                (0.35, 0.20, 0.22, 0.40, "mixed_evidence")]:
            df = synth_postmortem("LFP", 1.0, lam35, lam25, c35, c25,
                                  sd=1e-6, seed=1)
            v = analyze_postmortem(df)
            assert v.decision == expect

    def test_postmortem_unpaired_raises(self):
        df = synth_postmortem("LFP", 1.0, 0.3, 0.2, 0.3, 0.2, sd=1e-6)
        df = df[df.temp_c == 35.0]
        with pytest.raises(ValueError):
            analyze_postmortem(df)


class TestEndToEndSmoke:
    def test_e1_smoke(self):
        r = e1_end_to_end(seed=0, m_true=2.3)
        assert 1.5 < r["verdict"]["m_est"] < 3.5
        assert isinstance(r["correct"], bool)

    def test_e2_smoke(self):
        assert e2_end_to_end(seed=0, direction="b")["correct"] is True
        assert e2_end_to_end(seed=0, direction="a")["correct"] is True

    def test_synth_aging_schema(self):
        df = synth_aging([{"chem": "LFP", "c_dis": 1.0, "temp_c": 25.0,
                           "rate_per_efc": 2e-5}], n_cells=2, sd_rate=1e-6)
        out = validate_schema(df, AGING_SCHEMA, "synth")
        assert set(out.cell_id.unique()) == {"LFP_1.0C_25.0C_0",
                                             "LFP_1.0C_25.0C_1"}
