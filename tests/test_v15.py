"""v1.5 深化测试：snl_aging / panasonic_rct 护栏。

重点覆盖 v1.5 踩坑里长出来的性质：
- SNL 文件名解析（温度/SOC 窗/倍率）与 20–80 窗混组防线；
- 逐循环解析的重复导出去重与 fade 窗口径；
- 两组闭式解在 t_ratio 倒挂时拒绝、s_raw 未截断值保留；
- HPPC 温度解析（n10→−10）与脉冲 R 提取；
- R(T) 拟合约定（Ea 往返）与单调性护栏；
- 欧姆产热指数 p=2+dlnR/dlnC 往返。
"""
from __future__ import annotations

import numpy as np
import pytest

from bmthermal.snl_aging import (
    fade_window_rate,
    group_stats,
    pair_share,
    parse_name,
    summarize_cycles,
)
from bmthermal.panasonic_rct import (
    extract_pulses,
    fit_r_int_t,
    parse_temp,
    rate_nonlinearity,
)


# ---------------------------------------------------------------- SNL 解析

class TestSnlParsing:
    def test_parse_name_full(self):
        d = parse_name("SNL_18650_NMC_25C_0-100_0.5-3C_b.pkl")
        assert d["chem"] == "NMC"
        assert d["temp_c"] == 25.0
        assert (d["soc_lo"], d["soc_hi"]) == (0.0, 100.0)
        assert (d["c_chg"], d["c_dis"]) == (0.5, 3.0)
        assert d["rep"] == "b"

    def test_parse_name_soc_window(self):
        d = parse_name("SNL_18650_NCA_25C_20-80_0.5-0.5C_a.pkl")
        assert (d["soc_lo"], d["soc_hi"]) == (20.0, 80.0)

    def test_parse_name_rejects_garbage(self):
        with pytest.raises(ValueError):
            parse_name("not_a_snl_file.pkl")

    def test_fade_window_linear(self):
        n = np.arange(1, 201, dtype=float)
        f = 2e-4 * n  # 斜率 2e-4
        assert abs(fade_window_rate(n, f) - 2e-4) < 1e-9

    def test_fade_window_ignores_out_of_window(self):
        n = np.arange(1, 201, dtype=float)
        f = 2e-4 * n
        f[:2] = 0.005  # 1% 窗之外的首循环离群
        assert abs(fade_window_rate(n, f) - 2e-4) < 1e-9

    def _fake_cycles(self, n=40, dup=False):
        cycles = []
        for k in range(1, n + 1):
            c = {
                "cycle_number": k,
                "discharge_capacity_in_Ah": [0.0, 3.0 - 0.001 * k],
                "time_in_s": [0.0, 10000.0],
            }
            cycles.append(c)
            if dup:
                cycles.append(dict(c))  # 连续重复导出
        return cycles

    def test_summarize_dedupes_duplicates(self):
        s1 = summarize_cycles(self._fake_cycles(40, dup=False), nominal_ah=3.0)
        s2 = summarize_cycles(self._fake_cycles(40, dup=True), nominal_ah=3.0)
        assert s2["n_cycles"] == s1["n_cycles"]  # 重复条目被去掉
        assert abs(s2["fade_rate_per_efc"] - s1["fade_rate_per_efc"]) < 1e-12

    def test_summarize_drops_partial_entries(self):
        cycles = self._fake_cycles(30)
        cycles.append({
            "cycle_number": 31,
            "discharge_capacity_in_Ah": [0.0, 0.5],   # 远低于阈值
            "time_in_s": [0.0, 500.0],
        })
        s = summarize_cycles(cycles, nominal_ah=3.0)
        assert s["n_cycles"] == 30


class TestSnlGroups:
    def _df(self):
        import pandas as pd
        rows = []
        for chem, soc_lo, cdis, fade, dt, rep_n in (
            ("NMC", 0, 1.0, 1.7e-4, 10000.0, 3),
            ("NMC", 0, 3.0, 1.5e-4, 8000.0, 3),
            ("NMC", 20, 1.0, 1.0e-4, 7000.0, 2),   # 20–80 窗干扰项
        ):
            for i in range(rep_n):
                rows.append({
                    "cell_id": f"{chem}_{soc_lo}_{cdis}_{i}",
                    "chem": chem, "temp_c": 25.0,
                    "soc_lo": soc_lo, "soc_hi": 100.0 if soc_lo == 0 else 80.0,
                    "c_chg": 0.5, "c_dis": cdis, "rep": chr(97 + i),
                    "n_cycles": 500, "cap0_ah": 3.0,
                    "fade_rate_per_efc": fade, "dt_s_per_cycle": dt,
                    "fade_final": 0.1,
                })
        return pd.DataFrame(rows)

    def test_group_stats_excludes_other_soc_window(self):
        df = self._df()
        g = group_stats(df, "NMC", 25.0, 1.0)          # 只取 0–100 窗
        assert g["n_cells"] == 3
        assert abs(g["fade_per_efc"] - 1.7e-4) < 1e-12  # 20–80 窗不混入

    def test_pair_share_basic(self):
        df = self._df()
        p = pair_share(df, "NMC", 25.0, 1.0, 3.0)
        assert p["t_ratio"] == pytest.approx(10000 / 8000)
        assert p["fade_ratio"] == pytest.approx(1.7e-4 / 1.5e-4)
        # s_raw = (r−1)/(t−1)，正值且 <1
        assert 0 < p["s_raw"] < 1
        assert p["s_cal_at_fast"] == pytest.approx(p["s_raw"])
        assert p["eligible"] is True

    def test_pair_share_rejects_inverted_t(self):
        import pandas as pd
        df = self._df()
        # 构造 t_ratio<1：慢组时长反而短
        df.loc[df.c_dis == 1.0, "dt_s_per_cycle"] = 7000.0
        p = pair_share(df, "NMC", 25.0, 1.0, 3.0)
        assert p["t_ratio"] < 1.0
        assert np.isnan(p["s_cal_at_fast"])   # 资格否决
        assert p["eligible"] is False

    def test_pair_share_negative_s_raw_kept(self):
        import pandas as pd
        df = self._df()
        # 慢组 fade 反而小 → s_raw<0
        df.loc[df.c_dis == 1.0, "fade_rate_per_efc"] = 1.0e-4
        p = pair_share(df, "NMC", 25.0, 1.0, 3.0)
        assert p["s_raw"] < 0
        assert p["s_cal_at_fast"] == 0.0     # 截断
        assert p["eligible"] is False        # 负 s_raw 不算资格对


# ------------------------------------------------------------ Panasonic

class TestPanasonic:
    def test_parse_temp_positive(self):
        assert parse_temp("03-11-17_08.47 25degC_5Pulse_HPPC_Pan18650PF.mat") == 25.0

    def test_parse_temp_negative(self):
        assert parse_temp("06-07-17_12.29 n10degC_5pulse_HPPC_Pan18650PF.mat") == -10.0
        assert parse_temp("06-15-17_11.31 n20degC_5Pulse_HPPC_Pan18650PF.mat") == -20.0

    def _fake_hppc(self, r_ohm=0.04):
        """合成 HPPC：静置 100 s + 10 s 脉冲，R 精确已知。"""
        rows = []
        t = 0.0
        for level in (-1.45, -2.9, -5.8):
            for _ in range(100):            # 静置
                rows.append((t, 0.0, 4.0, -0.05, 25.0))
                t += 1.0
            v_pulse = 4.0 + r_ohm * level   # 欧姆降（放电负 → 电压降）
            for _ in range(10):
                rows.append((t, level, v_pulse, -0.05 - abs(level) * 10 / 3600, 25.0))
                t += 1.0
        import pandas as pd
        return pd.DataFrame(rows, columns=["t", "i", "v", "ah", "t_batt"])

    def test_extract_pulses_r(self):
        df = self._fake_hppc(r_ohm=0.04)
        p = extract_pulses(df, temp_c=25.0)
        assert len(p) == 3
        # r_end 应还原 0.04 Ω（合成数据无极化）
        assert np.allclose(p["r_end"], 0.04, rtol=0.05)
        assert np.allclose(p["r_inst"], 0.04, rtol=0.05)

    def test_fit_r_int_t_roundtrip(self):
        import pandas as pd
        ea_true = 20000.0  # J/mol
        r_gas = 8.314462618
        rows = []
        for tc in (-20.0, -10.0, 0.0, 10.0, 25.0):
            g = np.exp(ea_true / r_gas * (1.0 / (tc + 273.15) - 1.0 / 298.15))
            r = 0.03 * g
            for _ in range(3):
                rows.append({
                    "temp_c": tc, "i_level": -2.9, "c_rate": 2.0, "soc": 0.5,
                    "v_pre": 4.0, "r_inst": r, "r_end": r,
                    "t_mid": 0.0, "t_batt": tc, "dur_s": 10.0,
                })
        pulses = pd.DataFrame(rows)
        fit = fit_r_int_t(pulses, soc_ref=0.5, c_ref=2.0, metric="r_end")
        assert fit.monotonic is True
        assert fit.ea_j_mol == pytest.approx(ea_true, rel=0.02)
        assert fit.ratio(-10.0) == pytest.approx(
            np.exp(ea_true / r_gas * (1.0 / 263.15 - 1.0 / 298.15)), rel=0.02)

    def test_fit_r_int_t_rejects_flat(self):
        import pandas as pd
        rows = [
            {"temp_c": tc, "i_level": -2.9, "c_rate": 2.0, "soc": 0.5,
             "v_pre": 4.0, "r_inst": 0.03, "r_end": 0.03,
             "t_mid": 0.0, "t_batt": tc, "dur_s": 10.0}
            for tc in (-10.0, 0.0, 10.0, 25.0) for _ in range(3)
        ]
        pulses = pd.DataFrame(rows)
        fit = fit_r_int_t(pulses, soc_ref=0.5, c_ref=2.0, metric="r_end")
        # R 不随 T 变 → 斜率 0、r² 不可辨；不抛但 Ea≈0
        assert abs(fit.ea_j_mol) < 1.0

    def test_rate_nonlinearity_groups(self):
        import pandas as pd
        rows = []
        for c, r in ((0.5, 0.036), (1.0, 0.036), (2.0, 0.036)):
            for _ in range(2):
                rows.append({
                    "temp_c": 25.0, "i_level": -2.9, "c_rate": c, "soc": 0.5,
                    "v_pre": 4.0, "r_inst": r, "r_end": r,
                    "t_mid": 0.0, "t_batt": 25.0, "dur_s": 10.0,
                })
        pulses = pd.DataFrame(rows)
        nl = rate_nonlinearity(pulses, 25.0, soc_ref=0.5)
        assert len(nl) == 3
        # 平坦 R → Q 指数 2
        slope = np.polyfit(np.log(nl.c_rate), np.log(nl.r_end), 1)[0]
        assert 2.0 + slope == pytest.approx(2.0, abs=1e-6)


class TestHighrateExponent:
    def test_q_exp_from_slope(self):
        # R ∝ C^-0.2 → p = 1.8（scripts/28 的核心换算）
        c = np.array([0.5, 1.0, 2.0, 4.0, 6.0])
        r = 0.1 * c ** (-0.2)
        slope = np.polyfit(np.log(c), np.log(r), 1)[0]
        assert 2.0 + slope == pytest.approx(1.8, abs=1e-6)
