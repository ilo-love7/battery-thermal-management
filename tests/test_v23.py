"""v1.12 测试：EIS 几何特征 / 配对差分 / Maccor 块容量。"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from bmthermal.eis_features import (
    EIS_N_POINTS,
    _identify_columns,
    block_cc_discharge_capacity,
    block_charge_capacity,
    block_discharge_capacity,
    load_eis_spectrum,
    load_maccor_block,
    nyquist_features,
    pair_deltas,
    parse_block_member,
    parse_eis_member,
)

EIS_ZIP = Path("D:/项目/02_数据集/01_国际公开集/Oxford_PathDependent/EIS.zip")
GROUP_ZIP = Path("D:/项目/02_数据集/01_国际公开集/Oxford_PathDependent/Group_1.zip")


def _synth_spectrum(hook: float = -0.01, arc: float = 0.004,
                    tail: float = 0.006, r0: float = 0.045):
    """合成谱：高频感性钩 + 单弧 + 扩散尾（索引 0 = 高频端）。"""
    n = EIS_N_POINTS
    idx = np.arange(n, dtype=float)
    z_re = r0 + 0.0004 * idx + 0.012 * (idx / n) ** 3      # 单调升
    # z_im: hook(负) → 0 → 弧峰 → 谷 → 尾(升)
    z_im = np.empty(n)
    hook_pts = 5
    z_im[:hook_pts] = hook * np.linspace(1.0, 0.0, hook_pts)
    arc_pts = 15
    arc_x = np.linspace(0, 1, arc_pts)
    z_im[hook_pts:hook_pts + arc_pts] = arc * np.sin(np.pi * arc_x) ** 2
    tail_start = hook_pts + arc_pts
    z_im[tail_start:] = np.linspace(0.001, tail, n - tail_start)
    # 弧谷自然落在 tail_start
    return z_re, z_im


class TestNyquistFeatures:
    def test_basic_geometry(self):
        z_re, z_im = _synth_spectrum()
        f = nyquist_features(z_re, z_im)
        assert f.n_pts == EIS_N_POINTS
        assert f.r_ohm == pytest.approx(0.045, rel=1e-6)
        assert f.r_pol == pytest.approx(float(z_re.max() - z_re.min()), rel=1e-9)
        assert f.z_im_hf == pytest.approx(-0.01, rel=1e-6)
        assert f.zero_cross == 4                       # hook 末点恰为 0
        assert f.arc_height > 0.003                    # 弧峰 0.004 附近
        assert f.tail_slope > 0                        # 尾段上行
        assert f.lf_abs > f.r_ohm * 0.9
        assert f.quality == "ok"

    def test_no_hook_zero_cross_at_start(self):
        z_re, z_im = _synth_spectrum(hook=0.0)
        z_im[:5] = 0.0
        f = nyquist_features(z_re, z_im)
        assert f.zero_cross == 0

    def test_monotonic_tail_no_valley(self):
        z_re = np.linspace(0.04, 0.09, EIS_N_POINTS)
        z_im = np.linspace(-0.01, 0.02, EIS_N_POINTS)   # 过零后单调升
        f = nyquist_features(z_re, z_im)
        assert f.quality == "no_arc"                   # 无局部极大=无弧
        assert f.r_ohm == pytest.approx(0.04)

    def test_pair_deltas(self):
        z_re, z_im = _synth_spectrum()
        bol = nyquist_features(z_re, z_im)
        # MoL（物理扰动）：R_ohm +2 mΩ（实轴平移）+ 弧幅 0.004→0.005
        z_re2, z_im2 = _synth_spectrum(arc=0.005)
        mol = nyquist_features(z_re2 + 0.002, z_im2)
        p = pair_deltas(bol, mol, group=1, cell=15, soc=20)
        assert p.deltas["d_r_ohm"] == pytest.approx(0.002, rel=1e-6)
        assert p.deltas["d_arc_height"] == pytest.approx(0.001, rel=1e-3)
        assert p.group == 1 and p.cell == 15 and p.soc == 20

    def test_arc_height_shift_invariant(self):
        # Z_im 整体平移是差值不变量（弧高=峰−起点）——设计如此
        z_re, z_im = _synth_spectrum()
        f0 = nyquist_features(z_re, z_im)
        f1 = nyquist_features(z_re, z_im + 0.001)
        assert f1.arc_height == pytest.approx(f0.arc_height, rel=1e-6)

    def test_to_dict_serializable(self):
        z_re, z_im = _synth_spectrum()
        d = nyquist_features(z_re, z_im).to_dict()
        assert isinstance(d["r_ohm"], float)
        assert isinstance(d["zero_cross"], int)


class TestMaccorBlock:
    def _df(self, t, i):
        import pandas as pd
        return pd.DataFrame({"TestTime": t, "Amps": i,
                             "Volts": np.full_like(t, 3.7),
                             "Amphr": np.zeros_like(t)})

    def test_discharge_capacity_integral(self):
        # TestTime 单位分钟：2 A × 30 min → 1.0 Ah
        t = np.linspace(0, 30, 361)
        i = np.full_like(t, -2.0)
        q = block_discharge_capacity(self._df(t, i))
        assert q == pytest.approx(1.0, rel=1e-6)

    def test_charge_capacity_integral(self):
        t = np.linspace(0, 60, 361)
        i = np.full_like(t, 1.5)
        assert block_charge_capacity(self._df(t, i)) == pytest.approx(1.5, rel=1e-6)
        assert block_discharge_capacity(self._df(t, i)) == pytest.approx(0.0, abs=1e-9)

    def test_mixed_block(self):
        t = np.linspace(0, 60, 721)
        i = np.where(t < 30, -2.0, 1.0)
        # 阶跃处梯形离散化误差 ~0.2%
        assert block_discharge_capacity(self._df(t, i)) == pytest.approx(1.0, rel=5e-3)
        assert block_charge_capacity(self._df(t, i)) == pytest.approx(0.5, rel=5e-3)

    def test_cc_discharge_segment(self):
        # 两段放电：短脉冲 + 长 CC（1.5 A × 40 min = 1.0 Ah）→ 取 CC 段
        t = np.linspace(0, 50, 601)
        i = np.zeros_like(t)
        i[t < 2] = -4.0                       # 脉冲
        i[(t >= 5) & (t <= 45)] = -1.5        # 参考放电
        q_cc = block_cc_discharge_capacity(self._df(t, i))
        assert q_cc == pytest.approx(1.0, rel=0.02)

    def test_identify_columns_behavioral(self):
        import pandas as pd
        n = 2000
        df = pd.DataFrame({
            "c0": np.zeros(n),
            "c1": np.linspace(0, 5000, n),        # TestTime
            "c2": -1.5 + 0.002 * np.arange(n),    # Amps（有正有负）
            "c3": np.linspace(3.0, 4.2, n),       # Volts
            "c4": np.linspace(0, 2.8, n),         # Amphr
        })
        m = _identify_columns(df)
        assert m.get("c1") == "TestTime"
        assert m.get("c2") == "Amps"
        assert m.get("c3") == "Volts"
        assert m.get("c4") == "Amphr"


class TestEISNaming:
    def test_parse_member(self):
        d = parse_eis_member("EIS/BoL/Group1_BoL_Cell15_SOC20.mat")
        assert d == {"stage": "BoL", "group": 1, "cell": 15, "soc": 20}
        d2 = parse_eis_member("EIS/MoL/Group10_MoL_Cell22_SOC80.mat")
        assert d2 == {"stage": "MoL", "group": 10, "cell": 22, "soc": 80}
        assert parse_eis_member("EIS/BoL/readme.txt") is None

    def test_parse_block_member(self):
        # 命名不规则：空格/有无小数步
        assert parse_block_member("Group 1/TPG1 - Cell 15.mat") == \
            {"tpg": 1, "step": 1, "cell": 15}
        assert parse_block_member("Group 1/TPG1.2 - Cell15.mat") == \
            {"tpg": 1, "step": 2, "cell": 15}
        assert parse_block_member("Group 1/TPG1.10-Cell15.mat") == \
            {"tpg": 1, "step": 10, "cell": 15}
        assert parse_block_member("readme.txt") is None

    def test_load_real_eis_spectrum(self):
        if not EIS_ZIP.exists():
            pytest.skip("EIS.zip not on this machine")
        with zipfile.ZipFile(EIS_ZIP) as z:
            z_re, z_im = load_eis_spectrum(
                z.read("EIS/BoL/Group1_BoL_Cell15_SOC20.mat"))
        assert z_re.size == EIS_N_POINTS
        f = nyquist_features(z_re, z_im)
        assert 0.02 < f.r_ohm < 0.2          # 18650 欧姆段量级
        assert f.r_pol > 0

    def test_load_real_maccor_block(self):
        if not GROUP_ZIP.exists():
            pytest.skip("Group_1.zip not on this machine")
        with zipfile.ZipFile(GROUP_ZIP) as z:
            df = load_maccor_block(z.read("Group 1/TPG1 - Cell 15.mat"))
        for req in ("TestTime", "Amps", "Volts", "Amphr"):
            assert req in df.columns
        assert not df.columns.duplicated().any()
        q = block_discharge_capacity(df)
        assert q > 5.0                     # 块吞吐（41 循环，分钟制）
        assert block_charge_capacity(df) > 5.0
        assert 2.5 < block_cc_discharge_capacity(df) < 3.5   # 参考放电 ~3.1 Ah
