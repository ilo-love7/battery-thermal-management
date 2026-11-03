"""v1.11 测试：电极耦合映射验证（electrode_coupling）。

覆盖：MAT 元素流解码往返、MCOS 列抽取、电极曲线归一、
耦合重建单调性、三损伤注入方向性、恢复矩阵对角占优、
真实曲线几何灵敏度（shift/shape 合成扰动恢复）。
"""
from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from bmthermal.electrode_coupling import (
    DAMAGE_KINDS,
    _build_vq,
    electrode_ux,
    fit_coupling,
    geometry_sensitivity,
    inject_damage,
    load_mcos_columns,
    load_oxford_halfcell,
    parse_mat_elements,
    recovery_matrix,
    synthetic_shape,
    synthetic_shift,
)

# ---------------------------------------------------------------------------
# 测试用合成电极曲线（LCO 风格正极 + 石墨风格负极）
# ---------------------------------------------------------------------------


def _xp_up(n=200):
    x = np.linspace(0, 1, n)
    # 正极：随嵌锂度单调降，带平台斜率变化
    u = 4.25 - 1.1 * x - 0.18 * np.sin(np.pi * x)
    return x, u


def _xn_un(n=200):
    x = np.linspace(0, 1, n)
    # 负极：石墨风格——初段陡、后段平台
    u = 0.085 + 0.16 * np.exp(-9.0 * x) + 0.05 * x
    return x, u


@pytest.fixture()
def electrodes():
    return _xp_up(), _xn_un()


# ---------------------------------------------------------------------------
# MAT 元素流解码
# ---------------------------------------------------------------------------


def _mk_double_element(vals: np.ndarray) -> bytes:
    """手工构造一个 miDOUBLE 元素（普通 tag + 对齐填充）。"""
    data = np.asarray(vals, dtype="<f8").tobytes()
    tag = struct.pack("<II", 9, len(data))
    pad = b"\x00" * ((8 - len(data) % 8) % 8)
    return tag + data + pad


def test_parse_mat_elements_roundtrip():
    vals = np.linspace(0.0, 1.0, 37)
    buf = _mk_double_element(vals) + _mk_double_element(vals * 2)
    out = parse_mat_elements(buf)
    arrays = [a for typ, _d, a in out if typ == "double"]
    assert len(arrays) == 2
    assert arrays[0] == pytest.approx(vals)
    assert arrays[1] == pytest.approx(vals * 2)


def test_load_mcos_columns_from_workspace(tmp_path):
    import scipy.io as sio

    vals = np.linspace(2.0, 4.0, 150)
    blob = _mk_double_element(vals)
    # scipy 读 __function_workspace__ 的前提是同名变量存在
    sio.savemat(tmp_path / "t.mat", {"__function_workspace__": np.frombuffer(blob, np.uint8)})
    cols = load_mcos_columns((tmp_path / "t.mat").read_bytes())
    assert len(cols) == 1
    assert cols[0] == pytest.approx(vals)


def test_load_mcos_columns_missing_workspace():
    import scipy.io as sio, io, tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.mat"
        sio.savemat(p, {"other": np.zeros(3)})
        with pytest.raises(ValueError):
            load_mcos_columns(p.read_bytes())


# ---------------------------------------------------------------------------
# 电极曲线与耦合
# ---------------------------------------------------------------------------


def test_electrode_ux_normalized(electrodes):
    (xp, Up), (xn, Un) = electrodes
    # 造半电池 df（Q 升序）
    import pandas as pd
    q = np.linspace(0, 0.005, 120)
    df = pd.DataFrame({"Q_Ah": q, "V": np.interp(q / q[-1], xp, Up)})
    x, u = electrode_ux(df, lithiation=True)
    assert x[0] == pytest.approx(0.0) and x[-1] == pytest.approx(1.0)
    assert np.all(np.diff(x) >= -1e-12)
    assert np.all(np.diff(u) < 0.02)  # 单调降（正极）


def test_build_vq_monotonic(electrodes):
    (xp, Up), (xn, Un) = electrodes
    q, v = _build_vq(xp, Up, xn, Un, 1.0, 1.15, 0.15, 0.85)
    assert q[0] == pytest.approx(0.0)
    assert np.all(np.diff(q) > 0)
    assert np.all(np.diff(v) < 0.01)   # 放电整体降
    assert v[0] > 3.5 and (v[0] - v[-1]) > 0.5   # 满电高、放电跌落显著


def test_fit_coupling_recovers_capacity(electrodes):
    (xp, Up), (xn, Un) = electrodes
    # 真值曲线作 POCV 目标
    q_true, v_true = _build_vq(xp, Up, xn, Un, 1.0, 1.15, 0.15, 0.85, 300)
    scale = 2.8 / float(q_true[-1])   # 绝对容量 2.8 Ah
    fit, (q_m, v_m) = fit_coupling(xp, Up, xn, Un, q_true * scale, v_true)
    assert fit.rmse_v < 0.05          # 重建贴合
    assert fit.q_use_norm == pytest.approx(float(q_true[-1]), rel=0.10)
    assert fit.x_p0 == pytest.approx(0.15, abs=0.05)   # x_p0 可辨识
    # ⚠️ (x_n0, N/P) 沿山脊弱可辨识（V(q) 只约束轨迹组合）——
    # 只断言落在搜索网格内，不逐点对真值（v1.11 可辨识性记录）
    assert 1.0 <= fit.np_ratio <= 1.6


# ---------------------------------------------------------------------------
# 损伤注入与恢复矩阵
# ---------------------------------------------------------------------------


def test_inject_damage_capacity_loss(electrodes):
    (xp, Up), (xn, Un) = electrodes
    q_true, v_true = _build_vq(xp, Up, xn, Un, 1.0, 1.15, 0.15, 0.85, 300)
    scale = 2.8 / float(q_true[-1])
    fit, _ = fit_coupling(xp, Up, xn, Un, q_true * scale, v_true)
    for kind in DAMAGE_KINDS:
        q, v = inject_damage(xp, Up, xn, Un, fit, kind, 0.10)
        loss = 1.0 - float(q[-1]) / fit.q_meas_ah
        assert loss == pytest.approx(0.10, abs=0.02), kind


def test_recovery_matrix_ordering(electrodes):
    """LLI 应比 LAM 更 shift-like（share_shape 更低）。"""
    (xp, Up), (xn, Un) = electrodes
    q_true, v_true = _build_vq(xp, Up, xn, Un, 1.0, 1.15, 0.15, 0.85, 300)
    scale = 2.8 / float(q_true[-1])
    fit, _ = fit_coupling(xp, Up, xn, Un, q_true * scale, v_true)
    df = recovery_matrix(xp, Up, xn, Un, fit, q_loss_frac=0.10)
    sh = {r["injected"]: r["share_shape"] for _, r in df.iterrows()}
    assert sh["lli"] < sh["lam_pe"]
    assert sh["lli"] < sh["lam_ne"]
    # LLI 至少应有可辨识的 shift 分量
    assert sh["lli"] < 0.6


def test_synthetic_shift_shape_pure(electrodes):
    """合成扰动应恢复出近纯份额（估计器自检）。"""
    (xp, Up), (xn, Un) = electrodes
    q, v = _build_vq(xp, Up, xn, Un, 1.0, 1.15, 0.15, 0.85, 300)
    g = geometry_sensitivity(q, v, loss_frac=0.10)
    assert g["shift_injected"]["share_shape"] < 0.15
    assert g["shape_injected"]["share_shape"] > 0.85
    assert not g["shift_injected"]["low_quality"]
    assert not g["shape_injected"]["low_quality"]


def test_synthetic_helpers_loss_amount(electrodes):
    (xp, Up), (xn, Un) = electrodes
    q, v = _build_vq(xp, Up, xn, Un, 1.0, 1.15, 0.15, 0.85, 300)
    q_s, v_s = synthetic_shift(q, v, 0.10)
    q_h, v_h = synthetic_shape(q, v, 0.10)
    assert float(q_s[-1]) == pytest.approx(float(q[-1]) * 0.90, rel=1e-6)
    assert float(q_h[-1]) == pytest.approx(float(q[-1]) * 0.90, rel=1e-6)
    assert q_s[0] < 2 * float(q[1] - q[0])   # 平移后起点≈0（一个网格间距内）
    # shift 是平移：间距不变（丢掉 q<δ 的头）；shape 是缩放：间距 ×0.9
    assert np.allclose(np.diff(q_s), np.diff(q)[-(len(q_s) - 1):], rtol=1e-9)
    assert np.allclose(np.diff(q_h), np.diff(q) * 0.9, rtol=1e-9)


# ---------------------------------------------------------------------------
# 真实数据集成（数据存在才跑）
# ---------------------------------------------------------------------------

_OXFORD_ZIP = Path("D:/项目/02_数据集/01_国际公开集/Oxford_PathDependent/Half_Cell.zip")


@pytest.mark.skipif(not _OXFORD_ZIP.exists(), reason="Oxford Half_Cell.zip 不在本机")
def test_oxford_halfcell_loads():
    data = load_oxford_halfcell(_OXFORD_ZIP)
    for key in ("anode_dch", "cathode_dch", "full_dch"):
        assert key in data, key
        df = data[key]
        assert len(df) > 1000
        # DCH_HC_anode 尾部有 NaN 填充（官方导出），≥90% 有限即可
        assert np.isfinite(df["V"]).mean() > 0.90
    # 电荷守恒粗校验（单位已标定）
    cath = data["cathode_dch"]
    assert cath["Q_Ah"].max() < 0.1   # 半电池片级 mAh 容量
