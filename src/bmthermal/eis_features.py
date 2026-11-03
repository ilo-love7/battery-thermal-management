"""EIS Nyquist 几何特征 + Oxford PathDependent 阻抗-退化模式分析（v1.12）。

数据源：Oxford_PathDependent/EIS.zip（117 谱，BoL/MoL × SOC{20,50,80}）
与 Group_*.zip（Maccor 表块，RPT 容量轨迹）。

⚠️ 频率轴边界（如实声明）：EIS .mat 只存 (Z_re, Z_im) 35 点，不存频率
（ORA 指南/论文实验条件均不可得）。全部特征在**索引空间**定义——
同一仪器网格下谱间严格可比，绝对 Hz 不可溯源。不做任何频率依赖量
（DRT/弛豫时间常数）；引用时只用差分/比值口径。

几何约定：索引 0 = 高频端（Z_im 负向感性钩、Z_re≈R_ohm），索引 -1 =
低频端（扩散尾）。该方向由全部 117 谱一致的形状（钩→过零→弧→谷→尾）
验证，非单谱假设。
"""
from __future__ import annotations

import io
import re
import struct
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bmthermal.electrode_coupling import load_mcos_columns, parse_mat_elements

EIS_N_POINTS = 35          # 全部 117 谱统一 35 点
_TAIL_PTS = 5              # 低频尾斜率取末 5 点


# ---------------------------------------------------------------------------
# EIS 谱加载
# ---------------------------------------------------------------------------

def load_eis_spectrum(mat_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """EIS .mat 字节 → (z_re, z_im)。取等长 float 列对（35 点）。"""
    cols = load_mcos_columns(mat_bytes, min_len=10)
    pairs = [np.asarray(c, dtype=float) for c in cols
             if np.asarray(c).size == EIS_N_POINTS]
    if len(pairs) < 2:
        raise ValueError(f"need 2 x {EIS_N_POINTS}-pt columns, got "
                         f"{[np.asarray(c).size for c in cols]}")
    z_re, z_im = pairs[0], pairs[1]
    if not (np.isfinite(z_re).all() and np.isfinite(z_im).all()):
        raise ValueError("non-finite values in spectrum")
    return z_re, z_im


# ---------------------------------------------------------------------------
# 频率无关 Nyquist 几何特征
# ---------------------------------------------------------------------------

@dataclass
class NyquistFeatures:
    """单谱几何特征（索引空间，谱间可比）。"""
    n_pts: int
    r_ohm: float          # 高频端实部 Z_re[0]（R_ohm 代理）
    r_pol: float          # 实部展宽 max-min（极化电阻代理）
    z_im_hf: float        # 高频端 Z_im[0]（感性钩深）
    z_im_peak: float      # 弧峰 Z_im
    arc_height: float     # 弧峰 − 过零点 Z_im（≈弧峰−0）
    zero_cross: int       # Z_im 首过零索引（-1 = 未过零）
    valley_idx: int       # 弧谷索引（弧/尾分割，-1 = 未找到）
    arc_dre: float        # 弧段实部宽 Z_re[valley] − Z_re[zero_cross]
    tail_slope: float     # 尾段 dZ_im/dZ_re（末 _TAIL_PTS 点）
    lf_abs: float         # 低频端 |Z|
    lf_phase_deg: float   # 低频端相位角 atan2(Z_im, Z_re)（度）
    quality: str = "ok"   # ok | no_zero_cross | no_valley | no_arc

    def to_dict(self) -> dict:
        return {k: (int(v) if isinstance(v, (int, np.integer)) else
                    float(v) if isinstance(v, (float, np.floating)) else v)
                for k, v in self.__dict__.items()}


def nyquist_features(z_re: np.ndarray, z_im: np.ndarray) -> NyquistFeatures:
    """单谱几何特征提取。

    分割规则（全部由谱形几何驱动，不依赖频率）：
    - 感性钩 = Z_im < 0 的前段；
    - 过零 = Z_im 首次 ≥ 0 的索引；
    - 弧峰 = 过零后 Z_im 最大点；
    - 弧谷 = 弧峰后 Z_im 局部最小点（弧与扩散尾的分割）；
    - 尾 = 弧谷 → 末端。
    """
    z_re = np.asarray(z_re, dtype=float)
    z_im = np.asarray(z_im, dtype=float)
    n = z_re.size
    if n < 8:
        raise ValueError(f"too few points: {n}")

    r_ohm = float(z_re[0])
    r_pol = float(z_re.max() - z_re.min())
    z_im_hf = float(z_im[0])

    above = np.nonzero(z_im >= 0)[0]
    zero_cross = int(above[0]) if above.size else -1
    start = zero_cross if zero_cross >= 0 else 0

    # 弧峰 = 过零后**首个局部极大**（尾段可高于弧峰——真实谱同构，
    # 全局最大会误判到低频端）；弧谷 = 峰后首个局部极小
    peak_idx, valley_idx = -1, -1
    for i in range(start + 1, n - 1):
        if z_im[i] >= z_im[i - 1] and z_im[i] >= z_im[i + 1] \
                and z_im[i] > z_im[start]:
            peak_idx = i
            break
    if peak_idx == -1:
        quality = "no_zero_cross" if zero_cross == -1 else "no_arc"
        peak_idx = int(np.argmax(z_im))
        arc_height = float(z_im[peak_idx] - z_im[start])
        arc_dre = 0.0
        valley_idx = -1
    else:
        for i in range(peak_idx + 1, n - 1):
            if z_im[i] <= z_im[i - 1] and z_im[i] <= z_im[i + 1] \
                    and z_im[i] < z_im[peak_idx]:
                valley_idx = i
                break
        quality = "ok" if valley_idx != -1 else "no_valley"
        if valley_idx == -1:
            valley_idx = peak_idx
        arc_height = float(z_im[peak_idx] - z_im[start])
        arc_dre = float(z_re[valley_idx] - z_re[start])

    k = min(_TAIL_PTS, n // 3)
    dre = float(z_re[-1] - z_re[-k])
    tail_slope = float((z_im[-1] - z_im[-k]) / dre) if dre > 0 else float("nan")

    lf_abs = float(np.hypot(z_re[-1], z_im[-1]))
    lf_phase = float(np.degrees(np.arctan2(z_im[-1], z_re[-1])))
    z_im_peak = float(z_im[peak_idx])

    return NyquistFeatures(
        n_pts=n, r_ohm=r_ohm, r_pol=r_pol, z_im_hf=z_im_hf,
        z_im_peak=z_im_peak, arc_height=arc_height,
        zero_cross=zero_cross, valley_idx=valley_idx, arc_dre=arc_dre,
        tail_slope=tail_slope, lf_abs=lf_abs, lf_phase_deg=lf_phase,
        quality=quality,
    )


# ---------------------------------------------------------------------------
# BoL→MoL 配对差分
# ---------------------------------------------------------------------------

_DELTA_FIELDS = ("r_ohm", "r_pol", "z_im_hf", "arc_height", "arc_dre",
                 "tail_slope", "lf_abs", "lf_phase_deg")


@dataclass
class EISPair:
    """同 (cell, SOC) 的 BoL→MoL 对。"""
    group: int
    cell: int
    soc: int
    bol: NyquistFeatures
    mol: NyquistFeatures
    deltas: dict = field(default_factory=dict)

    def __post_init__(self):
        for k in _DELTA_FIELDS:
            self.deltas[f"d_{k}"] = float(getattr(self.mol, k)
                                          - getattr(self.bol, k))

    @property
    def fade_share_rpol(self) -> float:
        """ΔR_pol 中弧段占比（>1 或 <0 如实透传）。"""
        d = self.deltas
        if d["d_r_pol"] <= 0:
            return float("nan")
        return float(d["d_arc_dre"] / d["d_r_pol"])


def pair_deltas(bol: NyquistFeatures, mol: NyquistFeatures,
                group: int, cell: int, soc: int) -> EISPair:
    return EISPair(group=group, cell=cell, soc=soc, bol=bol, mol=mol)


# ---------------------------------------------------------------------------
# Maccor 块加载（Group_*.zip → RPT 容量轨迹）
# ---------------------------------------------------------------------------

# 列名（Maccor 导出，元素流 utf8 负载解码，v1.12）：
# Cyc, Step, TestTime, StepTime, Amphr, Watthr, Amps, Volts, Temp1, VARx1-15
# ⚠️ TestTime/StepTime 单位是**分钟**（d(Amphr)/(I·dt)=1/60 实测标定，
# 1 Hz 采样）——容量积分必须 /60。
_MACCOR_REQUIRED = ("TestTime", "Amps", "Volts", "Amphr")
_MACCOR_NAME_RE = re.compile(
    r"^(Cyc|Step|TestTime|StepTime|Amphr|Watthr|Amps|Volts|Temp\d*|VARx\d+)$")


def _utf_texts(blob: bytes) -> list[str]:
    """元素流 utf 负载 → 字符串列表（按出现顺序 = varnames 顺序）。"""
    out: list[str] = []
    for typ, _depth, a in parse_mat_elements(blob):
        if typ in ("utf8", "utf16", "utf32") and isinstance(a, (bytes, bytearray)):
            enc = {"utf8": "utf-8", "utf16": "utf-16-le", "utf32": "utf-32-le"}[typ]
            s = bytes(a).decode(enc, "replace").strip("\x00")
            if s and all(32 <= ord(c) < 127 for c in s):
                out.append(s)
    return out


def load_maccor_block(mat_bytes: bytes, min_len: int = 1000) -> pd.DataFrame:
    """Maccor 表块 .mat 字节 → DataFrame（列名来自 utf 元素流）。

    列名经正则精确过滤（24 名 ↔ 24 列）；数量不符时按量程行为学
    回退识别（Amps: 有正有负 / Volts: 2–5 V / TestTime: 单调），
    回退时列名加后缀防撞车。
    """
    import scipy.io as sio

    m = sio.loadmat(io.BytesIO(mat_bytes), struct_as_record=False,
                    squeeze_me=True)
    if "__function_workspace__" not in m:
        raise ValueError("no __function_workspace__ (not MCOS)")
    fw = np.asarray(m["__function_workspace__"])
    blob = fw.tobytes() if fw.dtype == np.uint8 else bytes(fw)

    names = [s for s in _utf_texts(blob) if _MACCOR_NAME_RE.match(s)]
    cols_all: list[np.ndarray] = []
    for typ, _d, a in parse_mat_elements(blob):
        if typ in ("double", "single") and isinstance(a, np.ndarray) \
                and a.size >= min_len:
            cols_all.append(np.asarray(a, dtype=float))
    # 主表行数 = 最常见列长（个别块混入异长数组，v1.12 坑）
    if cols_all:
        from collections import Counter
        n_rows = Counter(a.size for a in cols_all).most_common(1)[0][0]
        cols = [a for a in cols_all if a.size == n_rows]
    else:
        cols = cols_all

    if len(names) == len(cols):
        df = pd.DataFrame({n: c for n, c in zip(names, cols)})
    else:
        df = pd.DataFrame({f"col{i}": c for i, c in enumerate(cols)})
        mapping = _identify_columns(df)
        df = df.rename(columns=mapping)
        if df.columns.duplicated().any():
            seen: dict[str, int] = {}
            new_cols = []
            for c in df.columns:
                if c in seen:
                    seen[c] += 1
                    new_cols.append(f"{c}_{seen[c]}")
                else:
                    seen[c] = 0
                    new_cols.append(c)
            df.columns = new_cols
    for req in _MACCOR_REQUIRED:
        if req not in df.columns:
            raise ValueError(f"missing column {req}; have {list(df.columns)}")
    return df


def _identify_columns(df: pd.DataFrame) -> dict[str, str]:
    """量程行为学列识别：返回 {旧名: 新名}。"""
    out: dict[str, str] = {}
    used: set[str] = set()
    for c in df.columns:
        v = df[c].to_numpy()
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if c in used:
            continue
        if lo < -0.5 and hi > 0.5 and hi < 50:
            out[c] = "Amps"; used.add(c)
        elif 2.0 <= lo and hi <= 5.0 and (hi - lo) > 0.3:
            out[c] = "Volts"; used.add(c)
        elif np.all(np.diff(v[np.isfinite(v)]) >= -1e-9) and hi > 60:
            out[c] = "TestTime"; used.add(c)
        elif 0 <= lo and 0.5 < hi < 10 and np.all(np.diff(v[:50]) >= -1e-9):
            out[c] = "Amphr"; used.add(c)
    return out


def block_discharge_capacity(df: pd.DataFrame) -> float:
    """块内放电容量 [Ah] = ∫max(−I,0)dt / 60（TestTime 单位分钟）。

    对 RPT 块（含充/放/静置）返回总放出 Ah；对纯充电块返回 ≈0。
    """
    t = df["TestTime"].to_numpy(dtype=float)
    i = df["Amps"].to_numpy(dtype=float)
    dt = np.diff(t)
    i_mid = 0.5 * (i[1:] + i[:-1])
    q_dis = np.cumsum(np.maximum(-i_mid, 0.0) * np.maximum(dt, 0.0))
    return float(q_dis[-1] / 60.0) if q_dis.size else 0.0


def block_charge_capacity(df: pd.DataFrame) -> float:
    """块内充电容量 [Ah] = ∫max(I,0)dt / 60。"""
    t = df["TestTime"].to_numpy(dtype=float)
    i = df["Amps"].to_numpy(dtype=float)
    dt = np.diff(t)
    i_mid = 0.5 * (i[1:] + i[:-1])
    q_chg = np.cumsum(np.maximum(i_mid, 0.0) * np.maximum(dt, 0.0))
    return float(q_chg[-1] / 60.0) if q_chg.size else 0.0


def block_cc_discharge_capacity(df: pd.DataFrame,
                                i_min: float = -0.1,
                                tol: float = 0.05) -> float:
    """块内最大单段恒流放电容量 [Ah]（参考放电的稳健代理）。

    找 |I − median(段)| ≤ tol·|median| 的最长负电流段，积分其 Ah。
    对 RPT 块返回参考放电容量；对脉冲/杂块返回较小值。
    """
    t = df["TestTime"].to_numpy(dtype=float)
    i = df["Amps"].to_numpy(dtype=float)
    dis = i < i_min
    if not dis.any():
        return 0.0
    # 找连续 True 段
    best_q = 0.0
    idx = np.nonzero(dis)[0]
    splits = np.nonzero(np.diff(idx) > 1)[0]
    segs = np.split(idx, splits + 1)
    for seg in segs:
        if seg.size < 3:
            continue
        i_seg = i[seg]
        i_med = float(np.median(i_seg))
        if i_med >= i_min:
            continue
        cc = np.abs(i_seg - i_med) <= tol * abs(i_med) + 1e-9
        # 取段内最长 CC 子段
        sub_idx = np.nonzero(cc)[0]
        if sub_idx.size < 3:
            continue
        s2 = np.nonzero(np.diff(sub_idx) > 1)[0]
        for sub in np.split(sub_idx, s2 + 1):
            if sub.size < 3:
                continue
            tt = t[seg[sub]]
            ii = i[seg[sub]]
            q = float(np.sum(0.5 * (ii[1:] + ii[:-1]) * np.diff(tt)) / 60.0)
            best_q = max(best_q, -q)
    return best_q


# ---------------------------------------------------------------------------
# 文件名解析
# ---------------------------------------------------------------------------

_EIS_NAME_RE = re.compile(
    r"EIS/(BoL|MoL)/Group(\d+)_(?:BoL|MoL)_Cell(\d+)_SOC(\d+)\.mat")

# Maccor 块成员名不规则：'TPG1 - Cell 15.mat' / 'TPG1.2 - Cell15.mat' /
# 'TPG1.4 - Cell 15.mat' / 'TPG1.6-Cell15.mat' 皆有之
_BLOCK_NAME_RE = re.compile(r"TPG(\d+)(?:\.(\d+))?\s*-\s*Cell\s*(\d+)\.mat")


def parse_eis_member(name: str) -> dict | None:
    """EIS 成员名 → {stage, group, cell, soc}。"""
    m = _EIS_NAME_RE.match(name)
    if not m:
        return None
    return {"stage": m.group(1), "group": int(m.group(2)),
            "cell": int(m.group(3)), "soc": int(m.group(4))}


def parse_block_member(name: str) -> dict | None:
    """Maccor 块成员名 → {tpg, step, cell}（step=1 表示 TPGx 无后缀块）。"""
    m = _BLOCK_NAME_RE.search(name.replace("/", " "))
    if not m:
        return None
    step = int(m.group(2)) if m.group(2) else 1
    return {"tpg": int(m.group(1)), "step": step, "cell": int(m.group(3))}
