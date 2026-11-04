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
        self.deltas.update(stage_deltas(self.bol, self.mol))

    @property
    def fade_share_rpol(self) -> float:
        """ΔR_pol 中弧段占比（>1 或 <0 如实透传）。"""
        d = self.deltas
        if d["d_r_pol"] <= 0:
            return float("nan")
        return float(d["d_arc_dre"] / d["d_r_pol"])


def stage_deltas(before: NyquistFeatures,
                 after: NyquistFeatures) -> dict[str, float]:
    """任意两阶段特征差分（BoL→MoL / MoL→EoL / BoL→EoL 通用）。

    与 EISPair.deltas 同口径；阶段链可加性由测试把关
    （d_BoL→EoL ≈ d_BoL→MoL + d_MoL→EoL，数值误差 < 1e-9）。
    """
    return {f"d_{k}": float(getattr(after, k) - getattr(before, k))
            for k in _DELTA_FIELDS}


def chain_share(seg1: dict[str, float], seg2: dict[str, float],
                key: str = "d_arc_dre") -> float:
    """两段增量中后段占比（MoL→EoL 段占全链 BoL→EoL 的份额）。

    分母 ≤ 0 时返回 nan（如实透传，不裁剪）。
    """
    total = seg1.get(key, 0.0) + seg2.get(key, 0.0)
    if total <= 0:
        return float("nan")
    return float(seg2.get(key, 0.0) / total)


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

# ---- 脏数据防线（v1.15）：块内 TestTime 哨兵（~1.3e15）/荒谬电流
# 会让 dt 积分爆炸（q 6.5e292、EFC 1e12 量级野值）。
I_MAX_A = 50.0        # 18650 物理上限（C/2≈1.5 A；脉冲 ≲4C≈12 A）
T_SENTINEL_MIN = 1e9  # |TestTime| 超此视为哨兵脏值（分钟）
DT_MAX_MIN = 600.0    # 相邻样本可信最大间隔（1 s 采样；越界视为断点按 0 计）

# ---- 静置窗口径（v1.15 rest dose）----
REST_I_THRESH = 0.01   # A：|I| 低于此视为静置
REST_MIN_MIN = 1.0     # 分钟：静置窗最小时长
REST_EXT_MIN = 120.0   # 分钟：延长静置窗（恢复相关）阈值


def _ok_mask(t: np.ndarray, i: np.ndarray,
             v: np.ndarray | None = None) -> np.ndarray:
    """清洗掩码（t/i 同 _sanitize_mi；v 传入时要求其有限）。"""
    ok = (np.isfinite(t) & np.isfinite(i)
          & (np.abs(t) < T_SENTINEL_MIN) & (np.abs(i) <= I_MAX_A))
    if v is not None:
        ok = ok & np.isfinite(v)
    return ok


def _sanitize_mi(t: np.ndarray, i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """清洗时间/电流：剔除非有限值、时间哨兵、荒谬电流。"""
    t = np.asarray(t, dtype=float)
    i = np.asarray(i, dtype=float)
    if t.size != i.size:
        raise ValueError("t/i size mismatch")
    ok = _ok_mask(t, i)
    return t[ok], i[ok]


def rest_run_durations(t: np.ndarray, i: np.ndarray,
                       i_thresh: float = REST_I_THRESH,
                       min_run_min: float = REST_MIN_MIN) -> np.ndarray:
    """静置窗时长数组 [分钟]：|I|<i_thresh 的连续段（跨段不合并）。

    时长 = 末样本时间 − 首样本时间（单样本段记 0，被 min_run_min 滤除）。
    """
    t = np.asarray(t, dtype=float)
    i = np.asarray(i, dtype=float)
    if t.size < 2:
        return np.empty(0)
    mask = np.abs(i) < i_thresh
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    durs = t[ends] - t[starts]
    return durs[np.isfinite(durs) & (durs >= min_run_min)]
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
    t, i = _sanitize_mi(df["TestTime"].to_numpy(dtype=float),
                        df["Amps"].to_numpy(dtype=float))
    if t.size < 2:
        return 0.0
    dt = np.where(np.diff(t) > DT_MAX_MIN, 0.0, np.maximum(np.diff(t), 0.0))
    i_mid = 0.5 * (i[1:] + i[:-1])
    q_dis = np.cumsum(np.maximum(-i_mid, 0.0) * dt)
    return float(q_dis[-1] / 60.0) if q_dis.size else 0.0


def block_charge_capacity(df: pd.DataFrame) -> float:
    """块内充电容量 [Ah] = ∫max(I,0)dt / 60。"""
    t, i = _sanitize_mi(df["TestTime"].to_numpy(dtype=float),
                        df["Amps"].to_numpy(dtype=float))
    if t.size < 2:
        return 0.0
    dt = np.where(np.diff(t) > DT_MAX_MIN, 0.0, np.maximum(np.diff(t), 0.0))
    i_mid = 0.5 * (i[1:] + i[:-1])
    q_chg = np.cumsum(np.maximum(i_mid, 0.0) * dt)
    return float(q_chg[-1] / 60.0) if q_chg.size else 0.0


def block_cc_discharge_capacity(df: pd.DataFrame,
                                i_min: float = -0.1,
                                tol: float = 0.05) -> float:
    """块内最大单段恒流放电容量 [Ah]（参考放电的稳健代理）。

    找 |I − median(段)| ≤ tol·|median| 的最长负电流段，积分其 Ah。
    对 RPT 块返回参考放电容量；对脉冲/杂块返回较小值。
    """
    t, i = _sanitize_mi(df["TestTime"].to_numpy(dtype=float),
                        df["Amps"].to_numpy(dtype=float))
    if t.size < 3:
        return 0.0
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
            q = float(np.sum(0.5 * (ii[1:] + ii[:-1])
                          * np.where(np.diff(tt) > DT_MAX_MIN, 0.0, np.maximum(np.diff(tt), 0.0))) / 60.0)
            best_q = max(best_q, -q)
    return best_q


def block_cc_discharge_segment(df: pd.DataFrame,
                               i_min: float = -0.1,
                               tol: float = 0.05) -> dict:
    """最大 CC 放电段 + 定位（合并块 Profile+RPT 的容量点定位用）。

    返回 {q, end_idx, t_end, ah_tp_at_end, ah_dis_at_end}：q 与
    block_cc_discharge_capacity 同口径；end_idx 为段内最后一个样本
    下标；ah_tp_at_end = ∫|I|dt/60 到该点（含之前所有段）；
    ah_dis_at_end = ∫max(−I,0)dt/60 到该点（EFC 轴用放电口径）。
    """
    t, i = _sanitize_mi(df["TestTime"].to_numpy(dtype=float),
                        df["Amps"].to_numpy(dtype=float))
    if t.size < 3:
        return {"q": 0.0, "end_idx": -1, "t_end": float("nan"),
                "ah_tp_at_end": 0.0, "ah_dis_at_end": 0.0}
    dt = np.where(np.diff(t) > DT_MAX_MIN, 0.0, np.maximum(np.diff(t), 0.0))
    i_mid = 0.5 * (i[1:] + i[:-1])
    tp_cum = np.cumsum(np.abs(i_mid) * dt) / 60.0
    dis_cum = np.cumsum(np.maximum(-i_mid, 0.0) * dt) / 60.0
    dis = i < i_min
    best_q, best_end = 0.0, -1
    if dis.any():
        idx = np.nonzero(dis)[0]
        splits = np.nonzero(np.diff(idx) > 1)[0]
        for seg in np.split(idx, splits + 1):
            if seg.size < 3:
                continue
            i_seg = i[seg]
            i_med = float(np.median(i_seg))
            if i_med >= i_min:
                continue
            cc = np.abs(i_seg - i_med) <= tol * abs(i_med) + 1e-9
            sub_idx = np.nonzero(cc)[0]
            if sub_idx.size < 3:
                continue
            s2 = np.nonzero(np.diff(sub_idx) > 1)[0]
            for sub in np.split(sub_idx, s2 + 1):
                if sub.size < 3:
                    continue
                tt = t[seg[sub]]
                ii = i[seg[sub]]
                q = float(np.sum(0.5 * (ii[1:] + ii[:-1])
                                * np.where(np.diff(tt) > DT_MAX_MIN, 0.0, np.maximum(np.diff(tt), 0.0)))
                          / 60.0)
                if -q > best_q:
                    best_q = -q
                    best_end = int(seg[sub[-1]])
    return {
        "q": float(best_q),
        "end_idx": best_end,
        "t_end": float(t[best_end]) if best_end >= 0 else float("nan"),
        "ah_tp_at_end": float(tp_cum[best_end - 1]) if best_end > 0 else 0.0,
        "ah_dis_at_end": float(dis_cum[best_end - 1]) if best_end > 0 else 0.0,
    }


def block_cc_discharge_curve(df: pd.DataFrame,
                             i_min: float = -0.1,
                             tol: float = 0.05) -> dict:
    """最大 CC 放电段的 V(q) 曲线（V(Q) 两模式分解输入）。

    选段口径与 block_cc_discharge_capacity 一致（段内 |I−median| ≤
    tol·|median| 的最长子段中取 q 最大者）；对合并块 Profile+RPT 返回
    其中 RPT 参考放电（C/25 ≈ −0.125 A）的 V(q)。返回：
    {q, v, i_med, q_end, n_pts}——q 为段内容量 [Ah]（单调化、自 0
    起），v 为对应电压 [V]；q 与 v 等长。
    """
    t = df["TestTime"].to_numpy(dtype=float)
    i = df["Amps"].to_numpy(dtype=float)
    v = df["Volts"].to_numpy(dtype=float)
    ok = _ok_mask(t, i, v)
    t, i, v = t[ok], i[ok], v[ok]
    if t.size < 3:
        return {"q": np.empty(0), "v": np.empty(0), "i_med": float("nan"),
                "q_end": 0.0, "n_pts": 0}
    dis = i < i_min
    best: tuple | None = None
    best_q = 0.0
    if dis.any():
        idx = np.nonzero(dis)[0]
        splits = np.nonzero(np.diff(idx) > 1)[0]
        for seg in np.split(idx, splits + 1):
            if seg.size < 3:
                continue
            i_seg = i[seg]
            i_med = float(np.median(i_seg))
            if i_med >= i_min:
                continue
            cc = np.abs(i_seg - i_med) <= tol * abs(i_med) + 1e-9
            sub_idx = np.nonzero(cc)[0]
            if sub_idx.size < 3:
                continue
            s2 = np.nonzero(np.diff(sub_idx) > 1)[0]
            for sub in np.split(sub_idx, s2 + 1):
                if sub.size < 3:
                    continue
                g = seg[sub]
                tt, ii, vv = t[g], i[g], v[g]
                dt = np.where(np.diff(tt) > DT_MAX_MIN, 0.0,
                              np.maximum(np.diff(tt), 0.0))
                q_inc = np.maximum(-(0.5 * (ii[1:] + ii[:-1]) * dt), 0.0)
                q_cum = np.concatenate([[0.0], np.cumsum(q_inc)]) / 60.0
                q_end = float(q_cum[-1])
                if q_end > best_q:
                    best_q = q_end
                    best = (q_cum, vv, i_med)
    if best is None:
        return {"q": np.empty(0), "v": np.empty(0), "i_med": float("nan"),
                "q_end": 0.0, "n_pts": 0}
    q_cum, vv, i_med = best
    q_cum = np.maximum.accumulate(q_cum)  # 容量列偶发毛刺
    return {"q": q_cum, "v": np.asarray(vv, dtype=float), "i_med": i_med,
            "q_end": float(q_cum[-1]), "n_pts": int(q_cum.size)}


def block_rpt_curves(df: pd.DataFrame,
                     i_ref: float = -0.125,
                     i_tol: float = 0.03,
                     q_min: float = 0.5,
                     min_pts: int = 500) -> list[dict]:
    """块内全部参考放电段（i_med ≈ i_ref）的 V(q) 曲线（模式轨迹用）。

    合并块 Profile+RPTx-RPTy 内含多个 C/25（−0.125 A）RPT 参考放电
    段（细采样数千~数万点）——每段是一次不同深度的 V(q) 观测。筛选：
    |i_med − i_ref| ≤ i_tol、段容量 ≥ q_min Ah、点数 ≥ min_pts。
    返回按段内时间顺序的列表：{q, v, q_end, i_med, n_pts}。
    """
    t = df["TestTime"].to_numpy(dtype=float)
    i = df["Amps"].to_numpy(dtype=float)
    v = df["Volts"].to_numpy(dtype=float)
    ok = _ok_mask(t, i, v)
    t, i, v = t[ok], i[ok], v[ok]
    if t.size < 3:
        return []
    out: list[dict] = []
    dis = i < -0.1
    if not dis.any():
        return out
    idx = np.nonzero(dis)[0]
    splits = np.nonzero(np.diff(idx) > 1)[0]
    for seg in np.split(idx, splits + 1):
        if seg.size < min_pts:
            continue
        i_seg = i[seg]
        i_med = float(np.median(i_seg))
        if abs(i_med - i_ref) > i_tol:
            continue
        g = seg
        tt, ii, vv = t[g], i[g], v[g]
        dt = np.where(np.diff(tt) > DT_MAX_MIN, 0.0,
                      np.maximum(np.diff(tt), 0.0))
        q_inc = np.maximum(-(0.5 * (ii[1:] + ii[:-1]) * dt), 0.0)
        q_cum = np.maximum.accumulate(
            np.concatenate([[0.0], np.cumsum(q_inc)]) / 60.0)
        q_end = float(q_cum[-1])
        if q_end < q_min:
            continue
        out.append({"q": q_cum, "v": np.asarray(vv, dtype=float),
                    "q_end": q_end, "i_med": i_med, "n_pts": int(g.size)})
    return out


def block_throughput_stats(mat_bytes: bytes) -> dict:
    """块吞吐/时长/参考容量统计（EFC 轨迹构件，v1.14）。

    双估计器：∫|I|dt/60（主） vs 分步 Amphr 极差和（核对）——
    Amphr 步内累积、步间重置（v1.13 坑）；TestTime 分钟制
    （v1.12 坑十五）。q_cc 定位到块内结束时点，合并块
    （Profile+多 RPT）的容量点由此对齐吞吐轴。
    """
    df = load_maccor_block(mat_bytes)
    t, i = _sanitize_mi(df["TestTime"].to_numpy(dtype=float),
                        df["Amps"].to_numpy(dtype=float))
    if t.size < 2:
        return {"n_rows": 0, "t_min": 0.0, "ah_tp": 0.0, "ah_chg": 0.0,
                "ah_dis": 0.0, "ah_amphr": 0.0, "q_cc": 0.0,
                "t_cc_end_min": float("nan"), "ah_tp_at_cc": 0.0,
                "ah_dis_at_cc": 0.0, "rest_min_total": 0.0, "rest_n": 0,
                "rest_med_min": 0.0, "rest_max_min": 0.0,
                "ext_rest_min": 0.0, "ext_rest_n": 0,
                "ext_rest_max_min": 0.0}
    dt = np.where(np.diff(t) > DT_MAX_MIN, 0.0, np.maximum(np.diff(t), 0.0))
    i_mid = 0.5 * (i[1:] + i[:-1])
    ah_chg = float(np.sum(np.maximum(i_mid, 0.0) * dt) / 60.0)
    ah_dis = float(np.sum(np.maximum(-i_mid, 0.0) * dt) / 60.0)
    # 双估计器：步内 Amphr 极差求和
    ah_amphr = 0.0
    if "Step" in df.columns and "Amphr" in df.columns:
        st = df["Step"].to_numpy(dtype=float)
        am = df["Amphr"].to_numpy(dtype=float)
        if st.size > 1:
            cuts = np.nonzero(np.diff(st) != 0)[0] + 1
            for r in np.split(np.arange(st.size), cuts):
                if r.size:
                    ah_amphr += float(np.nanmax(am[r]) - np.nanmin(am[r]))
        else:
            ah_amphr = float(np.nanmax(am) - np.nanmin(am)) if am.size else 0.0
    seg = block_cc_discharge_segment(df)
    del df
    durs = rest_run_durations(t, i)
    ext = durs[durs >= REST_EXT_MIN]
    return {
        "n_rows": int(t.size),
        "t_min": float(t[-1] - t[0]) if t.size else 0.0,
        "ah_tp": ah_chg + ah_dis,
        "ah_chg": ah_chg,
        "ah_dis": ah_dis,
        "ah_amphr": float(ah_amphr),
        "q_cc": seg["q"],
        "t_cc_end_min": seg["t_end"],
        "ah_tp_at_cc": seg["ah_tp_at_end"],
        "ah_dis_at_cc": seg["ah_dis_at_end"],
        "rest_min_total": float(durs.sum()) if durs.size else 0.0,
        "rest_n": int(durs.size),
        "rest_med_min": float(np.median(durs)) if durs.size else 0.0,
        "rest_max_min": float(durs.max()) if durs.size else 0.0,
        "ext_rest_min": float(ext.sum()) if ext.size else 0.0,
        "ext_rest_n": int(ext.size),
        "ext_rest_max_min": float(ext.max()) if ext.size else 0.0,
    }


# ---------------------------------------------------------------------------
# 文件名解析
# ---------------------------------------------------------------------------

_EIS_NAME_RE = re.compile(
    r"EIS/(BoL|MoL|EoL)/Group(\d+)_(?:BoL|MoL|EoL)_Cell(\d+)_SOC(\d+)\.mat")

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
