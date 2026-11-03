"""电极级半电池耦合——shift/shape ↔ LLI/LAM 映射的第一性原理验证。

v1.10 温度轴负结果的两读法（映射失效 / 高温 LAM 加速）数据侧
不可区分；Oxford PathDependent 的 Half_Cell 数据（电极半电池
U(x) + 全池 POCV，Raj & Howey 2020）可以做**几何层裁决**：

1. 耦合重建：V_full(q) = U_p(x_p) − U_n(x_n)（锂守恒 + N/P 比），
   对 FC_POCV 实测曲线拟合 → 电极耦合是否自洽；
2. 合成损伤注入：纯 LLI（可循环锂↓）/ LAM-PE（正极容量↓）/
   LAM-NE（负极容量↓）→ 生成退化 V(Q)，跑 degrade_modes 两模式
   分解 → **恢复矩阵**（注入型 vs 恢复 share_shape）；
3. 恢复矩阵对角占优 → 映射几何成立（shape-like 确为 LAM 代理、
   shift-like 确为 LLI 代理）→ 温度轴 35°C shape↑ 支持「LAM 加速」
   读法；交叉污染严重 → 「映射失效」读法得权。

⚠️ 边界：① 半电池是 LCO/石墨 Kokam（Raj & Howey），SNL 是
LFP/NMC/NCA 18650——**几何验证不等于化学体系统一**，LFP 平台型
曲线的可辨识性需单独用真实 RPT 合成扰动检验（本模块提供）；
② 合成损伤是 OCV 级（准平衡），不含动力学/阻抗贡献；③ 恢复
矩阵只能裁决「映射方向」，不能点名材料位点（仍需物理 post-mortem）。

数据解码
--------
Oxford Half_Cell/EIS 的 .mat 是 MATLAB v5 **MCOS table 对象**
（scipy 只能拿到 opaque 壳）；表数据在隐藏变量
``__function_workspace__`` 的 MAT 元素流里（可能 zlib 压缩）。
本模块带一个小型元素流解码器（parse_mat_elements）。
列语义按文件族标定（电荷守恒 I·t≈Q 校验）：

- HC 文件（半电池）：V, Q_Ah, I_A, t_min, cycle, soc_pct
- FC 文件（全池 POCV）：t_s, V, I_mA, Q_mAh, soc_pct
"""
from __future__ import annotations

import io
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# MAT v5 元素流解码（MCOS __function_workspace__ 负载）
# ---------------------------------------------------------------------------

_MI_TO_NP = {
    1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
    5: np.int32, 6: np.uint32, 7: np.float32, 9: np.float64,
    12: np.int64, 13: np.uint64,
}
_MI_NAME = {
    1: "i8", 2: "u8", 3: "i16", 4: "u16", 5: "i32", 6: "u32",
    7: "single", 9: "double", 12: "i64", 13: "u64",
    14: "matrix", 15: "compressed", 16: "utf8", 17: "utf16", 18: "utf32",
}


def parse_mat_elements(buf: bytes, off: int = 0, end: int | None = None) -> list:
    """递归解析 MAT v5 元素流 → [(type_name, depth, ndarray|None), ...]。

    处理小数据元素（small data element）、miCOMPRESSED（zlib）、
    miMATRIX（递归进负载）。返回扁平列表供上层做启发式抽取。
    """
    end = len(buf) if end is None else end
    out: list = []
    while off + 8 <= end:
        tag = struct.unpack_from("<I", buf, off)[0]
        small = (tag >> 16) & 0xFFFF
        if small:
            typ, ln, data_off = tag & 0xFFFF, small, off + 4
            nxt = off + 8
        else:
            typ = tag
            ln = struct.unpack_from("<I", buf, off + 4)[0]
            data_off = off + 8
            nxt = (data_off + ln + 7) & ~7
        if typ == 15:                      # compressed
            try:
                dec = zlib.decompress(buf[data_off:data_off + ln])
                out.extend(parse_mat_elements(dec))
            except Exception:
                out.append(("zlib-err", 0, None))
        elif typ == 14:                    # matrix → 递归
            out.extend(parse_mat_elements(buf, data_off, data_off + ln))
        elif typ in _MI_TO_NP:
            dt = np.dtype(_MI_TO_NP[typ])
            a = np.frombuffer(buf, dtype=dt,
                              count=ln // dt.itemsize, offset=data_off)
            out.append((_MI_NAME.get(typ, str(typ)), 0, a))
        else:
            out.append((str(typ), 0, None))
        off = nxt
    return out


def load_mcos_columns(mat_bytes: bytes, min_len: int = 100) -> list[np.ndarray]:
    """MCOS table 的 .mat 字节 → 浮点列数组列表（按出现顺序）。

    scipy 读不出 MATLAB table；这里从 __function_workspace__ 元素流
    启发式抽取所有 n>=min_len 的 double/single 列。
    """
    import scipy.io as sio

    m = sio.loadmat(io.BytesIO(mat_bytes), struct_as_record=False,
                    squeeze_me=True)
    if "__function_workspace__" not in m:
        raise ValueError("no __function_workspace__ (not an MCOS object?)")
    fw = np.asarray(m["__function_workspace__"])
    blob = fw.tobytes() if fw.dtype == np.uint8 else bytes(fw)
    cols = []
    for typ, _depth, a in parse_mat_elements(blob):
        if typ in ("double", "single") and a is not None and a.size >= min_len:
            cols.append(np.asarray(a, dtype=float))
    return cols


# ---------------------------------------------------------------------------
# Oxford Half_Cell 加载（列语义标定）
# ---------------------------------------------------------------------------

HC_COLS = ["V", "Q_Ah", "I_A", "t_min", "cycle", "soc_pct"]
FC_COLS = ["t_s", "V", "I_mA", "Q_mAh", "soc_pct"]

# 文件 stem → 语义键（精确匹配——'CH_HC_anode' 是 'DCH_HC_anode'
# 的子串，子串匹配会张冠李戴，v1.11 坑）
_HC_FILES = {
    "CH_HC_anode": "anode_chg",      # 负极脱锂（V 升）
    "DCH_HC_anode": "anode_dch",     # 负极嵌锂（V 降）
    "CH_HC_cathode": "cathode_chg",  # 正极脱锂（V 升）
    "DHC_HC_cathode": "cathode_dch", # 正极嵌锂（V 降，官方文件名拼写 DHC）
    "CH_FC_POCV": "full_chg",
    "DCH_FC_POCV": "full_dch",
}


def load_oxford_halfcell(zip_path: str | Path) -> dict[str, pd.DataFrame]:
    """Half_Cell.zip → {anode_chg, anode_dch, cathode_chg, cathode_dch,
    full_chg, full_dch: DataFrame}。"""
    out: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".mat"):
                continue
            stem = Path(name).stem
            key = _HC_FILES.get(stem)
            if key is None:
                continue
            cols = load_mcos_columns(z.read(name))
            is_fc = "FC_POCV" in name
            names = FC_COLS if is_fc else HC_COLS
            if len(cols) < len(names):
                raise ValueError(f"{name}: {len(cols)} cols < {len(names)}")
            df = pd.DataFrame({n: c for n, c in zip(names, cols)})
            if is_fc:
                df["Q_Ah"] = df["Q_mAh"] / 1000.0
                df["I_A"] = df["I_mA"] / 1000.0
            out[key] = df
    return out


# ---------------------------------------------------------------------------
# 电极耦合：V(q) = U_p(x_p) − U_n(x_n)
# ---------------------------------------------------------------------------


def electrode_ux(df: pd.DataFrame, lithiation: bool) -> tuple[np.ndarray, np.ndarray]:
    """半电池放电段 → 归一化 (x, U)。x=0 起点、x=1 终点（按容量）。"""
    q = np.asarray(df["Q_Ah"], float)
    v = np.asarray(df["V"], float)
    ok = np.isfinite(q) & np.isfinite(v)
    q, v = q[ok], v[ok]
    o = np.argsort(q)
    q, v = q[o], v[o]
    # 去重（插值需严格递增）
    uniq = np.concatenate([[True], np.diff(q) > 1e-12])
    q, v = q[uniq], v[uniq]
    x = (q - q[0]) / (q[-1] - q[0] + 1e-15)
    if not lithiation:          # 脱锂曲线反向，使 x=嵌锂度
        x = 1.0 - x
        order = np.argsort(x)
        x, v = x[order], v[order]
    return x, v


@dataclass
class CouplingFit:
    """耦合重建结果。

    ⚠️ 可辨识性：x_p0 与 q_use 稳；(x_n0, N/P) 沿山脊弱可辨识
    （V(q) 只约束轨迹组合 x_n0 − q/r）——引用时以 RMSE/q_use 为准。
    """
    x_p0: float          # 100% SOC 正极嵌锂度
    x_n0: float          # 100% SOC 负极嵌锂度
    np_ratio: float      # 负极/正极容量比
    rmse_v: float        # POCV 重建 RMSE [V]
    q_meas_ah: float
    q_use_norm: float    # 模型归一可用容量（q_p_max=1 单位）

    def to_dict(self) -> dict:
        return {
            "x_p0": self.x_p0, "x_n0": self.x_n0,
            "np_ratio": self.np_ratio, "rmse_v": self.rmse_v,
            "q_meas_ah": self.q_meas_ah, "q_use_norm": self.q_use_norm,
        }


def _build_vq(xp, Up, xn, Un, q_p_max, q_n_max, x_p0, x_n0, n_grid=400):
    """放电段 V(q)：q 从 0 到可用容量。返回 (q, v)（q 单位=正极容量单位）。"""
    q_use = min(q_p_max * (1.0 - x_p0), q_n_max * x_n0)
    if q_use <= 0:
        raise ValueError("no usable capacity")
    q = np.linspace(0.0, q_use, n_grid)
    x_p = x_p0 + q / q_p_max
    x_n = x_n0 - q / q_n_max
    v = np.interp(x_p, xp, Up) - np.interp(x_n, xn, Un)
    return q, v


def fit_coupling(
    xp: np.ndarray, Up: np.ndarray,
    xn: np.ndarray, Un: np.ndarray,
    pocv_q: np.ndarray, pocv_v: np.ndarray,
    n_grid: int = 400,
) -> tuple[CouplingFit, tuple[np.ndarray, np.ndarray]]:
    """网格搜索 (x_p0, x_n0, N/P) 使重建 V(q/Q_use) 拟合 POCV 归一化曲线。

    q_p_max 归一为 1；绝对容量由 POCV 实测 Q 标定。
    """
    q_meas = float(np.nanmax(pocv_q))
    # 归一化目标曲线
    ok = np.isfinite(pocv_q) & np.isfinite(pocv_v)
    qq, vv = np.asarray(pocv_q)[ok], np.asarray(pocv_v)[ok]
    o = np.argsort(qq)
    qq, vv = qq[o], vv[o]
    qq_n = qq / (qq[-1] + 1e-15)
    v_tgt = np.interp(np.linspace(0, 1, n_grid), qq_n, vv)

    best = None
    for x_p0 in np.linspace(0.02, 0.55, 12):
        for x_n0 in np.linspace(0.50, 0.98, 12):
            for r in np.linspace(1.0, 1.6, 7):
                try:
                    q, v = _build_vq(xp, Up, xn, Un, 1.0, r, x_p0, x_n0, n_grid)
                except ValueError:
                    continue
                q_n = q / (q[-1] + 1e-15)
                v_mod = np.interp(np.linspace(0, 1, n_grid), q_n, v)
                rmse = float(np.sqrt(np.mean((v_mod - v_tgt) ** 2)))
                if best is None or rmse < best[0]:
                    best = (rmse, x_p0, x_n0, r, q[-1])
    if best is None:
        raise ValueError("coupling fit failed")
    rmse, x_p0, x_n0, r, q_use = best
    fit = CouplingFit(x_p0=x_p0, x_n0=x_n0, np_ratio=r, rmse_v=rmse,
                      q_meas_ah=q_meas, q_use_norm=float(q_use))
    q, v = _build_vq(xp, Up, xn, Un, 1.0, r, x_p0, x_n0, n_grid)
    return fit, (q / (q[-1] + 1e-15) * q_meas, v)


# ---------------------------------------------------------------------------
# 合成损伤注入
# ---------------------------------------------------------------------------

DAMAGE_KINDS = ("lli", "lam_pe", "lam_ne")


def inject_damage(
    xp: np.ndarray, Up: np.ndarray,
    xn: np.ndarray, Un: np.ndarray,
    fit: CouplingFit, kind: str, q_loss_frac: float, n_grid: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """按损伤类型退化耦合模型 → 退化放电 V(q)（绝对容量单位 Ah）。

    - lli:    可循环锂 λ 减少——起点沿耦合线滑动
      （x_p0+a, x_n0−a/r），V(q)=V_BoL(q+a) 纯平移截断 → shift
    - lam_pe: 正极容量 1→1−a（轨迹线改变 → 形变 → shape）
    - lam_ne: 负极容量 r→r(1−a)（轨迹线改变 → 形变 → shape）
    目标容量损失 q_loss_frac·Q_BoL。
    """
    if kind not in DAMAGE_KINDS:
        raise ValueError(f"unknown kind {kind}")
    x_p0, x_n0, r = fit.x_p0, fit.x_n0, fit.np_ratio
    q_bol = min(1.0 * (1 - x_p0), r * x_n0)

    def q_use_of(a: float) -> float:
        if kind == "lli":
            return min(1.0 * (1 - x_p0 - a), r * x_n0 - a) \
                if a < min(1 - x_p0, r * x_n0) else 0.0
        if kind == "lam_pe":
            return min((1.0 - a) * (1 - x_p0), r * x_n0)
        return min(1.0 * (1 - x_p0), r * (1.0 - a) * x_n0)

    scale = fit.q_meas_ah / q_bol        # 正极归一容量 → Ah
    if kind == "lli":
        # 纯 shift：a 即容量损失（闭式，不必二分）
        a = q_bol * q_loss_frac
        q, v = _build_vq(xp, Up, xn, Un, 1.0, r, x_p0 + a, x_n0 - a / r,
                         n_grid)
    else:
        # 二分求 a 使容量损失达目标
        target = q_bol * (1.0 - q_loss_frac)
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if q_use_of(mid) > target:
                lo = mid
            else:
                hi = mid
        a = 0.5 * (lo + hi)
        if kind == "lam_pe":
            # 正极容量 1→1−a，x_p0 不变（同截止电压→同化学计量）
            q, v = _build_vq(xp, Up, xn, Un, 1.0 - a, r, x_p0, x_n0, n_grid)
        else:
            q, v = _build_vq(xp, Up, xn, Un, 1.0, r * (1.0 - a), x_p0, x_n0,
                             n_grid)
    return q * scale, v


def recovery_matrix(
    xp: np.ndarray, Up: np.ndarray,
    xn: np.ndarray, Un: np.ndarray,
    fit: CouplingFit,
    q_loss_frac: float = 0.10, n_grid: int = 400,
) -> pd.DataFrame:
    """三种损伤 → 两模式分解 → 恢复表（share_shape 等）。"""
    from bmthermal.degrade_modes import mode_split

    q_bol, v_bol = _build_vq(xp, Up, xn, Un, 1.0, fit.np_ratio,
                             fit.x_p0, fit.x_n0, n_grid)
    scale = fit.q_meas_ah / float(q_bol[-1])
    q_bol = q_bol * scale
    rows = []
    for kind in DAMAGE_KINDS:
        q, v = inject_damage(xp, Up, xn, Un, fit, kind, q_loss_frac, n_grid)
        ms = mode_split(q_bol, v_bol, q, v, cycle_number=0)
        rows.append({
            "injected": kind,
            "q_loss_frac": q_loss_frac,
            "loss_eod_ah": ms.loss_eod,
            "q_shift_ah": ms.q_shift,
            "q_shape_ah": ms.q_shape,
            "share_shape": ms.share_shape,
            "r2": ms.r2,
            "boundary": ms.boundary,
            "low_quality": ms.low_quality,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SNL 真实曲线几何灵敏度（LFP 平台型可辨识性）
# ---------------------------------------------------------------------------


def synthetic_shift(q: np.ndarray, v: np.ndarray, loss_frac: float):
    """纯 shift：L(V)=δ 常数（q 轴左移常量，V 形状不变）。"""
    delta = loss_frac * float(q[-1])
    keep = q >= delta
    return q[keep] - delta, v[keep]


def synthetic_shape(q: np.ndarray, v: np.ndarray, loss_frac: float):
    """纯 shape：L(V)=f·Q₀(V)（各电压点容量等比例损失 → γ=f, δ=0）。"""
    return q * (1.0 - loss_frac), v.copy()


def geometry_sensitivity(
    q: np.ndarray, v: np.ndarray, loss_frac: float = 0.10,
) -> dict:
    """真实 BoL 曲线上合成 δ/γ 扰动 → 估计器恢复率。

    与 inject_damage 的差别：不走电极耦合，直接检验 mode_split
    在该曲线几何（平台型 LFP / 斜坡型 NMC）下能否分离两模式。
    """
    from bmthermal.degrade_modes import mode_split

    # 纯 shift：L(V)=δ 常数 → share_shape≈0
    q_s, v_s = synthetic_shift(q, v, loss_frac)
    ms_s = mode_split(q, v, q_s, v_s)
    # 纯 shape：L(V)=f·Q0(V) → share_shape≈1
    q_h, v_h = synthetic_shape(q, v, loss_frac)
    ms_h = mode_split(q, v, q_h, v_h)
    return {
        "loss_frac": loss_frac,
        "shift_injected": {"share_shape": ms_s.share_shape,
                           "q_shift": ms_s.q_shift, "q_shape": ms_s.q_shape,
                           "r2": ms_s.r2, "low_quality": ms_s.low_quality,
                           "basis_corr": ms_s.basis_corr},
        "shape_injected": {"share_shape": ms_h.share_shape,
                           "q_shift": ms_h.q_shift, "q_shape": ms_h.q_shape,
                           "r2": ms_h.r2, "low_quality": ms_h.low_quality,
                           "basis_corr": ms_h.basis_corr},
    }
