"""无损退化模式分解——RPT 曲线位移/形变两模式（post-mortem 的数据侧替身）。

开放项②「post-mortem 材料位点」的数据侧收口
--------------------------------------------
物理 post-mortem（拆解）能点名材料位点，但需要硬件；SNL 逐循环
时序里含周期性 **0.5C 参考性能测试（RPT）段**（细采样 ~10 s、
放电段 >50 点，每 ~200 循环 4–8 条，全化学体系同口径）——统一
0.5C 放电 V(Q) 曲线随寿命的演化，可做无损替身：

    L(V) = Q₀(V) − Q_k(V) = δ + γ·Q₀(V) + 残差

- δ（shift-like，LLI 代理）：损失在各锚点电压上近似均匀——
  可循环锂减少把整条 V(Q) 沿容量轴平移截断；
- γ·Q₀(V)（shape-like，LAM 代理）：损失与已有容量位置成比例
  ——活性材料损失/颗粒开裂改变曲线形变（平台/斜率）。

⚠️ 口径警示（与 mech_attribution「形状归属非物理份额」同源）：
两模式是**曲线几何归属**，不是拆解级材料点名。LLI/LAM 标签是
文献惯例映射（Dubarry et al. 2020, eTransportation 综述；Birkl
et al. 2017 诊断分解），不是实测位点。与 v1.9 锚定超额分解的
「机械份额」对拍是**独立估计器互证**，不是同口径重复计算。

RPT 识别口径
------------
- 放电段（I < −0.05 A）点数 ≥ min_pts（默认 50）；
- 平均放电电流 ≈ rate_c·nominal（默认 0.5C，容差 ±25%）；
- 同签名连续条目去重（CS2/SNL 重复导出坑，与
  ``snl_aging.summarize_cycles`` 同款签名）。

分解质量旗标
------------
- basis 两列（1, Q₀）共线性（corr）过高 → 份额是形状归属非
  物理份额（v1.9 教训），只用跨组对比；
- r² / 误差界 |δ+γ·Q₀_eod − loss_eod|/loss_eod 超阈值 → 该 RPT
  点标 low_quality，聚合时剔除或降权。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 放电段电流阈值 [A]（忽略静置/充电）
I_DISCHG_MAX = -0.05

# RPT 细采样判定
RPT_MIN_PTS = 50
RPT_RATE_TOL = 0.25

# 损失剖面锚点：取参考曲线 10–90% DOD 内电压窗（避开膝区/头部）
ANCHOR_LO_FRAC = 0.10
ANCHOR_HI_FRAC = 0.90
N_ANCHORS = 11

# 分解质量阈值
QSUM_TOL = 0.15          # |δ+γ·Q0_eod − loss|/loss
R2_MIN = 0.80
BASIS_CORR_WARN = 0.995  # 共线警戒（v1.9 教训）


def _dedup_sign(cycles: list, ) -> list:
    """同签名连续条目去重（重复导出坑）。"""
    out, last_sig = [], None
    for c in cycles:
        t = np.asarray(c["time_in_s"], dtype=float)
        dis = np.asarray(c["discharge_capacity_in_Ah"], dtype=float)
        if dis.size == 0 or t.size < 2:
            continue
        sig = (round(float(t[-1] - t[0]), 3), t.size,
               round(float(np.max(dis)), 5))
        if sig == last_sig:
            continue
        last_sig = sig
        out.append(c)
    return out


def identify_rpt_segments(
    cycle_data: list,
    nominal_ah: float,
    rate_c: float = 0.5,
    tol: float = RPT_RATE_TOL,
    min_pts: int = RPT_MIN_PTS,
) -> list[dict]:
    """逐循环列表 → RPT 放电段（细采样 0.5C）轨迹点列表。

    返回元素：{cycle_number, q (Ah, 段内放电容量), v, t, temp, q_end,
    i_mean}。q 单调化（np.maximum.accumulate，容量列偶发毛刺）。
    """
    out = []
    for c in _dedup_sign(cycle_data):
        I = np.asarray(c["current_in_A"], dtype=float)
        V = np.asarray(c["voltage_in_V"], dtype=float)
        t = np.asarray(c["time_in_s"], dtype=float)
        dis = np.asarray(c["discharge_capacity_in_Ah"], dtype=float)
        temp = np.asarray(c["temperature_in_C"], dtype=float)
        m = I < I_DISCHG_MAX
        if int(m.sum()) < min_pts:
            continue
        i_mean = float(np.mean(np.abs(I[m])))
        if not (rate_c * (1 - tol) <= i_mean / nominal_ah <= rate_c * (1 + tol)):
            continue
        q = dis[m] - dis[m][0]
        q = np.maximum.accumulate(q)
        if q[-1] < 0.2 * nominal_ah:
            continue
        out.append({
            "cycle_number": int(c["cycle_number"]),
            "q": q, "v": V[m], "t": t[m], "temp": temp[m],
            "q_end": float(q[-1]), "i_mean": i_mean,
        })
    return out


def resample_discharge(seg: dict, n_grid: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """RPT 放电段 → 均匀容量网格上的 V(q)。"""
    q, v = seg["q"], seg["v"]
    q_grid = np.linspace(0.0, float(q[-1]), n_grid)
    v_grid = np.interp(q_grid, q, v)
    return q_grid, v_grid


def _q_at_v(v_grid: np.ndarray, q_grid: np.ndarray, v_anchor: float) -> float:
    """单调放电 V(q) 的逆：给定电压 → 已放容量。"""
    # V 随 q 递减；np.interp 需升序 xp
    return float(np.interp(v_anchor, v_grid[::-1], q_grid[::-1]))


def anchor_profile(
    q_ref: np.ndarray, v_ref: np.ndarray,
    q: np.ndarray, v: np.ndarray,
    n_anchors: int = N_ANCHORS,
) -> dict:
    """参考 vs 当前 RPT 的锚点损失剖面。

    返回 q0_anchors（Q₀(V_a)）、loss（L(V_a)）、v_anchors、
    loss_eod（端点容量损失）、q_end_ref。
    """
    q_end_ref = float(q_ref[-1])
    # 锚点电压窗 = 参考曲线 10–90% DOD 处的电压（V(q) 直接插值——
    # 坑：不要拿容量值走 _q_at_v 反查，那会把容量当电压）
    v10 = float(np.interp(ANCHOR_LO_FRAC * q_end_ref, q_ref, v_ref))
    v90 = float(np.interp(ANCHOR_HI_FRAC * q_end_ref, q_ref, v_ref))
    # V(q) 递减 → v90 < v10
    v_anchors = np.linspace(v90, v10, n_anchors)
    q0 = np.array([_q_at_v(v_ref, q_ref, a) for a in v_anchors])
    qk = np.array([_q_at_v(v, q, a) for a in v_anchors])
    loss = q0 - qk
    return {
        "v_anchors": v_anchors, "q0_anchors": q0, "loss": loss,
        "loss_eod": q_end_ref - float(q[-1]), "q_end_ref": q_end_ref,
    }


@dataclass
class ModeSplit:
    """两模式分解结果（单个 RPT 点 vs 参考）。"""

    cycle_number: int
    loss_eod: float
    q_shift: float          # δ（Ah，shift-like/LLI 代理）
    q_shape: float          # γ·Q0_eod（Ah，shape-like/LAM 代理）
    share_shape: float      # q_shape/(q_shift+q_shape)；和为 1 的归一份额
    gamma: float
    r2: float
    basis_corr: float       # 常数列与 Q0 列的相关（可辨识性）
    qsum_err: float         # |q_shift+q_shape − loss_eod|/loss_eod
    low_quality: bool = False
    boundary: str = "interior"

    def to_dict(self) -> dict:
        return {
            "cycle_number": self.cycle_number, "loss_eod": self.loss_eod,
            "q_shift": self.q_shift, "q_shape": self.q_shape,
            "share_shape": self.share_shape, "gamma": self.gamma,
            "r2": self.r2, "basis_corr": self.basis_corr,
            "qsum_err": self.qsum_err, "low_quality": self.low_quality,
            "boundary": self.boundary,
        }


def _nnls2(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, str]:
    """非负最小二乘（2 列）：interior 解或边界闭式重估。

    v1.9 教训回归：截断后不可沿用联合解幅度（分量相消会虚高），
    边界分支必须对留下的那支闭式重估。
    """
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    if (coef >= 0).all():
        return coef, "interior"
    # 边界：三支候选（纯第 1 列 / 纯第 2 列 / 哪支 RSS 小用哪支）
    best, best_rss, best_name = None, np.inf, ""
    for j, name in ((0, "shift_only"), (1, "shape_only")):
        x = X[:, j]
        a = float(x @ y / (x @ x)) if x @ x > 0 else 0.0
        if a < 0:
            a = 0.0
        rss = float(np.sum((y - a * x) ** 2))
        if rss < best_rss:
            best, best_rss, best_name = np.array(
                [a, 0.0] if j == 0 else [0.0, a]), rss, name
    return best, best_name


def mode_split(
    q_ref: np.ndarray, v_ref: np.ndarray,
    q: np.ndarray, v: np.ndarray,
    cycle_number: int = 0,
) -> ModeSplit:
    """锚点损失剖面 → shift/shape 两模式分解。

    模型：L(V_a) = δ + γ·Q₀(V_a)（δ, γ ≥ 0）。份额定义在 EOD：
    q_shift = δ、q_shape = γ·Q₀_eod，归一 share_shape =
    q_shape/(q_shift+q_shape)。
    """
    prof = anchor_profile(q_ref, v_ref, q, v)
    q0, loss = prof["q0_anchors"], prof["loss"]
    X = np.column_stack([np.ones_like(q0), q0])
    coef, boundary = _nnls2(X, loss)
    delta, gamma = float(coef[0]), float(coef[1])

    pred = delta + gamma * q0
    ss_res = float(np.sum((loss - pred) ** 2))
    ss_tot = float(np.sum((loss - loss.mean()) ** 2))
    # 纯 shift 边界（v1.11 教训）：损失剖面近常数时 ss_tot≈0，r² 数值
    # 无意义（残差噪声会被放大成 r²=0.19 之类）——拟合残差<1% 损失
    # 尺度即视为完美（r²=1），否则走常规定义
    rmse = float(np.sqrt(ss_res / max(len(loss), 1)))
    loss_scale = float(abs(loss.mean())) + 1e-12
    if rmse < 0.01 * loss_scale:
        r2 = 1.0
    elif ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = 1.0

    # 可分性诊断（v1.9 教训）：Q0 在锚点窗内的变异系数——常数列与
    # Q0 列的 Pearson 相关在常数列下方差为 0 不可用，改用 CV 判可分
    basis_corr = (float(np.std(q0) / (np.mean(q0) + 1e-12))
                  if np.std(q0) > 0 else 0.0)

    q_shift = delta
    q_shape = gamma * prof["q_end_ref"]
    q_sum = q_shift + q_shape
    loss_eod = prof["loss_eod"]
    share = float(q_shape / q_sum) if q_sum > 0 else float("nan")
    qsum_err = float(abs(q_sum - loss_eod) / loss_eod) if loss_eod > 0 else float("nan")
    low = bool(
        (r2 < R2_MIN)
        or (qsum_err > QSUM_TOL)
        or (loss_eod <= 0)
        or (basis_corr < 0.05)   # Q0 在锚点窗内几乎不变 → 两模式不可分
    )
    return ModeSplit(
        cycle_number=cycle_number, loss_eod=loss_eod,
        q_shift=q_shift, q_shape=q_shape, share_shape=share,
        gamma=gamma, r2=r2, basis_corr=basis_corr,
        qsum_err=qsum_err, low_quality=low, boundary=boundary,
    )


def dqdv_peaks(
    q_grid: np.ndarray, v_grid: np.ndarray,
    smooth_win: int = 7, min_prom_frac: float = 0.05,
) -> np.ndarray:
    """均匀 q 网格 V(q) → dQ/dV 峰表 [(v_peak, height)]。

    简化实现：dQ/dV = 1/(dV/dQ)（V(q) 平滑后梯度），峰 = 局部极大
    且高出中位 min_prom_frac·range。仅用于 NMC/NCA 可辨识场景；
    LFP 平台宽、峰弱——返回空数组是常态（如实记录）。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    v = pd.Series(v_grid).rolling(smooth_win, center=True,
                                  min_periods=3).mean().to_numpy()
    dv_dq = np.gradient(v, q_grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        dqdv = np.where(np.abs(dv_dq) > 1e-9, 1.0 / dv_dq, np.nan)
    ok = np.isfinite(dqdv)
    if ok.sum() < 10:
        return np.empty((0, 2))
    vals = dqdv[ok]
    qs = q_grid[ok]
    vs = v[ok]
    prom = min_prom_frac * (np.nanmax(vals) - np.nanmin(vals))
    med = np.nanmedian(vals)
    peaks = []
    w = 5
    for i in range(w, len(vals) - w):
        win = vals[i - w:i + w + 1]
        if vals[i] >= win.max() and (vals[i] - med) > prom:
            peaks.append((float(vs[i]), float(vals[i])))
    # 简单去重（V 相距 < 20 mV 取高者）
    peaks.sort(key=lambda p: -p[1])
    kept = []
    for p in peaks:
        if all(abs(p[0] - k[0]) > 0.02 for k in kept):
            kept.append(p)
    return np.array(kept) if kept else np.empty((0, 2))


def track_peaks(ref: np.ndarray, cur: np.ndarray,
                tol_v: float = 0.05) -> dict:
    """参考/当前 dQ/dV 峰配对 → 峰位漂移与峰高比（LLI/LAM 交叉证据）。

    配对：每个参考峰找 |ΔV|<tol_v 的最近当前峰。返回 matched、
    mean_shift_v（正=峰位下移）、height_ratio（中位）。
    """
    if ref.size == 0 or cur.size == 0:
        return {"matched": 0, "mean_shift_v": float("nan"),
                "height_ratio": float("nan")}
    shifts, ratios = [], []
    for v0, h0 in ref:
        d = np.abs(cur[:, 0] - v0)
        j = int(np.argmin(d))
        if d[j] < tol_v:
            shifts.append(float(cur[j, 0] - v0))
            if h0 != 0:
                ratios.append(float(cur[j, 1] / h0))
    return {
        "matched": len(shifts),
        "mean_shift_v": float(np.mean(shifts)) if shifts else float("nan"),
        "height_ratio": float(np.median(ratios)) if ratios else float("nan"),
    }


@dataclass
class CellModes:
    """单芯 RPT 模式轨迹汇总。"""

    cell_id: str
    chem: str
    temp_c: float
    c_dis: float
    rep: str
    n_rpt: int
    traj: pd.DataFrame = field(repr=False, default=None)
    share_shape_eol: float = float("nan")
    share_shape_at_fade: float = float("nan")
    fade_at_eol: float = float("nan")
    peak_track: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id, "chem": self.chem, "temp_c": self.temp_c,
            "c_dis": self.c_dis, "rep": self.rep, "n_rpt": self.n_rpt,
            "share_shape_eol": self.share_shape_eol,
            "share_shape_at_fade": self.share_shape_at_fade,
            "fade_at_eol": self.fade_at_eol,
            "peak_matched": int(self.peak_track.get("matched", 0)),
            "peak_shift_v": float(self.peak_track.get("mean_shift_v", float("nan"))),
            "peak_hratio": float(self.peak_track.get("height_ratio", float("nan"))),
        }


def cell_mode_trajectory(
    cycle_data: list, nominal_ah: float,
    cell_id: str = "", meta: dict | None = None,
    fade_anchor: float = 0.10,
) -> CellModes:
    """单芯：RPT 序列 → 模式轨迹 + fade 锚点份额 + 峰跟踪。

    参考曲线 = 第一条 RPT。fade = 1 − q_end/q_end0（RPT 口径，
    与主老化 fade 同向）。share_shape_at_fade 在 fade 首次线性
    插值到 fade_anchor 处取（未到达取末点）。
    """
    meta = meta or {}
    segs = identify_rpt_segments(cycle_data, nominal_ah)
    cols = {"cycle_number": [], "q_end": [], "fade": [],
            "q_shift": [], "q_shape": [], "share_shape": [],
            "r2": [], "qsum_err": [], "low_quality": [], "boundary": [],
            "peak_matched": [], "peak_shift_v": [], "peak_hratio": []}
    if len(segs) < 2:
        return CellModes(
            cell_id=cell_id, chem=meta.get("chem", ""),
            temp_c=float(meta.get("temp_c", np.nan)),
            c_dis=float(meta.get("c_dis", np.nan)),
            rep=meta.get("rep", ""), n_rpt=len(segs), traj=pd.DataFrame(cols),
        )
    q_ref, v_ref = resample_discharge(segs[0])
    peaks_ref = dqdv_peaks(q_ref, v_ref)
    q0_end = segs[0]["q_end"]
    for seg in segs:
        qg, vg = resample_discharge(seg)
        sp = mode_split(q_ref, v_ref, qg, vg, cycle_number=seg["cycle_number"])
        pk = track_peaks(peaks_ref, dqdv_peaks(qg, vg))
        cols["cycle_number"].append(seg["cycle_number"])
        cols["q_end"].append(seg["q_end"])
        cols["fade"].append(1.0 - seg["q_end"] / q0_end)
        for k, v in (("q_shift", sp.q_shift), ("q_shape", sp.q_shape),
                     ("share_shape", sp.share_shape), ("r2", sp.r2),
                     ("qsum_err", sp.qsum_err)):
            cols[k].append(v)
        cols["low_quality"].append(sp.low_quality)
        cols["boundary"].append(sp.boundary)
        cols["peak_matched"].append(pk["matched"])
        cols["peak_shift_v"].append(pk["mean_shift_v"])
        cols["peak_hratio"].append(pk["height_ratio"])
    traj = pd.DataFrame(cols)
    good = traj[~traj.low_quality]
    if good.empty:
        good = traj
    # fade 锚点插值（share 随 fade 线性；未到达取末点）
    f = good["fade"].to_numpy(float)
    s = good["share_shape"].to_numpy(float)
    order = np.argsort(f)
    f, s = f[order], s[order]
    share_at = float("nan")
    hit = np.nonzero(f >= fade_anchor)[0]
    if hit.size:
        i = int(hit[0])
        if i == 0:
            share_at = float(s[0])
        else:
            w = (fade_anchor - f[i - 1]) / (f[i] - f[i - 1] + 1e-12)
            share_at = float(s[i - 1] + w * (s[i] - s[i - 1]))
    else:
        share_at = float(s[-1])
    # 峰跟踪：取末条（对参考）matched 峰
    last = traj.iloc[-1]
    peak_track = {
        "matched": int(last["peak_matched"]),
        "mean_shift_v": float(last["peak_shift_v"]),
        "height_ratio": float(last["peak_hratio"]),
    }
    return CellModes(
        cell_id=cell_id, chem=meta.get("chem", ""),
        temp_c=float(meta.get("temp_c", np.nan)),
        c_dis=float(meta.get("c_dis", np.nan)),
        rep=meta.get("rep", ""), n_rpt=len(segs), traj=traj,
        share_shape_eol=float(good["share_shape"].iloc[-1]),
        share_shape_at_fade=share_at,
        fade_at_eol=float(traj["fade"].iloc[-1]),
        peak_track=peak_track,
    )


def mode_split_trajectory(
    curves: list[dict], ref_idx: int = 0,
) -> pd.DataFrame:
    """多条 V(q) 段（dict 含 q, v, + 任意元数据列）→ 逐条 mode_split。

    与 cell_mode_trajectory 同分解核，但输入是已提取的 V(q) 曲线段
    （Oxford RPT 块的 block_cc_discharge_curve 输出），元数据（group/
    cell/step/efc/fade_pct…）原样透传到输出表。参考 = curves[ref_idx]
    （BoL 段）。fade_pct 可由调用方预计算（=1−q_end/q_bol）后透传。
    输出列：元数据 + loss_eod/q_shift/q_shape/share_shape/gamma/r2/
    basis_corr/qsum_err/low_quality/boundary/q_end。
    """
    if not curves:
        return pd.DataFrame()
    ref = curves[ref_idx]
    q_ref, v_ref = resample_discharge(
        {"q": np.asarray(ref["q"], dtype=float),
         "v": np.asarray(ref["v"], dtype=float)})
    rows = []
    for k, cur in enumerate(curves):
        qg, vg = resample_discharge(
            {"q": np.asarray(cur["q"], dtype=float),
             "v": np.asarray(cur["v"], dtype=float)})
        sp = mode_split(q_ref, v_ref, qg, vg, cycle_number=k)
        row = sp.to_dict()
        row["q_end"] = float(np.asarray(cur["q"], dtype=float)[-1]) \
            if len(cur["q"]) else 0.0
        for key, val in cur.items():
            if key not in ("q", "v"):
                row[key] = val
        rows.append(row)
    return pd.DataFrame(rows)
