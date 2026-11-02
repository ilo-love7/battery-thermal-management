"""m(c) 机理点归因——形状指纹 + 两分量混合分解。

开放项「m(c) 机理点归因」的定量版：不再只列候选机制，而是把
LFP 3C 的每 EFC 损伤升高分解为

    f(N) = a_s·N^{b_s}  +  a_m·N

两项之和——

- 亚线性项（b_s<1）：SEI 型本底，形状锚定同体系低倍率组
  （LFP 25 °C 0.5/1/2C 的 b 分布）；
- 线性项（b=1）：机械损伤型累积（颗粒开裂/接触丧失），文献
  形状指纹是近线性 + 组内散布放大（Vetter et al. 2005；Anseán
  et al. 2016 LFP）。

对 (a_s, a_m) 的拟合在固定 b_s 后是线性最小二乘（非负约束，
边界解闭式重估单分量），线性份额 = a_m·N/(a_s·N^{b_s}+a_m·N)
在实测到达的衰减深度评估；bootstrap 过组内电芯给 CI，b_s
锚点扫描（0.64/0.83/0.89）给敏感性带。

**可辨识性警示**：有限窗口内 n^{b_s} 与 n 高度共线（b=0.83 在
N∈[300,3500] 上 corr>0.99，n^{0.95}≈0.7n）——绝对份额带基依赖，
可辨识的是**跨组对比**（3C vs 对照组在锚点扫描下方向是否稳定）。
basis_corr 量化共线性；share_profile 给不同衰减深度的份额剖面
（线性份额随深度增长本身是机理信息）。

析锂护栏：SNL 全组 0.5C 恒流充电，析锂候选（Waldmann et al.
2014）优先级低——分解出的线性分量归机械损伤候选，不锁死
（post-mortem 才能点名材料位点）。

NMC 对照：NMC 3C 的 m<1 但形状指数与 1C 相同（亚线性不变）→
线性分量拟合出来应 ≈0，定量确认「m<1 是批次异质性 nuisance
而非负向机理」。

平滑口径与 ``snl_aging.first_crossing`` 同源：滚动 11 点中位
（CS2/SNL 单点毛刺防线，v1.7 教训）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# EOL 衰减水平（与 abs_life.FADE_EOL 一致）
FADE_EOL = 0.2

# b_s 锚点候选：LFP 25 °C 低倍率实测分布（0.5C/2C/1C）
B_SEI_ANCHORS = (0.64, 0.83, 0.89)
B_SEI_DEFAULT = 0.83

# 文献机制锚（形状指纹口径，非 post-mortem 点归因）
MECH_ANCHORS = {
    "sei_sublinear": {
        "shape": "亚线性 b<1（抛物线型）",
        "refs": "Vetter et al. 2005（JPS, 老化机制综述）",
    },
    "mechanical_linear": {
        "shape": "近线性 b≈1 + 组内散布放大",
        "refs": "Vetter et al. 2005；Anseán et al. 2016（LFP 开裂/接触丧失）",
    },
    "plating": {
        "shape": "低温+快充加速（b>1 或突降）",
        "refs": "Waldmann et al. 2014（析锂温度窗）",
        "guard": "SNL 全组 0.5C 充电 → 析锂候选优先级低",
    },
}


def smooth_fade(f: np.ndarray, window: int = 11) -> np.ndarray:
    """滚动中位平滑——与 snl_aging.first_crossing 同口径。"""
    f = np.asarray(f, float)
    return pd.Series(f).rolling(window, center=True, min_periods=3).median().to_numpy()


@dataclass
class ShapeFingerprint:
    """组级形状指纹。"""

    chem: str
    temp_c: float
    c_dis: float
    n_cells: int
    b_median: float
    b_mean: float
    b_std: float
    cv_fade: float          # fade 率组内变异系数（散布指纹）
    db_vs_ref: float        # b − b_ref（参考=同体系 25C 1C 组）
    ref_c: float = 1.0

    def to_dict(self) -> dict:
        return {
            "chem": self.chem, "temp_c": self.temp_c, "c_dis": self.c_dis,
            "n_cells": self.n_cells,
            "b_median": self.b_median, "b_mean": self.b_mean, "b_std": self.b_std,
            "cv_fade": self.cv_fade,
            "db_vs_ref": self.db_vs_ref, "ref_c": self.ref_c,
        }


def shape_fingerprint_group(df: pd.DataFrame, chem: str, temp_c: float,
                            c_dis: float, ref_c: float = 1.0) -> ShapeFingerprint:
    """组级形状指纹（df 为 snl_summary 行）。"""
    sub = df[(df.chem == chem) & (df.temp_c == temp_c) & (df.c_dis == c_dis)]
    ref = df[(df.chem == chem) & (df.temp_c == temp_c) & (df.c_dis == ref_c)]
    b_ref = float(ref.b_fade.median()) if len(ref) else float("nan")
    b = sub.b_fade.to_numpy(float)
    fade = sub.fade_rate_per_efc.to_numpy(float)
    return ShapeFingerprint(
        chem=chem, temp_c=temp_c, c_dis=c_dis, n_cells=len(sub),
        b_median=float(np.median(b)), b_mean=float(np.mean(b)),
        b_std=float(np.std(b, ddof=1)) if b.size > 1 else float("nan"),
        cv_fade=float(np.std(fade, ddof=1) / np.mean(fade)) if fade.size > 1 else float("nan"),
        db_vs_ref=float(np.median(b) - b_ref) if np.isfinite(b_ref) else float("nan"),
        ref_c=ref_c,
    )


@dataclass
class MixDecomp:
    """两分量分解结果（identifiability 警示见模块 docstring）。"""

    a_sei: float
    a_mech: float
    b_sei: float
    f_target: float
    n_at_target: float          # 平滑曲线到 f_target 的循环数
    mech_share: float           # f_target 处线性分量占比（未达 target 时=末深度）
    reached_target: bool
    rss: float
    rss_sei_only: float         # 单分量对照 RSS
    boundary: str               # "interior" / "sei_only" / "mech_only"
    n_points: int
    basis_corr: float           # corr(n^b_sei, n) 窗口共线性
    share_profile: dict = field(default_factory=dict)  # {f: share} 实测到达深度

    def to_dict(self) -> dict:
        return {
            "a_sei": self.a_sei, "a_mech": self.a_mech, "b_sei": self.b_sei,
            "f_target": self.f_target, "n_at_target": self.n_at_target,
            "mech_share": self.mech_share, "reached_target": self.reached_target,
            "rss": self.rss, "rss_sei_only": self.rss_sei_only,
            "boundary": self.boundary, "n_points": self.n_points,
            "basis_corr": self.basis_corr, "share_profile": self.share_profile,
        }


def two_component_fit(n: np.ndarray, f: np.ndarray, b_sei: float = B_SEI_DEFAULT,
                      f_target: float = FADE_EOL,
                      profile_targets: tuple = (0.05, 0.10, 0.20)) -> MixDecomp:
    """f = a_s·N^{b_s} + a_m·N 的非负最小二乘（b_s 固定 → 线性于幅度）。

    - 窗口：f∈[0.01, 0.25] 与 fit_fade_power 同窗；
    - 边界解：任一幅度解出负值 → **闭式重估**留下那支（不能沿用联合
      解幅度——联合解靠两分量相消，截断后幅度虚高，v1.9 坑）；
    - mech_share 在平滑曲线首次到 f_target 处评估；未到用末点并置
      reached_target=False。
    """
    n = np.asarray(n, float)
    f = np.asarray(f, float)
    fs = smooth_fade(f)
    m = (fs > 0.01) & (fs < 0.25) & (n > 0) & np.isfinite(fs)
    n_w, f_w = n[m], fs[m]

    def _empty(boundary: str, npts: int) -> MixDecomp:
        return MixDecomp(float("nan"), float("nan"), b_sei, f_target,
                         float("nan"), float("nan"), False,
                         float("nan"), float("nan"), boundary, npts,
                         float("nan"), {})

    if n_w.size < 5:
        return _empty("insufficient", int(n_w.size))
    x_s = n_w ** b_sei
    x_m = n_w
    basis_corr = float(np.corrcoef(x_s, x_m)[0, 1])
    a_mat = np.column_stack([x_s, x_m])
    coef, *_ = np.linalg.lstsq(a_mat, f_w, rcond=None)
    a_s, a_m = float(coef[0]), float(coef[1])
    boundary = "interior"

    def _rss(pred: np.ndarray) -> float:
        return float(np.sum((f_w - pred) ** 2))

    if a_s < 0 or a_m < 0:
        # 边界：只留非负那支并闭式重估幅度
        if a_m < 0:
            a_s = float(x_s @ f_w / (x_s @ x_s))
            a_m = 0.0
            boundary = "sei_only"
        else:
            a_m = float(x_m @ f_w / (x_m @ x_m))
            a_s = 0.0
            boundary = "mech_only"
        if a_s < 0 or a_m < 0:  # 数值兜底
            a_s, a_m = max(a_s, 0.0), max(a_m, 0.0)
            boundary = "sei_only" if a_s >= a_m else "mech_only"
    pred = a_s * x_s + a_m * x_m
    rss = _rss(pred)
    a_solo = float(x_s @ f_w / (x_s @ x_s))
    rss_sei = _rss(a_solo * x_s)

    def _share_at(level: float) -> tuple:
        """(share, n_at, reached)——线性份额在平滑曲线到 level 处。"""
        hit = np.nonzero(fs >= level)[0]
        reached = bool(hit.size)
        if reached:
            n_t = float(n[hit[0]])
        else:
            fin = np.isfinite(fs)
            n_t = float(n[fin][-1]) if fin.any() else float(n[-1])
        f_mod = a_s * n_t ** b_sei + a_m * n_t
        share = float(a_m * n_t / f_mod) if f_mod > 0 else float("nan")
        return share, n_t, reached

    share, n_t, reached = _share_at(f_target)
    profile = {}
    for lv in profile_targets:
        sh_lv, _, reached_lv = _share_at(lv)
        if reached_lv:
            profile[str(lv)] = sh_lv
    return MixDecomp(a_s, a_m, b_sei, f_target, n_t, share, reached,
                     rss, rss_sei, boundary, int(n_w.size),
                     basis_corr, profile)


def mech_share_bootstrap(curves: list, b_sei: float = B_SEI_DEFAULT,
                         f_target: float = FADE_EOL, n_boot: int = 2000,
                         seed: int = 20261102) -> dict:
    """组内电芯 bootstrap：curves=[(cell_id, n, f)] → 线性份额 CI。

    重采样电芯集合，逐芯拟合取份额中位——报 median / 95% CI /
    边界解计数 / 达 target 比例。
    """
    rng = np.random.default_rng(seed)
    shares, bounds, reached = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(curves), len(curves))
        sh = []
        for i in idx:
            _, n, f = curves[i]
            d = two_component_fit(n, f, b_sei, f_target)
            if np.isfinite(d.mech_share):
                sh.append(d.mech_share)
                bounds.append(d.boundary)
                reached.append(d.reached_target)
        if sh:
            shares.append(float(np.median(sh)))
    shares_arr = np.asarray(shares, float)
    return {
        "b_sei": b_sei,
        "n_cells": len(curves),
        "n_boot": n_boot,
        "mech_share_median": float(np.median(shares_arr)) if shares_arr.size else float("nan"),
        "mech_share_ci95": [
            float(np.percentile(shares_arr, 2.5)) if shares_arr.size else float("nan"),
            float(np.percentile(shares_arr, 97.5)) if shares_arr.size else float("nan"),
        ],
        "sei_only_frac": float(np.mean([b == "sei_only" for b in bounds])) if bounds else float("nan"),
        "mech_only_frac": float(np.mean([b == "mech_only" for b in bounds])) if bounds else float("nan"),
        "reached_target_frac": float(np.mean(reached)) if reached else float("nan"),
    }


def mech_share_sensitivity(curves: list, f_target: float = FADE_EOL,
                           anchors: tuple = B_SEI_ANCHORS) -> list:
    """b_sei 锚点扫描 → 份额敏感性带（每个锚点报中位份额）。"""
    rows = []
    for b in anchors:
        sh = []
        for _, n, f in curves:
            d = two_component_fit(n, f, b, f_target)
            if np.isfinite(d.mech_share):
                sh.append(d.mech_share)
        rows.append({
            "b_sei": b,
            "mech_share_median": float(np.median(sh)) if sh else float("nan"),
            "n_fits": len(sh),
        })
    return rows


@dataclass
class AnchoredExcess:
    """锚定 SEI 本底的超额分解（主口径点归因估计器）。

    与 two_component_fit 的区别：a_s 不从本曲线拟合（那会被共线基
    吸收），而是**外部锚定**同体系 1C 组的 SEI 幅度——SEI 每 EFC
    近似倍率无关（Vetter 2005），超额 f − a_s·N^{b_s} 才是机械候选。
    a_m 为超额对 N 的非负回归，物理份额 = a_m·N/(a_s·N^{b_s}+a_m·N)
    在实测到达深度评估——可辨识性远好于自由双分量拟合。

    保守方向：3C 每 EFC 壁钟更短 → SEI（含日历分量）实际比 1C 少，
    1C 锚定**高估** SEI 本底 → 机械份额是下界侧。
    """

    a_sei_anchored: float
    a_mech: float
    b_sei: float
    f_target: float
    n_at_target: float
    mech_share: float
    sei_at_target: float
    mech_at_target: float
    reached_target: bool
    excess_negative: bool     # 超额整体为负（本底盖过实测）→ 份额记 0

    def to_dict(self) -> dict:
        return {
            "a_sei_anchored": self.a_sei_anchored, "a_mech": self.a_mech,
            "b_sei": self.b_sei, "f_target": self.f_target,
            "n_at_target": self.n_at_target, "mech_share": self.mech_share,
            "sei_at_target": self.sei_at_target, "mech_at_target": self.mech_at_target,
            "reached_target": self.reached_target,
            "excess_negative": self.excess_negative,
        }


def fit_sei_amplitude(n: np.ndarray, f: np.ndarray,
                      b_sei: float = B_SEI_DEFAULT) -> float:
    """单分量 SEI 幅度 a_s（闭式最小二乘，f = a_s·N^{b_s}）。"""
    n = np.asarray(n, float)
    fs = smooth_fade(f)
    m = (fs > 0.01) & (fs < 0.25) & (n > 0) & np.isfinite(fs)
    x = n[m] ** b_sei
    y = fs[m]
    if x.size < 5:
        return float("nan")
    return float(x @ y / (x @ x))


def anchored_excess_decomp(n: np.ndarray, f: np.ndarray, a_sei: float,
                           b_sei: float = B_SEI_DEFAULT,
                           f_target: float = FADE_EOL) -> AnchoredExcess:
    """锚定 a_s 后的超额机械分解。"""
    n = np.asarray(n, float)
    fs = smooth_fade(f)
    m = (fs > 0.01) & (fs < 0.25) & (n > 0) & np.isfinite(fs)
    n_w, f_w = n[m], fs[m]
    if n_w.size < 5 or not np.isfinite(a_sei):
        return AnchoredExcess(a_sei, float("nan"), b_sei, f_target,
                              float("nan"), float("nan"), float("nan"),
                              float("nan"), False, False)
    resid = f_w - a_sei * n_w ** b_sei
    x_m = n_w
    a_m_raw = float(x_m @ resid / (x_m @ x_m))
    excess_neg = bool(a_m_raw < 0)
    a_m = max(a_m_raw, 0.0)
    hit = np.nonzero(fs >= f_target)[0]
    reached = bool(hit.size)
    if reached:
        n_t = float(n[hit[0]])
    else:
        fin = np.isfinite(fs)
        n_t = float(n[fin][-1]) if fin.any() else float(n[-1])
    sei_part = float(a_sei * n_t ** b_sei)
    mech_part = float(a_m * n_t)
    tot = sei_part + mech_part
    share = float(mech_part / tot) if tot > 0 else float("nan")
    return AnchoredExcess(a_sei, a_m, b_sei, f_target, n_t, share,
                          sei_part, mech_part, reached, excess_neg)


def anchored_share_bootstrap(curves: list, a_sei: float,
                             b_sei: float = B_SEI_DEFAULT,
                             f_target: float = FADE_EOL,
                             n_boot: int = 2000,
                             seed: int = 20261102) -> dict:
    """锚定分解的组内电芯 bootstrap（份额 median / 95% CI）。"""
    rng = np.random.default_rng(seed)
    shares, negs, reached = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(curves), len(curves))
        sh = []
        for i in idx:
            _, n, f = curves[i]
            d = anchored_excess_decomp(n, f, a_sei, b_sei, f_target)
            if np.isfinite(d.mech_share):
                sh.append(d.mech_share)
                negs.append(d.excess_negative)
                reached.append(d.reached_target)
        if sh:
            shares.append(float(np.median(sh)))
    arr = np.asarray(shares, float)
    return {
        "a_sei_anchored": a_sei, "b_sei": b_sei,
        "n_cells": len(curves), "n_boot": n_boot,
        "mech_share_median": float(np.median(arr)) if arr.size else float("nan"),
        "mech_share_ci95": [
            float(np.percentile(arr, 2.5)) if arr.size else float("nan"),
            float(np.percentile(arr, 97.5)) if arr.size else float("nan"),
        ],
        "excess_negative_frac": float(np.mean(negs)) if negs else float("nan"),
        "reached_target_frac": float(np.mean(reached)) if reached else float("nan"),
    }
