"""绝对寿命标定：实测 fade 锚点 × 相对损伤模型 → 绝对 EFC。

背景（报告开放项 #3）
---------------------
v1.0–v1.6 只报**相对**寿命倍数——损伤积分 D 的前因子 A 未标定，
D_EOL 在方案比值里消掉（feedback.py 的推导）。开放项 #3 指出
SNL 各体系 fade_per_efc 是现成绝对锚点，可按体系重标。

标定链
------
SNL 在参考工况（25 °C、0.5C 充/1C 放、0–100% 窗）实测::

    fade_ref = Δf / ΔEFC          [每 EFC 衰减，1/EFC]

模型在同参考工况算出每 EFC 损伤 d_ref（等效 25 °C 秒，aging.py）。
两者之比即该体系的衰减-损伤比例::

    A_chem = fade_ref / d_ref     [每损伤秒的衰减]

任一场景（d_s 损伤/EFC，倍率效应 m(c)）的绝对寿命::

    N_EOL = FADE_EOL / (A_chem · d_s · m(c))        [线性形状]

**跨电芯格式迁移是量级性的**：A_chem 在 SNL 18650 上标定，套到
项目基准电芯（模型几何）只保证量级，不保证 ±10%。报告必须带
此边界——绝对 EFC 是"SNL 同类化学体系的量级参考"，不是台架实测。

形状检验（线性 vs 幂律）
------------------------
fade 窗口斜率是 f∈[1%,25%] 的平均斜率。若 f=a·N^b 且 b<1（SEI 型
亚线性），窗口平均斜率高于 f=0.2 处的局部斜率 → 线性外推的 N_EOL
**偏乐观**（比真实寿命长）。幂律口径把形状显式建模，通常更近真值。
``audit_extrapolation`` 用**到达 EOL 的**电芯（CS2 深衰减组）对拍
三种口径：平滑实测交叉（参考）vs 线性窗口 vs 幂律窗口，并与宽窗
幂律（1–40%）交叉验证。曲线必须滚动中位平滑——单点容量毛刺会
伪造 EOL 交叉（v1.7 踩坑记录）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .feedback import FADE_EOL

__all__ = [
    "FADE_EOL",
    "ChemAnchor",
    "n_eol_linear",
    "n_eol_power",
    "n_eol_from_curve",
    "audit_extrapolation",
    "scale_scenario_fade",
]


def n_eol_linear(fade_per_efc: float, fade_eol: float = FADE_EOL) -> float:
    """线性口径：窗口斜率外推到 EOL。亚线性衰减时偏乐观（上界侧）。"""
    if not np.isfinite(fade_per_efc) or fade_per_efc <= 0:
        return float("nan")
    return float(fade_eol / fade_per_efc)


def n_eol_power(a: float, b: float, fade_eol: float = FADE_EOL) -> float:
    """幂律口径：f = a·N^b → N_EOL = (f_eol/a)^(1/b)。"""
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return float("nan")
    return float((fade_eol / a) ** (1.0 / b))


def n_eol_from_curve(n: np.ndarray, f: np.ndarray, fade_eol: float = FADE_EOL) -> float:
    """直接口径：衰减曲线首次穿过 f_eol 的循环数（实测交叉）。"""
    from .snl_aging import first_crossing  # 局部导入避免环

    return first_crossing(n, f, fade_eol)


@dataclass
class ChemAnchor:
    """某化学体系在参考工况的绝对寿命锚。"""

    chem: str
    temp_c: float
    c_dis: float
    #: 参考工况实测每 EFC 衰减（SNL 组均值）
    fade_ref_per_efc: float
    n_cells: int
    fade_std_per_efc: float
    #: 参考工况模型损伤（等效 25 °C 秒/EFC）
    d_ref_s: float
    #: 幂律形状（可选）
    b_fade: float = float("nan")
    a_fade: float = float("nan")
    n_eol_cross_median: float = float("nan")
    note: str = ""

    @property
    def fade_per_damage_s(self) -> float:
        """A_chem = fade_ref / d_ref（每损伤秒衰减）。"""
        if not np.isfinite(self.d_ref_s) or self.d_ref_s <= 0:
            return float("nan")
        return float(self.fade_ref_per_efc / self.d_ref_s)

    @property
    def n_eol_ref(self) -> float:
        """参考工况无反馈 N_EOL（线性口径）。"""
        return n_eol_linear(self.fade_ref_per_efc)

    def scenario_fade(self, d_s: float, m_c: float = 1.0) -> float:
        """场景每 EFC 衰减 = A_chem · d_s · m(c)。"""
        return float(self.fade_per_damage_s * d_s * m_c)

    def n_eol(self, d_s: float, m_c: float = 1.0,
              shape: str = "linear") -> float:
        """场景绝对寿命。

        shape='linear'：窗口斜率口径（亚线性时偏保守）；
        shape='power'：幂律口径，须已标定 b_fade/a_fade；
        'power' 下倍率效应按"时钟缩放"处理：f=a·(k·N)^b，
        N_EOL = N_ref / k，k=(d_s/d_ref)·m(c)——与线性同为 1/k，
        差别只在 N_ref 本身。
        """
        k = float(d_s / self.d_ref_s) * float(m_c)
        if not np.isfinite(k) or k <= 0:
            return float("nan")
        if shape == "power":
            n_ref = n_eol_power(self.a_fade, self.b_fade)
            return float(n_ref / k)
        return n_eol_linear(self.scenario_fade(d_s, m_c))

    def to_dict(self) -> dict:
        return {
            "chem": self.chem, "temp_c": self.temp_c, "c_dis": self.c_dis,
            "fade_ref_per_efc": self.fade_ref_per_efc,
            "fade_std_per_efc": self.fade_std_per_efc,
            "n_cells": self.n_cells,
            "d_ref_s": self.d_ref_s,
            "fade_per_damage_s": self.fade_per_damage_s,
            "n_eol_ref_linear": self.n_eol_ref,
            "n_eol_ref_power": n_eol_power(self.a_fade, self.b_fade),
            "n_eol_cross_median": self.n_eol_cross_median,
            "b_fade": self.b_fade, "a_fade": self.a_fade,
            "note": self.note,
        }


def scale_scenario_fade(anchor: ChemAnchor, d_s: float, m_c: float = 1.0) -> float:
    """便捷封装：场景每 EFC 衰减。"""
    return anchor.scenario_fade(d_s, m_c)


@dataclass
class ExtrapolationAudit:
    """到达 EOL 电芯上三种 N_EOL 口径的外推审计。

    参考 = 滚动中位平滑后的实测交叉；同时给宽窗（1–40%）幂律
    交叉验证，确认参考本身不含毛刺偏差。
    """

    rows: list = field(default_factory=list)

    def add(self, cell_id: str, n_cross: float,
            n_lin: float, n_pow: float, b: float,
            n_oracle: float = float("nan")) -> None:
        def err(pred: float) -> float:
            if not (np.isfinite(pred) and np.isfinite(n_cross) and n_cross > 0):
                return float("nan")
            return float((pred - n_cross) / n_cross * 100.0)

        self.rows.append({
            "cell_id": cell_id, "n_cross": float(n_cross),
            "n_linear": float(n_lin), "n_power": float(n_pow),
            "n_oracle_wide_window": float(n_oracle),
            "b": float(b),
            "err_linear_pct": err(n_lin), "err_power_pct": err(n_pow),
            "err_oracle_pct": err(n_oracle),
        })

    def summary(self) -> dict:
        el = np.array([r["err_linear_pct"] for r in self.rows], float)
        ep = np.array([r["err_power_pct"] for r in self.rows], float)
        eo = np.array([r["err_oracle_pct"] for r in self.rows], float)
        el, ep, eo = (el[np.isfinite(el)], ep[np.isfinite(ep)], eo[np.isfinite(eo)])
        return {
            "n_cells": len(self.rows),
            "err_linear_median_pct": float(np.median(el)) if el.size else float("nan"),
            "err_power_median_pct": float(np.median(ep)) if ep.size else float("nan"),
            "err_oracle_median_pct": float(np.median(eo)) if eo.size else float("nan"),
            "err_linear_max_abs_pct": float(np.max(np.abs(el))) if el.size else float("nan"),
            "err_power_max_abs_pct": float(np.max(np.abs(ep))) if ep.size else float("nan"),
            "verdict": "",
        }

    def to_dict(self) -> dict:
        s = self.summary()
        s["rows"] = self.rows
        return s


def smooth_fade(f: np.ndarray, window: int = 11) -> np.ndarray:
    """滚动中位平滑（抗单点容量毛刺）。"""
    import pandas as pd

    return (pd.Series(np.asarray(f, float))
            .rolling(window, center=True, min_periods=3).median().to_numpy())


def audit_extrapolation(curves: list) -> ExtrapolationAudit:
    """对每条 (cell_id, n, f) 到达 EOL 的曲线做口径审计。

    curves: [(cell_id, n_array, f_array)]，只统计 f 穿过 0.2 且
    交叉单调性通过的电芯。拟合与交叉同用平滑曲线。
    """
    audit = ExtrapolationAudit()
    for cell_id, n, f in curves:
        n = np.asarray(n, float)
        f = np.asarray(f, float)
        fs = smooth_fade(f)
        n_cross = n_eol_from_curve(n, fs)
        if not np.isfinite(n_cross):
            continue
        m = (fs > 0.01) & (fs < 0.25) & (n > 0) & np.isfinite(fs)
        if m.sum() < 5:
            continue
        slope = float(np.polyfit(n[m], fs[m], 1)[0])
        pw = _fit_power(n[m], fs[m])
        # 宽窗幂律（1–40%）作参考交叉验证
        m2 = (fs > 0.01) & (fs < 0.40) & (n > 0) & np.isfinite(fs)
        if m2.sum() >= 5:
            pw2 = _fit_power(n[m2], fs[m2])
            n_oracle = n_eol_power(pw2["a"], pw2["b"])
        else:
            n_oracle = float("nan")
        audit.add(cell_id, n_cross,
                  n_eol_linear(slope),
                  n_eol_power(pw["a"], pw["b"]),
                  pw["b"], n_oracle)
    return audit


def _fit_power(n: np.ndarray, f: np.ndarray) -> dict:
    x = np.log(n)
    y = np.log(f)
    b, loga = np.polyfit(x, y, 1)
    return {"a": float(np.exp(loga)), "b": float(b)}
