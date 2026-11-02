"""倍率效应项 m(c)：每 EFC 损伤对放电倍率的偏离（WP23 共享振幅假设的实测修正）。

背景（WP28，scripts/28 开放项）
-------------------------------
WP23 的日历-循环分解假设"循环损伤振幅与倍率无关"（共享振幅）。
SNL 实测在 3C **双向失效**：LFP 3C/1C 每 EFC ×2.30、NMC ×0.64。
本模块把 m(c) 作为显式项入寿命账::

    d_per_efc(c, T) = m(c) · exp(-Ea/R·(1/T − 1/T_ref))

口径
----
- m(c) = fade_per_efc(c) / fade_per_efc(1C)，同体系/温度/SOC 窗（SNL）。
- 资格：n≥2 的组给点估计；n=1 只作指示。跨体系 envelope 是"体系未定"
  时的迁移带，不是统计 CI。
- m<1 **不是**"高倍率有益"的证据——v1.5 §36 已确认批次/协议异质性
  可以伪造方向；机理归因只给 plausibility 判词，不给点归因。
- 同倍率方案对比（C5：3C 液冷 vs 3C 自然）m 消掉，主口径不受影响；
  跨倍率设计权衡（分档选型）才用得上 m(c)。

机理归因（文献口径，非本项目实测）
---------------------------------
- 高倍率损伤↑ 候选机制：颗粒开裂/接触丧失（Vetter et al. 2005 老化机制
  综述；Anseán et al. 2016 LFP 应力演化）、快充析锂（Waldmann et al. 2014，
  当同组充电倍率也高时优先级上升）。
- 高倍率损伤↓：没有已知退化机制让高倍率更轻——按异质性 nuisance 处理，
  不入机制账（与 v1.5 §36 判词一致）。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .aging import EA_DEFAULT, arrhenius_factor
from .snl_aging import group_stats

#: m(c) 偏离阈值：|m−1| 超过此值才进入机理归因讨论
M_DEVIATION = 0.25


@dataclass(frozen=True)
class RatePoint:
    """某体系某放电倍率组的倍率效应点。"""

    chem: str
    temp_c: float
    c_dis: float
    n_cells: int
    c_chg_mean: float
    fade_per_efc: float
    fade_per_hour: float
    m_per_efc: float
    eligible: bool
    role: str  # "ref"（1C 基准）/ "solid"（n≥2）/ "indicative"（n=1）

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Attribution:
    """机理归因判词（plausibility，非点归因）。"""

    chem: str
    c_dis: float
    m_per_efc: float
    direction: str  # "elevated" / "depressed" / "neutral"
    plausible_mechanisms: list[str] = field(default_factory=list)
    verdict: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def rate_points(
    df: pd.DataFrame,
    chem: str,
    temp_c: float = 25.0,
    soc_lo: float = 0.0,
    soc_hi: float = 100.0,
    c_ref: float = 1.0,
) -> list[RatePoint]:
    """同体系各倍率组的 m(c)（相对 1C 组的每 EFC 损伤比）。"""
    base = group_stats(df, chem, temp_c, c_ref, soc_lo, soc_hi)
    mask = (
        (df["chem"] == chem) & (df["temp_c"] == temp_c)
        & (df["soc_lo"] == soc_lo) & (df["soc_hi"] == soc_hi)
    )
    rates = sorted(df.loc[mask, "c_dis"].dropna().unique())
    out: list[RatePoint] = []
    for c in rates:
        g = group_stats(df, chem, temp_c, float(c), soc_lo, soc_hi)
        m = g["fade_per_efc"] / base["fade_per_efc"]
        dt = g["dt_s"]
        fade_h = g["fade_per_efc"] / dt * 3600.0 if dt > 0 else float("nan")
        sub = df[mask & (df["c_dis"] == c)]
        c_chg = float(sub["c_chg"].mean()) if "c_chg" in sub else float("nan")
        n = int(g["n_cells"])
        role = "ref" if abs(c - c_ref) < 1e-9 else ("solid" if n >= 2 else "indicative")
        out.append(RatePoint(
            chem=chem, temp_c=float(temp_c), c_dis=float(c), n_cells=n,
            c_chg_mean=c_chg, fade_per_efc=float(g["fade_per_efc"]),
            fade_per_hour=float(fade_h), m_per_efc=float(m),
            eligible=(n >= 2), role=role,
        ))
    return out


def mechanism_attribution(chem: str, c_dis: float, m_per_efc: float,
                          c_chg_mean: float = float("nan")) -> Attribution:
    """对 m(c) 偏离做机理 plausibility 归因（文献口径）。"""
    if m_per_efc > 1.0 + M_DEVIATION:
        if c_dis <= 1.0 + 1e-9:
            direction = "elevated_low_rate"
            mech = []
            verdict = (
                f"{chem} {c_dis:g}C m={m_per_efc:.2f} 升高出现在低倍率侧，"
                "与机械损伤/析锂的倍率方向不符——按批次/协议异质性 nuisance 候选处理，"
                "不入机制账。"
            )
        else:
            direction = "elevated"
            mech = [
                "高倍率机械损伤：颗粒开裂/接触丧失（Vetter et al. 2005；Anseán et al. 2016 LFP）",
            ]
            if np.isfinite(c_chg_mean) and c_chg_mean >= 1.0:
                mech.append("快充析锂候选（Waldmann et al. 2014）——同组充电倍率≥1C，优先级上升")
            else:
                mech.append("快充析锂可能性低（同组充电倍率<1C）")
            verdict = (
                f"{chem} {c_dis:g}C m={m_per_efc:.2f} 升高有文献机制支持（机械损伤主导候选），"
                "点归因需 post-mortem，此处不锁。"
            )
    elif m_per_efc < 1.0 - M_DEVIATION:
        direction = "depressed"
        mech = []
        verdict = (
            f"{chem} {c_dis:g}C m={m_per_efc:.2f} 降低**无已知退化机制支持**"
            "（没有机制让高倍率每 EFC 更轻）——按批次/协议异质性 nuisance 处理"
            "（v1.5 §36 判词），不入机制账。"
        )
    else:
        direction = "neutral"
        mech = []
        verdict = f"{chem} {c_dis:g}C m={m_per_efc:.2f} 与 1C 无显著偏离（±{M_DEVIATION:.0%} 内）。"
    return Attribution(chem=chem, c_dis=float(c_dis), m_per_efc=float(m_per_efc),
                       direction=direction, plausible_mechanisms=mech, verdict=verdict)


def cross_chem_envelope(points: list[RatePoint]) -> dict:
    """各倍率跨体系 m 包络（体系未定时的迁移带）。"""
    by_c: dict[float, list[float]] = {}
    for p in points:
        if p.role == "ref":
            continue
        by_c.setdefault(p.c_dis, []).append(p.m_per_efc)
    return {
        float(c): {
            "lo": float(min(vs)), "hi": float(max(vs)),
            "n_chems": len(vs),
            "values": {p.chem: p.m_per_efc for p in points if p.c_dis == c},
        }
        for c, vs in sorted(by_c.items())
    }


def pair_m_ratio(fade_fast, fade_ref, n_boot: int = 5000,
                 seed: int = 20261102) -> dict:
    """m = mean(fade_fast)/mean(fade_ref) 的组内电芯 bootstrap。

    fade_fast/fade_ref 为逐芯 fade 率数组（同温度不同倍率组）。
    n<2 任一侧 → 点值仍给但 eligible=False（v1.5 资格判据同款）。
    """
    rng = np.random.default_rng(seed)
    fa = np.asarray(fade_fast, float)
    fr = np.asarray(fade_ref, float)
    if fa.size == 0 or fr.size == 0:
        return {"m": float("nan"), "ci95": [float("nan")] * 2,
                "n_fast": int(fa.size), "n_ref": int(fr.size), "eligible": False}
    m0 = float(np.mean(fa) / np.mean(fr))
    if fa.size < 2 or fr.size < 2:
        return {"m": m0, "ci95": [float("nan")] * 2,
                "n_fast": int(fa.size), "n_ref": int(fr.size), "eligible": False}
    stats = []
    for _ in range(n_boot):
        a = rng.choice(fa, size=fa.size, replace=True)
        r = rng.choice(fr, size=fr.size, replace=True)
        if np.mean(r) > 0:
            stats.append(float(np.mean(a) / np.mean(r)))
    arr = np.asarray(stats, float)
    return {
        "m": m0,
        "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
        "n_fast": int(fa.size), "n_ref": int(fr.size),
        "eligible": True,
    }


def matched_rep_m(rows_fast: list, rows_ref: list) -> dict:
    """rep 配对 m：同 rep 标签逐对比值取中位（批次异质性控制）。

    rows_*: [(rep, fade_rate)]。配对数 <2 → eligible=False。
    """
    ref = dict(rows_ref)
    ratios = []
    for rep, fade in rows_fast:
        if rep in ref and ref[rep] > 0:
            ratios.append(float(fade / ref[rep]))
    if len(ratios) < 2:
        return {"m_matched": float(np.median(ratios)) if ratios else float("nan"),
                "n_pairs": len(ratios), "eligible": False}
    return {
        "m_matched": float(np.median(ratios)),
        "n_pairs": len(ratios),
        "eligible": True,
    }


def apparent_ea_from_fade(fade_hi_t: float, fade_lo_t: float,
                          t_lo_c: float, t_hi_c: float) -> dict:
    """两温度 fade 比 → 表观 Ea（kJ/mol）。

    符号约定：fade 须随 T 升（fade_hi>fade_lo）→ Ea>0 合格；
    反向（负 Ea）→ eligible=False（NMC 温度轴失效口径，v1.7 §33）。
    """
    r_gas = 8.314462618
    t1, t2 = t_lo_c + 273.15, t_hi_c + 273.15
    if fade_lo_t <= 0 or fade_hi_t <= 0:
        return {"ea_kj_mol": float("nan"), "ratio": float("nan"), "eligible": False}
    ratio = float(fade_hi_t / fade_lo_t)
    ea = r_gas * np.log(ratio) / (1.0 / t1 - 1.0 / t2) / 1000.0
    return {
        "ea_kj_mol": float(ea),
        "ratio": ratio,
        "eligible": bool(ea > 0),
    }


def damage_per_efc(m_per_efc: float, t_core_c, ea_j_mol: float = EA_DEFAULT,
                   t_ref_c: float = 25.0) -> float:
    """每 EFC 相对损伤 = m(c) × <k(T_core)>（放电窗平均）。"""
    f = arrhenius_factor(np.asarray(t_core_c, dtype=float), ea_j_mol, t_ref_c)
    return float(m_per_efc * np.mean(f))


def tier_tradeoff_row(
    c_rate: float,
    m_per_efc: float,
    t_core_c,
    dt_s: float,
    ea_j_mol: float = EA_DEFAULT,
) -> dict:
    """跨倍率设计权衡的一行：损伤/EFC、相对寿命（EFC 与壁钟）、吞吐。"""
    k_t = float(np.mean(arrhenius_factor(np.asarray(t_core_c, dtype=float), ea_j_mol)))
    d = m_per_efc * k_t
    return {
        "c_rate": float(c_rate),
        "m_per_efc": float(m_per_efc),
        "k_t_mean": k_t,
        "damage_per_efc_rel": d,
        "life_efc_rel": 1.0 / d,          # 到 EOL 的相对 EFC 数（1C 基准=1 需外部归一）
        "dt_s": float(dt_s),
        "life_hours_rel": (1.0 / d) * float(dt_s) / 3600.0,
        "throughput_per_hour_rel": 3600.0 / (d * float(dt_s)),  # EFC/小时 到 EOL 的相对吞吐
    }
