"""EFC/日历轴轨迹与 fade 速率（v1.14，52 复用）。

口径：EFC = 累积放电 Ah / q_bol（q_bol=BoL 标签块最大 CC 放电）；
日历天 = 块时长求和 / 1440（TestTime 分钟制）；容量点 = RPT 标签
块的最大 CC 放电段（合并块块粒度近似，与 50 同口径）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bmthermal.guide_map import is_rpt

Q_MIN = 0.3  # 容量点门槛 [Ah]


def build_trajectories(tp: pd.DataFrame,
                       q_min: float = Q_MIN
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """全块吞吐表 → (容量点轨迹, 逐芯速率表)。

    tp 列约定：group, cell, step, label, ah_dis, t_min, q_cc,
    ah_dis_at_cc, t_cc_end_min（block_throughput_stats 输出）。
    速率：fade/100 EFC 割线 + OLS、fade/100 天割线。
    """
    q_rows, rate_rows = [], []
    for (g, cell), sub in tp.groupby(["group", "cell"]):
        sub = sub.sort_values("step")
        if len(sub) < 2:
            continue
        bol = sub[sub.label.str.contains("BoL", na=False)]
        cands = bol.q_cc[bol.q_cc >= q_min] if len(bol) else pd.Series(dtype=float)
        if len(cands):
            q_bol = float(cands.max())
        else:
            anyq = sub.q_cc[sub.q_cc >= q_min]
            q_bol = float(anyq.max()) if len(anyq) else float("nan")
        if not np.isfinite(q_bol) or q_bol <= 0:
            continue
        cum_dis, cum_t = 0.0, 0.0
        pts = []
        for _, r in sub.iterrows():
            if r.q_cc >= q_min and is_rpt(str(r.label)):
                efc = (cum_dis + float(r.ah_dis_at_cc)) / q_bol
                days = (cum_t + float(r.t_cc_end_min)) / 1440.0
                pts.append((efc, days, float(r.q_cc), int(r.step),
                            str(r.label)))
            cum_dis += float(r.ah_dis)
            cum_t += float(r.t_min)
        if len(pts) < 3:
            continue
        efc_a = np.array([p[0] for p in pts], dtype=float)
        days_a = np.array([p[1] for p in pts], dtype=float)
        fade_a = np.array([(1 - p[2] / q_bol) * 100 for p in pts], dtype=float)
        for efc, days, q, step, label in pts:
            q_rows.append({
                "group": g, "cell": int(cell), "step": step,
                "label": label, "q_cc": q,
                "efc": float(efc), "days": float(days),
                "fade_pct": float((1 - q / q_bol) * 100),
            })
        efc_tot = float(efc_a[-1])
        days_tot = float(days_a[-1])
        fade_eol = float(fade_a[-1])
        sec_efc = fade_eol / efc_tot * 100 if efc_tot > 0 else float("nan")
        sec_day = fade_eol / days_tot * 100 if days_tot > 0 else float("nan")
        if efc_a.max() - efc_a.min() > 1e-9 and len(pts) >= 3:
            slope = float(np.polyfit(efc_a, fade_a, 1)[0]) * 100
        else:
            slope = float("nan")
        rate_rows.append({
            "group": g, "cell": int(cell),
            "q_bol": q_bol, "n_pts": len(pts),
            "efc_total": efc_tot, "days_total": days_tot,
            "fade_eol_pct": fade_eol,
            "rate_efc_secant": sec_efc,
            "rate_efc_ols": slope,
            "rate_day_secant": sec_day,
        })
    return pd.DataFrame(q_rows), pd.DataFrame(rate_rows)


def group_rates(rates: pd.DataFrame,
                group_meta: dict | None = None) -> dict:
    """逐芯速率表 → 组级中位锚点（含 kind）。"""
    group_meta = group_meta or {}
    out = {}
    for g in sorted(rates.group.unique()):
        sub = rates[rates.group == g]
        def med(k: str, _s=sub):
            return float(np.nanmedian(_s[k])) if len(_s) else None
        out[f"g{g}"] = {
            "n": int(len(sub)),
            "kind": group_meta.get(g, {}).get("kind", "?"),
            "efc_total_med": med("efc_total"),
            "days_total_med": med("days_total"),
            "fade_eol_med": med("fade_eol_pct"),
            "rate_efc_secant_med": med("rate_efc_secant"),
            "rate_efc_ols_med": med("rate_efc_ols"),
            "rate_day_secant_med": med("rate_day_secant"),
        }
    return out
