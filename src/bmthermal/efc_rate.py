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


# ---------------------------------------------------------------------------
# matched-depth 速率（v1.15）：修「割线跨深度不可比」混杂
# ---------------------------------------------------------------------------

def matched_depth_rates(traj: pd.DataFrame, fade_cap: float = 15.0,
                        min_pts: int = 3) -> pd.DataFrame:
    """逐芯 matched-depth fade 速率（%/100 EFC）。

    - rate_early：fade ≤ fade_cap 的点做 OLS fade~efc → %/100 EFC
      （点数不足 min_pts 时取前 min_pts 点，flag=1）；负 fade 点
      （q_bol 参考噪声伪影，fade<0=容量高于参考）不入 OLS；
    - efc_at_cap：线性插值到 fade=fade_cap 的 EFC（未达 → NaN）；
    - rate_to_cap：fade_cap/efc_at_cap×100（到该深度割线）。
    fade 轨迹非单调时按 fade 排序后插值（如实，不平滑）。
    """
    rows = []
    for (g, cell), sub in traj.groupby(["group", "cell"]):
        sub = sub.sort_values("efc")
        sub = sub[sub.fade_pct >= 0]          # 滤 q_bol 噪声负 fade
        if len(sub) < min_pts:
            continue
        efc = sub.efc.to_numpy(float)
        fade = sub.fade_pct.to_numpy(float)
        early = sub[sub.fade_pct <= fade_cap]
        flag = 0
        if len(early) < min_pts:
            early = sub.head(min_pts)
            flag = 1
        ea = early.efc.to_numpy(float)
        fa = early.fade_pct.to_numpy(float)
        if ea.max() - ea.min() < 1e-9:
            slope = float("nan")
        else:
            slope = float(np.polyfit(ea, fa, 1)[0]) * 100.0
        efc_at_cap = float("nan")
        if len(sub) and np.nanmax(fade) >= fade_cap:
            order = np.argsort(fade)
            f_sorted = fade[order]
            e_sorted = efc[order]
            uniq_f, idx = np.unique(f_sorted, return_index=True)
            efc_at_cap = float(np.interp(fade_cap, uniq_f, e_sorted[idx]))
        rate_to_cap = (fade_cap / efc_at_cap * 100.0
                       if np.isfinite(efc_at_cap) and efc_at_cap > 0
                       else float("nan"))
        rows.append({
            "group": int(g), "cell": int(cell),
            "n_used": int(len(early)), "flag": flag,
            "rate_early": slope,
            "efc_at_cap": efc_at_cap,
            "rate_to_cap": rate_to_cap,
            "fade_max": float(np.nanmax(fade)),
            "efc_total": float(efc[-1]),
        })
    return pd.DataFrame(rows)


def fade_at_efc(traj: pd.DataFrame, target: float,
                efc_col: str = "efc", fade_col: str = "fade_pct") -> pd.DataFrame:
    """逐芯 fade@target EFC（线性插值；轨迹未达 → NaN）。

    同里程锚：比 fade 水平不比深度混杂割线（v1.15）。
    """
    rows = []
    for (g, cell), sub in traj.groupby(["group", "cell"]):
        s = sub.sort_values(efc_col)
        e = s[efc_col].to_numpy(float)
        f = s[fade_col].to_numpy(float)
        if len(s) < 2 or e[-1] < target:
            val = float("nan")
        else:
            val = float(np.interp(target, e, f))
        rows.append({"group": int(g), "cell": int(cell),
                     "fade_at_efc": val, "efc_max": float(e[-1])})
    return pd.DataFrame(rows)


def early_rate_band(traj: pd.DataFrame, fade_cap: float = 15.0,
                    max_skip: int = 2, min_pts: int = 3) -> dict:
    """逐芯 early 速率的 BoL 选点敏感带（skip first k=0..max_skip）。

    返回 {cell_key: (rate_k0, rate_k1, ...)} 与组级带；BoL 区容量
    点噪声（v1.13 坑十八变体）使 OLS 对起点敏感——报告带宽比单值
    诚实。负 fade 点（q_bol 噪声伪影）在 matched_depth_rates 已滤。
    """
    out: dict = {"per_cell": {}, "group_med_band": {}}
    for (g, cell), sub in traj.groupby(["group", "cell"]):
        s = sub.sort_values("efc")
        rates = []
        for k in range(max_skip + 1):
            sk = s.iloc[k:]
            early = sk[sk.fade_pct <= fade_cap]
            if len(early) < min_pts:
                early = sk.head(min_pts)
            ea = early.efc.to_numpy(float)
            fa = early.fade_pct.to_numpy(float)
            if ea.max() - ea.min() < 1e-9:
                rates.append(float("nan"))
            else:
                rates.append(float(np.polyfit(ea, fa, 1)[0]) * 100.0)
        out["per_cell"][f"g{int(g)}_c{int(cell)}"] = rates
    # 组级：各 skip 档的逐芯中位（带=档间极差）
    df = pd.DataFrame([
        {"group": int(k.split("_")[0][1:]), "rates": v}
        for k, v in out["per_cell"].items()
    ])
    for g, sub in df.groupby("group"):
        med_by_k = [float(np.nanmedian([r[i] for r in sub.rates]))
                    for i in range(max_skip + 1)]
        out["group_med_band"][f"g{int(g)}"] = {
            "med_by_skip": med_by_k,
            "band": [float(np.nanmin(med_by_k)), float(np.nanmax(med_by_k))],
        }
    return out
