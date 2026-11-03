"""硬件实验数据侧管线（v1.13，WP54）——§67 三项实验的数据模式与裁决接线。

定位：§67 已量化 E1（5C 台架）/E2（post-mortem）/E3（3C/5C×T 矩阵）
的样本量、精度预算与**预注册判据**（exp_design.decision_5c /
decision_postmortem）。本模块把「硬件数据到手 → 裁决」的路径代码化：

1. **数据模式**（schema）：三类实验的 CSV 列约定 + 校验；
2. **分析函数**：fade 率拟合 → m 比 + CI → 预注册判据调用；
3. **合成端到端验证**：按 §67 功效规格生成合成数据跑通全链，
   证明判据按预注册方向触发（不替代硬件，只证管线就绪）；
4. **精度门**：σ_Q ≤ 0.5955 mAh（0.057% Q_nom，45 的预算）——
   测量系统达标与否的预注册检查。

⚠️ 合成验证只证明「管线能判」，不证明「假设为真」；
5C sd 仍是 3C×1.2 代理（45 的假设定量化），首组真实数据到手
须回填重算功效。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bmthermal.exp_design import (
    M5_HI, M5_LO, M5_STAR, decision_5c, decision_postmortem,
    m_ratio_se, z_quantile,
)

# ---------------------------------------------------------------------------
# 数据模式（CSV 列约定）
# ---------------------------------------------------------------------------

# E1/E3 老化轨迹：每行 = 一个 RPT 观测点
AGING_SCHEMA = {
    "required": ("cell_id", "chem", "c_dis", "temp_c", "efc", "q_rpt_ah"),
    "dtypes": {"cell_id": str, "chem": str, "c_dis": float,
               "temp_c": float, "efc": float, "q_rpt_ah": float},
}

# E2 post-mortem：每行 = 一只牺牲芯的拆解指标
POSTMORTEM_SCHEMA = {
    "required": ("cell_id", "chem", "c_dis", "temp_c",
                 "fade_at_cut", "lam_pe", "lam_ne", "crack_pe", "crack_ne"),
    "dtypes": {"cell_id": str, "chem": str, "c_dis": float,
               "temp_c": float, "fade_at_cut": float,
               "lam_pe": float, "lam_ne": float,
               "crack_pe": float, "crack_ne": float},
}

# 精度预算（45_exp_design.json：σ_Q ≤ 0.5955 mAh = 0.057% Q_nom）
SIGMA_Q_REQ_AH = 5.9550728327599e-4
Q_NOM_AH = 1.039


def validate_schema(df: pd.DataFrame, schema: dict, name: str) -> pd.DataFrame:
    """列模式校验：缺列/类型不可解析即报错（如实失败，不静默补）。"""
    missing = [c for c in schema["required"] if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    out = df.copy()
    for col, typ in schema["dtypes"].items():
        try:
            out[col] = out[col].astype(typ)
        except (ValueError, TypeError) as e:
            raise ValueError(f"{name}: column {col} not {typ}: {e}") from e
    if out[list(schema["required"])].isna().any().any():
        raise ValueError(f"{name}: NaN in required columns")
    return out


def sigma_q_check(sigma_q_meas_ah: float,
                  sigma_q_req_ah: float = SIGMA_Q_REQ_AH) -> dict:
    """测量精度门：实测 σ_Q 对预算判定。"""
    if sigma_q_meas_ah <= 0:
        raise ValueError("sigma_q must be positive")
    return {
        "sigma_q_meas_ah": float(sigma_q_meas_ah),
        "sigma_q_req_ah": float(sigma_q_req_ah),
        "sigma_q_meas_rel": float(sigma_q_meas_ah / Q_NOM_AH),
        "adequate": bool(sigma_q_meas_ah <= sigma_q_req_ah + 1e-12),
    }


# ---------------------------------------------------------------------------
# fade 率拟合与 m 比裁决（E1/E3 共用）
# ---------------------------------------------------------------------------

@dataclass
class CellFade:
    cell_id: str
    chem: str
    c_dis: float
    temp_c: float
    rate_per_efc: float          # d(fade)/d(EFC)，fade=1−q/q0
    n_obs: int
    q0_ah: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def fit_cell_fade(df: pd.DataFrame, cell_id: str) -> CellFade:
    """单芯 fade 率：fade=1−q/q_ref 对 EFC 的最小二乘斜率。

    q_ref = 首个 RPT 点容量（BoL 参考；与 48 的「取 BoL 最大」不同——
    实验数据首点即 BoL RPT，无多 BoL 块问题）。
    """
    sub = df[df.cell_id == cell_id].sort_values("efc")
    if len(sub) < 3:
        raise ValueError(f"{cell_id}: need ≥3 RPT points, got {len(sub)}")
    q = sub.q_rpt_ah.to_numpy(dtype=float)
    efc = sub.efc.to_numpy(dtype=float)
    if not (q > 0).all():
        raise ValueError(f"{cell_id}: non-positive q_rpt")
    fade = 1.0 - q / q[0]
    # 最小二乘斜率（含截距；fade(0)=0 约束会放大噪声，不强加）
    slope = float(np.polyfit(efc, fade, 1)[0])
    first = sub.iloc[0]
    return CellFade(cell_id=cell_id, chem=str(first.chem),
                    c_dis=float(first.c_dis), temp_c=float(first.temp_c),
                    rate_per_efc=slope, n_obs=int(len(sub)),
                    q0_ah=float(q[0]))


def fit_arm(df: pd.DataFrame) -> list[CellFade]:
    """整表逐芯拟合（一臂 = 同 chem/c_dis/temp_c 的芯集合）。"""
    return [fit_cell_fade(df, cid) for cid in sorted(df.cell_id.unique())]


@dataclass
class MVerdict:
    chem: str
    temp_c: float
    m_est: float
    se: float
    ci_low: float
    ci_high: float
    n_ref: int
    n_c: int
    decision: str
    ref_rate: float
    c_rate: float
    note: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def compare_arms(ref: list[CellFade], target: list[CellFade],
                 m_lo: float = M5_LO, m_star: float = M5_STAR,
                 m_hi: float = M5_HI, note: str = "") -> MVerdict:
    """m = rate_target/rate_ref + 95% CI（delta 法）→ 预注册判据。

    先注册后测量：判据阈值 (m_lo, m_star, m_hi) 来自 v1.9 带界，
    本函数不因数据移动阈值。
    """
    if not ref or not target:
        raise ValueError("both arms need ≥1 cell")
    r_ref = np.array([c.rate_per_efc for c in ref], dtype=float)
    r_c = np.array([c.rate_per_efc for c in target], dtype=float)
    if (r_ref <= 0).any() or (r_c <= 0).any():
        raise ValueError("non-positive fade rate (check q trend)")
    m_est = float(r_c.mean() / r_ref.mean())
    se = m_ratio_se(float(r_ref.mean()), float(r_c.mean()),
                    float(r_ref.std(ddof=1)) if len(r_ref) > 1 else
                    0.2 * float(r_ref.mean()),
                    float(r_c.std(ddof=1)) if len(r_c) > 1 else
                    0.2 * float(r_c.mean()),
                    len(r_ref), len(r_c))
    z = z_quantile(0.975)
    ci_low, ci_high = m_est - z * se, m_est + z * se
    dec = decision_5c(m_est, ci_low, ci_high,
                      m_lo=m_lo, m_star=m_star, m_hi=m_hi)
    t0 = target[0]
    return MVerdict(chem=t0.chem, temp_c=t0.temp_c, m_est=m_est, se=se,
                    ci_low=float(ci_low), ci_high=float(ci_high),
                    n_ref=len(ref), n_c=len(target), decision=dec,
                    ref_rate=float(r_ref.mean()), c_rate=float(r_c.mean()),
                    note=note)


# ---------------------------------------------------------------------------
# E2 post-mortem 裁决接线
# ---------------------------------------------------------------------------

@dataclass
class PMVerdict:
    chem: str
    lam_35: float
    lam_25: float
    crack_35: float
    crack_25: float
    decision: str
    pair_note: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def analyze_postmortem(df: pd.DataFrame) -> PMVerdict:
    """35 vs 25 °C 同 fade 深度配对 → decision_postmortem。

    lam 指标 = (lam_pe+lam_ne)/2；crack 指标 = (crack_pe+crack_ne)/2；
    同温度多芯取均值（牺牲芯 2 只/臂）。
    """
    df = validate_schema(df, POSTMORTEM_SCHEMA, "postmortem")
    def _arm(temp: float) -> tuple[float, float]:
        sub = df[df.temp_c == temp]
        if sub.empty:
            raise ValueError(f"no cells at {temp}C")
        lam = float((sub.lam_pe + sub.lam_ne).mean() / 2)
        crack = float((sub.crack_pe + sub.crack_ne).mean() / 2)
        return lam, crack

    lam35, crack35 = _arm(35.0)
    lam25, crack25 = _arm(25.0)
    dec = decision_postmortem(lam35, lam25, crack35, crack25)
    chem = str(df.chem.iloc[0])
    return PMVerdict(chem=chem, lam_35=lam35, lam_25=lam25,
                     crack_35=crack35, crack_25=crack25, decision=dec)


# ---------------------------------------------------------------------------
# 合成数据生成（端到端验证用）
# ---------------------------------------------------------------------------

def synth_aging(arms: list[dict], n_cells: int, sd_rate: float,
                seed: int = 0, n_obs: int = 6,
                efc_max: float = 1200.0) -> pd.DataFrame:
    """合成老化轨迹：fade = rate·EFC + 噪声（rate 按臂给定）。

    arms: [{"chem","c_dis","temp_c","rate_per_efc"}, ...]
    返回 AGING_SCHEMA 列的 DataFrame。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for a in arms:
        for k in range(n_cells):
            cid = f"{a['chem']}_{a['c_dis']}C_{a['temp_c']}C_{k}"
            rate = a["rate_per_efc"] * (1 + rng.normal(0, 0.15))
            rate = max(rate, 1e-7)
            q0 = 1.039 * (1 + rng.normal(0, 0.002))
            efcs = np.linspace(0, efc_max, n_obs)
            for e in efcs:
                fade = rate * e + rng.normal(0, sd_rate * 60)
                rows.append({"cell_id": cid, "chem": a["chem"],
                             "c_dis": a["c_dis"], "temp_c": a["temp_c"],
                             "efc": float(e),
                             "q_rpt_ah": float(q0 * (1 - max(fade, -0.02)))})
    return pd.DataFrame(rows)


def synth_postmortem(chem: str, c_dis: float,
                     lam_true_35: float, lam_true_25: float,
                     crack_true_35: float, crack_true_25: float,
                     sd: float = 0.02, n_per_arm: int = 2,
                     seed: int = 0) -> pd.DataFrame:
    """合成 post-mortem 表（裂纹/活性材料损失指标围绕真值）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for temp, lam_t, crack_t in [(35.0, lam_true_35, crack_true_35),
                                 (25.0, lam_true_25, crack_true_25)]:
        for k in range(n_per_arm):
            rows.append({
                "cell_id": f"{chem}_{temp}C_{k}", "chem": chem,
                "c_dis": c_dis, "temp_c": temp,
                "fade_at_cut": 0.10 + rng.normal(0, 0.005),
                "lam_pe": float(max(lam_t + rng.normal(0, sd), 0)),
                "lam_ne": float(max(lam_t * 1.1 + rng.normal(0, sd), 0)),
                "crack_pe": float(max(crack_t + rng.normal(0, sd), 0)),
                "crack_ne": float(max(crack_t * 0.9 + rng.normal(0, sd), 0)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 端到端：合成数据 → 裁决（管线就绪证明）
# ---------------------------------------------------------------------------

def e1_end_to_end(seed: int = 0, n_cells: int = 5,
                  m_true: float = 2.3) -> dict:
    """E1（5C 台架）合成端到端：1C 参照臂 vs 5C 目标臂 → 裁决。

    m_true=2.3（带下界）→ 期望 mostly throughput_positive*；
    用 45 的 sd 代理（1.03e-5 rate 量纲）生成。
    """
    sd_rate = 1.0314488709113267e-5
    base = 2.5e-5
    df = synth_aging(
        [{"chem": "LFP", "c_dis": 1.0, "temp_c": 25.0, "rate_per_efc": base},
         {"chem": "LFP", "c_dis": 5.0, "temp_c": 25.0,
          "rate_per_efc": base * m_true}],
        n_cells=n_cells, sd_rate=sd_rate, seed=seed)
    ref = fit_arm(validate_schema(df[df.c_dis == 1.0], AGING_SCHEMA, "e1ref"))
    tgt = fit_arm(validate_schema(df[df.c_dis == 5.0], AGING_SCHEMA, "e1tgt"))
    v = compare_arms(ref, tgt)
    return {"m_true": m_true, "verdict": v.to_dict(),
            "correct": v.decision.startswith("throughput_positive")}


def e2_end_to_end(seed: int = 0, direction: str = "b") -> dict:
    """E2（post-mortem）合成端到端。

    direction="b"：35 °C 指标更高（读法 b 期望 confirmed）；
    direction="a"：25 °C 更高（读法 a 得权）。
    """
    if direction == "b":
        lam35, lam25, c35, c25 = 0.35, 0.20, 0.40, 0.22
        expect = "reading_b_confirmed"
    else:
        lam35, lam25, c35, c25 = 0.20, 0.35, 0.22, 0.40
        expect = "reading_a_gains_weight"
    df = synth_postmortem("LFP", 1.0, lam35, lam25, c35, c25, seed=seed)
    v = analyze_postmortem(df)
    return {"direction": direction, "verdict": v.to_dict(),
            "expect": expect, "correct": v.decision == expect}


def e3_end_to_end(seed: int = 0, n_cells: int = 4) -> dict:
    """E3（3C/5C×T）合成端到端：3C 臂 vs 1C 对照（25 °C）→ m(c,T)。

    注入 m_true=2.0（方向性验证：CI 是否随 n 收缩到可判区）。
    """
    sd_rate = 8.6e-6
    base = 2.5e-5
    df = synth_aging(
        [{"chem": "LFP", "c_dis": 1.0, "temp_c": 25.0, "rate_per_efc": base},
         {"chem": "LFP", "c_dis": 3.0, "temp_c": 25.0,
          "rate_per_efc": base * 2.0}],
        n_cells=n_cells, sd_rate=sd_rate, seed=seed, efc_max=800.0)
    ref = fit_arm(validate_schema(df[df.c_dis == 1.0], AGING_SCHEMA, "e3ref"))
    tgt = fit_arm(validate_schema(df[df.c_dis == 3.0], AGING_SCHEMA, "e3tgt"))
    # m_est 用 3C/1C；阈值带用 3C 专属（v1.9 m(2C;T) 带口径之外——
    # 这里只验证管线方向性，不套 5C 阈值）
    v = compare_arms(ref, tgt, m_lo=1.0, m_star=1.5, m_hi=2.5)
    return {"m_true": 2.0, "verdict": v.to_dict(),
            "direction_ok": v.m_est > 1.5}
