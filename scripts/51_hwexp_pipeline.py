# -*- coding: utf-8 -*-
"""WP54 硬件实验数据侧管线（v1.13 第三线）——§67 判据的管线就绪证明。

§67 已把 E1/E2/E3 的样本量/精度/判据预注册量化；本脚本把
「硬件数据到手 → 裁决」全链代码化并用**合成数据端到端验证**：

1. 数据模式校验（AGING_SCHEMA / POSTMORTEM_SCHEMA）——缺列/坏类型
   如实报错（防静默补数）；
2. E1 5C 台架：合成 1C vs 5C 臂（m_true=2.3，45 的 sd 代理）
   → fade 率拟合 → m 比 CI → decision_5c 预注册判据，多种子
   正确率（期望方向 throughput_positive*）；
3. E2 post-mortem：合成 35 vs 25 °C 配对指标 → decision_postmortem
   两读法裁决，双方向（b 确认 / a 得权）正确率；
4. E3 3C/5C×T：合成 3C vs 1C（m_true=2.0）→ m(c,T) 方向性；
5. 精度门：σ_Q ≤ 0.5955 mAh（0.057% Q_nom）——0.2% 容量计
   超预算负结果的预注册检查函数化。

⚠️ 合成验证只证「管线能判」不证「假设为真」；5C sd 是 3C×1.2
代理，首组真实数据到手须回填重算。

产物：results/metrics/51_hwexp_pipeline.json、
results/figures/51a_hwexp_pipeline.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal.hwexp_pipeline import (  # noqa: E402
    AGING_SCHEMA, POSTMORTEM_SCHEMA, SIGMA_Q_REQ_AH,
    e1_end_to_end, e2_end_to_end, e3_end_to_end,
    sigma_q_check, validate_schema,
)
from bmthermal.plotting import save as save_fig, setup_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT_M = ROOT / "results" / "metrics"
OUT_F = ROOT / "results" / "figures"

N_SEEDS = 40


def schema_negative_tests() -> dict:
    """模式校验负向测试：坏输入必须报错（不静默通过）。"""
    results = {}
    good = pd.DataFrame([{
        "cell_id": "c1", "chem": "LFP", "c_dis": 1.0, "temp_c": 25.0,
        "efc": 0.0, "q_rpt_ah": 1.03},
        {"cell_id": "c1", "chem": "LFP", "c_dis": 1.0, "temp_c": 25.0,
         "efc": 100.0, "q_rpt_ah": 1.02}])
    validate_schema(good, AGING_SCHEMA, "good")     # 应通过
    results["good_passes"] = True

    bad = good.drop(columns=["q_rpt_ah"])
    try:
        validate_schema(bad, AGING_SCHEMA, "bad")
        results["missing_col_caught"] = False
    except ValueError:
        results["missing_col_caught"] = True

    bad2 = good.copy()
    bad2["efc"] = "not_a_number"
    try:
        validate_schema(bad2, AGING_SCHEMA, "bad2")
        results["bad_dtype_caught"] = False
    except ValueError:
        results["bad_dtype_caught"] = True

    bad3 = good.copy()
    bad3.loc[0, "q_rpt_ah"] = np.nan
    try:
        validate_schema(bad3, AGING_SCHEMA, "bad3")
        results["nan_caught"] = False
    except ValueError:
        results["nan_caught"] = True

    pm = pd.DataFrame([{
        "cell_id": "p1", "chem": "LFP", "c_dis": 1.0, "temp_c": 35.0,
        "fade_at_cut": 0.1, "lam_pe": 0.3, "lam_ne": 0.35,
        "crack_pe": 0.3}])                        # 缺 crack_ne 列
    try:
        validate_schema(pm, POSTMORTEM_SCHEMA, "pm")
        results["pm_missing_col_caught"] = False
    except ValueError:
        results["pm_missing_col_caught"] = True

    # 分析层负向：只有 35 °C 行 → analyze_postmortem 应报缺配对
    from bmthermal.hwexp_pipeline import analyze_postmortem
    pm2 = pd.DataFrame([{
        "cell_id": "p1", "chem": "LFP", "c_dis": 1.0, "temp_c": 35.0,
        "fade_at_cut": 0.1, "lam_pe": 0.3, "lam_ne": 0.35,
        "crack_pe": 0.3, "crack_ne": 0.28}])
    try:
        analyze_postmortem(pm2)
        results["pm_unpaired_caught"] = False
    except ValueError:
        results["pm_unpaired_caught"] = True
    return results


def main() -> None:
    OUT_M.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)

    # ---- 1. 模式校验负向测试 ---------------------------------------
    schema_res = schema_negative_tests()
    print("[1] schema negative tests:", schema_res)

    # ---- 2. E1 多种子端到端 ----------------------------------------
    e1_runs = [e1_end_to_end(seed=s, m_true=2.3) for s in range(N_SEEDS)]
    e1_correct = sum(r["correct"] for r in e1_runs)
    e1_ms = [r["verdict"]["m_est"] for r in e1_runs]
    e1_cis = [(r["verdict"]["ci_low"], r["verdict"]["ci_high"])
              for r in e1_runs]
    e1_decisions = pd.Series([r["verdict"]["decision"] for r in e1_runs])
    e1_summary = {
        "m_true": 2.3, "n_seeds": N_SEEDS, "n_cells_per_arm": 5,
        "correct_rate": e1_correct / N_SEEDS,
        "m_est_mean": float(np.mean(e1_ms)),
        "m_est_sd": float(np.std(e1_ms, ddof=1)),
        "ci_width_mean": float(np.mean([b - a for a, b in e1_cis])),
        "decisions": {str(k): int(v)
                      for k, v in e1_decisions.value_counts().items()},
    }
    print(f"[2] E1: correct {e1_correct}/{N_SEEDS}, "
          f"m_est={e1_summary['m_est_mean']:.2f}±{e1_summary['m_est_sd']:.2f}")

    # ---- 3. E2 双方向端到端 ----------------------------------------
    e2_b = [e2_end_to_end(seed=s, direction="b") for s in range(N_SEEDS)]
    e2_a = [e2_end_to_end(seed=s, direction="a") for s in range(N_SEEDS)]
    e2_summary = {
        "n_seeds": N_SEEDS,
        "dir_b_correct_rate": sum(r["correct"] for r in e2_b) / N_SEEDS,
        "dir_a_correct_rate": sum(r["correct"] for r in e2_a) / N_SEEDS,
        "dir_b_decisions": {str(k): int(v) for k, v in pd.Series(
            [r["verdict"]["decision"] for r in e2_b]
        ).value_counts().items()},
        "dir_a_decisions": {str(k): int(v) for k, v in pd.Series(
            [r["verdict"]["decision"] for r in e2_a]
        ).value_counts().items()},
    }
    print(f"[3] E2: dir_b {e2_summary['dir_b_correct_rate']:.0%}, "
          f"dir_a {e2_summary['dir_a_correct_rate']:.0%}")

    # ---- 4. E3 方向性 ----------------------------------------------
    e3_runs = [e3_end_to_end(seed=s) for s in range(N_SEEDS)]
    e3_summary = {
        "m_true": 2.0, "n_seeds": N_SEEDS,
        "direction_ok_rate": sum(r["direction_ok"] for r in e3_runs)
        / N_SEEDS,
        "m_est_mean": float(np.mean([r["verdict"]["m_est"]
                                     for r in e3_runs])),
    }
    print(f"[4] E3: direction_ok {e3_summary['direction_ok_rate']:.0%}, "
          f"m_est={e3_summary['m_est_mean']:.2f}")

    # ---- 5. 精度门 --------------------------------------------------
    gates = {
        "sigma_0p2pct": sigma_q_check(0.002 * 1.039),   # 超预算（45 负结果）
        "sigma_0p057pct": sigma_q_check(SIGMA_Q_REQ_AH),  # 恰在预算
        "sigma_0p01pct": sigma_q_check(1.0e-4),          # 富裕
    }
    print(f"[5] gates: 0.2%→{gates['sigma_0p2pct']['adequate']}, "
          f"budget→{gates['sigma_0p057pct']['adequate']}, "
          f"0.01%→{gates['sigma_0p01pct']['adequate']}")

    # ---- 图 --------------------------------------------------------
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    ax.hist(e1_ms, bins=12, color="#c0392b", alpha=0.75)
    ax.axvline(2.3, color="#2c3e50", lw=1.2, ls="--", label="m_true=2.3")
    ax.axvline(3.1, color="#7f8c8d", lw=1.2, ls=":", label="m*=3.1")
    ax.set_xlabel("m̂ = rate_5C / rate_1C")
    ax.set_ylabel("count")
    ax.set_title(f"E1 synthetic m estimates (n={N_SEEDS} seeds)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    names = ["E1\ncorrect", "E2 dir_b\ncorrect", "E2 dir_a\ncorrect",
             "E3\ndirection"]
    rates = [e1_summary["correct_rate"], e2_summary["dir_b_correct_rate"],
             e2_summary["dir_a_correct_rate"],
             e3_summary["direction_ok_rate"]]
    ax.bar(np.arange(len(names)), rates, 0.55,
           color=["#c0392b", "#2980b9", "#8e44ad", "#27ae60"])
    ax.axhline(0.8, color="#7f8c8d", ls="--", lw=1, label="0.80")
    ax.set_xticks(np.arange(len(names)), names, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("correct rate")
    ax.set_title("pre-registered decision firing")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.axhline(SIGMA_Q_REQ_AH * 1e3, color="#c0392b", lw=1.2,
               label="budget 0.596 mAh")
    vals = [g["sigma_q_meas_ah"] * 1e3 for g in gates.values()]
    labels = ["0.2%\nmeter", "at\nbudget", "0.01%\nmeter"]
    colors = ["#c0392b", "#f39c12", "#27ae60"]
    ax.bar(np.arange(len(vals)), vals, 0.5, color=colors)
    ax.set_xticks(np.arange(len(vals)), labels, fontsize=8)
    ax.set_ylabel("σ_Q [mAh]")
    ax.set_title("measurement precision gate")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    order = sorted(e1_decisions.unique())
    counts = [int((e1_decisions == d).sum()) for d in order]
    ax.barh(np.arange(len(order)), counts, 0.55, color="#2c3e50")
    ax.set_yticks(np.arange(len(order)), order, fontsize=8)
    ax.set_xlabel("count")
    ax.set_title("E1 decision distribution")

    fig.suptitle("51a · hardware experiment pipeline readiness (synthetic)",
                 fontsize=11)
    fig.tight_layout()
    save_fig(fig, "51a_hwexp_pipeline", OUT_F)

    # ---- 判词与指标 -------------------------------------------------
    n_schema_tests = len(schema_res)
    n_schema_caught = sum(1 for v in schema_res.values() if v)
    verdict = (
        f"硬件实验数据侧管线收口（§67 E1/E2/E3 判据代码化）："
        f"数据模式（AGING/POSTMORTEM schema）+ fade 率拟合 + m 比 CI + "
        f"预注册判据（decision_5c / decision_postmortem）全链打通，"
        f"合成端到端 {N_SEEDS} 种子——E1（m_true=2.3，n=5/臂，45 的 "
        f"sd 代理）判对率 {e1_summary['correct_rate']:.0%}，m̂="
        f"{e1_summary['m_est_mean']:.2f}±{e1_summary['m_est_sd']:.2f}，"
        f"CI 宽 {e1_summary['ci_width_mean']:.2f}；"
        f"E2 两读法双方向判对率 {e2_summary['dir_b_correct_rate']:.0%}"
        f"/{e2_summary['dir_a_correct_rate']:.0%}；"
        f"E3（m_true=2.0）方向性 {e3_summary['direction_ok_rate']:.0%}。"
        f"精度门：σ_Q 预算 0.596 mAh（0.057% Q_nom）——0.2% 容量计"
        f"（2.08 mAh）超预算 3.5×（45 负结果函数化，adequate="
        f"{gates['sigma_0p2pct']['adequate']}），0.01% 级达标。"
        f"模式校验负向测试 {n_schema_caught}/{n_schema_tests} 拦截"
        f"（缺列/坏类型/NaN/PM 缺列/PM 缺配对，明细 schema_negative_tests）。"
        f"⚠️ 合成验证只证「管线能判」不证「假设为真」；5C sd 仍是 "
        f"3C×1.2 代理——首组真实 5C 数据到手须回填重算功效；"
        f"E1 m_true=2.3 是带下界情景（m*=3.1 处期望多为不可判），"
        f"判对率口径=方向正确非显著性。"
    )
    print("[6]", verdict[:130], "...")

    metrics = {
        "n_seeds": N_SEEDS,
        "schema_negative_tests": schema_res,
        "e1": e1_summary,
        "e2": e2_summary,
        "e3": e3_summary,
        "precision_gates": gates,
        "sigma_q_req_ah": SIGMA_Q_REQ_AH,
        "caveats": [
            "合成验证只证管线能判，不证假设为真",
            "5C sd 是 3C×1.2 代理（45）——真实数据到手须回填重算",
            "E1 判对率口径=方向正确（m_true=2.3 低于 m*，期望 positive）",
            "E2 指标 lam/crack 为合成代理——真实 SEM/EDS 量纲到手后标定",
        ],
        "verdict": verdict,
    }
    (OUT_M / "51_hwexp_pipeline.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[7] → {OUT_M / '51_hwexp_pipeline.json'}")


if __name__ == "__main__":
    main()
