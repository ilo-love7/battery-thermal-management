# -*- coding: utf-8 -*-
"""WP46 实验-开放项决策解析图 + v1.11 收口（第三线）。

把 v1.10 后的未决问题（宽度/二元）与三项硬件实验（45）做解析
映射——信息期望值（EVI）代理 = 未决带宽收缩量（非货币口径，
如实标注）：

- Q1 LFP 5C 方向（带宽 1.05）→ E1 5C 台架
- Q2 温度轴两读法（二元）→ E2 post-mortem（v1.44 几何证据已
  给读法 b 加权，物理点名仍需 E2）
- Q3 m(3C;T) 代理带（LFP 带宽 3.09）→ E3 3C/5C×T 组
- Q4 材料位点（正极颗粒/集流体，二元未知）→ E2
- Q5 LAM-PE vs LAM-NE 位点（v1.11 两模式分辨率边界）→ E2

产物：results/metrics/46_decision_map.json、
results/figures/46a_decision_map.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal.plotting import save as save_fig, setup_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

MET = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures"


def main() -> None:
    t0 = time.time()
    MET.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    m40 = json.loads((MET / "40_5c_refine.json").read_text(encoding="utf-8"))
    m42 = json.loads((MET / "42_rate_temp_surface.json").read_text(encoding="utf-8"))
    m44 = json.loads((MET / "44_electrode_mapping.json").read_text(encoding="utf-8"))
    m45 = json.loads((MET / "45_exp_design.json").read_text(encoding="utf-8"))

    # ---- 未决问题与带宽 -------------------------------------------
    m5 = m40.get("refined_bands", {}).get("LFP", [2.30, 3.35])
    if isinstance(m5, dict):
        band5 = m5.get("band") or m5.get("m_band") or [2.30, 3.35]
    else:
        band5 = list(m5)
    proxy = m42["m3c_t_proxy_bands"]["LFP"]["t_proxy_band_robust"]
    questions = [
        {"id": "Q1", "name": "LFP 5C 方向（m 带）",
         "width_before": float(band5[1] - band5[0]),
         "kind": "interval", "anchor": f"[{band5[0]:.2f}, {band5[1]:.2f}]"},
        {"id": "Q2", "name": "温度轴两读法（映射失效 vs LAM 加速）",
         "width_before": 1.0, "kind": "binary",
         "anchor": f"v1.44 几何证据倾向读法 b（{m44['temperature_axis_reading']}）"},
        {"id": "Q3", "name": "m(3C;T) 代理带（LFP）",
         "width_before": float(proxy[1] - proxy[0]),
         "kind": "interval", "anchor": f"[{proxy[0]:.2f}, {proxy[1]:.2f}]（2C 代理）"},
        {"id": "Q4", "name": "材料位点（颗粒开裂/集流体）",
         "width_before": 1.0, "kind": "binary", "anchor": "未知"},
        {"id": "Q5", "name": "LAM-PE vs LAM-NE 位点分辨",
         "width_before": 1.0, "kind": "binary",
         "anchor": "v1.44 两模式分辨率边界（不区分 PE/NE）"},
    ]

    # ---- 实验解析映射（EVI 代理 = 带宽收缩） ----------------------
    m_se_min = min(m45["bench_5c"]["m_se_by_n"].values())
    w1_after = 2 * 1.96 * m_se_min   # 5C 台架后 m CI 全宽
    experiments = [
        {"id": "E1", "name": "5C 台架 m(c) 实测",
         "resolves": {"Q1": min(1.0, 1 - w1_after / questions[0]["width_before"])},
         "n_cost": f"{m45['bench_5c']['n_per_arm_power80']}×2 臂"},
        {"id": "E2", "name": "post-mortem 拆解（8 牺牲芯）",
         "resolves": {"Q2": 1.0, "Q4": 1.0, "Q5": 0.8},
         "n_cost": f"{m45['postmortem_plan']['total_sacrifice']} 芯"},
        {"id": "E3", "name": "3C/5C×多温度老化组",
         "resolves": {"Q3": 1 - 1.0 / questions[2]["width_before"]},
         "n_cost": f"{m45['rate_temp_matrix']['total_cells']} 芯"},
    ]
    for e in experiments:
        e["evi_proxy"] = float(sum(
            questions[[q["id"] for q in questions].index(qid)]["width_before"] * frac
            for qid, frac in e["resolves"].items()))
        e["after"] = {qid: questions[[q["id"] for q in questions].index(qid)]["width_before"] * (1 - frac)
                      for qid, frac in e["resolves"].items()}
    ranked = sorted(experiments, key=lambda e: -e["evi_proxy"])

    # ---- 裁决 + 图 + JSON -----------------------------------------
    rank_txt = " > ".join(
        f"{e['id']} {e['name'].split('（')[0]}（{e['evi_proxy']:.2f}）"
        for e in ranked)
    verdict = (
        f"决策解析图收口（v1.11 第三线）：未决 5 问映射 3 实验——"
        f"{rank_txt}。"
        f"E2 一次裁决 3 问（两读法/材料位点/PE-NE 分辨）；"
        f"E1 把 LFP 5C 带 [{band5[0]:.2f},{band5[1]:.2f}] 收到 CI "
        f"±{1.96*m_se_min:.2f}；"
        f"E3 消掉 2C 代理带（宽 {proxy[1]-proxy[0]:.2f}）。"
        f"EVI 代理=带宽收缩口径（非货币），先后仅作参考——E2 依赖"
        f"破坏性拆解窗口，实际排期常与 E1 并行。"
    )
    print(verdict)

    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    qids = [q["id"] for q in questions]
    names = [f"{q['id']} {q['name'][:12]}" for q in questions]
    M = np.zeros((len(experiments), len(questions)))
    for i, e in enumerate(experiments):
        for j, q in enumerate(questions):
            M[i, j] = e["resolves"].get(q["id"], 0.0)
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(qids)), qids)
    ax.set_yticks(range(len(experiments)),
                  [f"{e['id']} {e['name'][:14]}" for e in experiments])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 0:
                ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center",
                        fontsize=9, color="white" if M[i, j] > 0.55 else "#2c3e50")
    ax.set_title("a) 实验 × 未决问题 解析强度")
    fig.colorbar(im, ax=ax, fraction=0.046, label="解析比例")

    ax = axes[1]
    order = ranked
    xs = np.arange(len(order))
    ax.barh(xs, [e["evi_proxy"] for e in order], color=["#c0392b", "#e67e22", "#2980b9"])
    ax.set_yticks(xs, [f"{e['id']}  {e['name'][:16]}" for e in order])
    ax.invert_yaxis()
    ax.set_xlabel("EVI 代理（带宽收缩加权）")
    ax.set_title("b) 实验优先级（宽度口径）")
    for i, e in enumerate(order):
        ax.text(e["evi_proxy"] + 0.02, i, e["n_cost"], va="center", fontsize=8)

    fig.suptitle("46a 实验-开放项决策解析图", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "46a_decision_map.png")

    out = {
        "questions": questions,
        "experiments": experiments,
        "ranking": [e["id"] for e in ranked],
        "evi_note": "EVI 代理 = 未决带宽收缩（非货币口径）",
        "sources": ["40_5c_refine.json", "42_rate_temp_surface.json",
                    "44_electrode_mapping.json", "45_exp_design.json"],
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (MET / "46_decision_map.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {out['elapsed_s']}s")


if __name__ == "__main__":
    main()
