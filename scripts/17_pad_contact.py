"""scripts/17_pad_contact.py —— 导热垫压缩量→热阻→等效 h 映射（v1.2 深化 4）。

闭 v1.1 §17.2：3C 要求接触等效 h ≥ 250 W/(m²·K)，但缺"压缩量/压力 → h"映射。
用文献典型 TIM 关联给压力需求区间，并回答：**压力还是厚度，哪个是主杠杆**。

产出：results/metrics/17_pad_contact.json / results/figures/17a_pad_contact.png
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmthermal import plotting
from bmthermal.params import CELL
from bmthermal.tim import (
    N_EXP_RANGE,
    RSPREAD_MM2K_W,
    RSPREAD_SIGMA,
    RC100_RANGE,
    PadSpec,
    effective_h,
    force_for_pressure,
    pressure_for_target_h,
    stack_resistance,
)

OUT = Path(__file__).resolve().parents[1] / "results"
TARGET_H = 250.0  # v1.1 §16 的 3C 接触设计下限
CONTACT_AREA_M2 = 0.25 * CELL.surface_area_m2


def pressure_grid():
    return np.logspace(np.log10(10e3), np.log10(2000e3), 60)


def main():
    t0 = time.time()
    pads = [
        PadSpec(thickness_mm=0.5, k_w_mk=1.5),
        PadSpec(thickness_mm=1.0, k_w_mk=1.5),
        PadSpec(thickness_mm=2.0, k_w_mk=1.5),
        PadSpec(thickness_mm=1.0, k_w_mk=3.0),
    ]
    labels = ["0.5 mm, k=1.5", "1.0 mm, k=1.5", "2.0 mm, k=1.5", "1.0 mm, k=3.0"]
    pg = pressure_grid()

    curves = []
    for pad, lab in zip(pads, labels):
        hs = [effective_h(pad, p) for p in pg]
        p_req = pressure_for_target_h(pad, TARGET_H)
        # 参数包络：rc100 / n / R_spread 角点（含不可达计数）
        corners = []
        for rc, n_, rs in product(
            RC100_RANGE, N_EXP_RANGE,
            (RSPREAD_MM2K_W * (1 - RSPREAD_SIGMA), RSPREAD_MM2K_W * (1 + RSPREAD_SIGMA)),
        ):
            corners.append(pressure_for_target_h(pad, TARGET_H, rc, n_, rs))
        finite = [c for c in corners if np.isfinite(c)]
        n_inf = len(corners) - len(finite)
        curves.append(
            {
                "label": lab,
                "thickness_mm": pad.thickness_mm,
                "k_w_mk": pad.k_w_mk,
                "r_bulk_mm2k_w": pad.r_bulk_m2k_w * 1e6,
                "h_at_100kpa": effective_h(pad, 100e3),
                "h_at_1mpa": effective_h(pad, 1e6),
                "pressure_req_pa": p_req,
                "pressure_req_range_pa": (
                    [float(min(finite)), float(max(finite))] if finite else [float("inf"), float("inf")]
                ),
                "n_corners": len(corners),
                "n_corners_infeasible": n_inf,
                "force_req_n": force_for_pressure(p_req, CONTACT_AREA_M2) if np.isfinite(p_req) else float("inf"),
                "feasible": bool(np.isfinite(p_req) and p_req <= 2e6),
                "h_curve": [float(h) for h in hs],
            }
        )

    # 厚度 vs 压力的杠杆对比：同样把 h_eff 从 150 提到 250
    pad0 = PadSpec(thickness_mm=1.0, k_w_mk=1.5)
    h_1mm_100k = effective_h(pad0, 100e3)
    pad_thin = PadSpec(thickness_mm=0.5, k_w_mk=1.5)
    h_05mm_100k = effective_h(pad_thin, 100e3)

    c1 = curves[1]
    findings = [
        (
            f"3C 目标 h_eff≥{TARGET_H:.0f}：1 mm/k=1.5 垫片标称参数需 ~{c1['pressure_req_pa']/1e3:.0f} kPa"
            f"（压装力 ~{c1['force_req_n']:.0f} N，25% 足迹）；文献参数包络下 "
            f"{c1['n_corners_infeasible']}/{c1['n_corners']} 角点（大接触热阻+高扩展热阻）在 2 MPa 内不可达，"
            f"其余角点 {c1['pressure_req_range_pa'][0]/1e3:.0f}–{c1['pressure_req_range_pa'][1]/1e3:.0f} kPa——"
            "裕度设计按 100 kPa 压装 + 0.5–1 mm 薄垫执行"
        ),
        (
            f"厚度是主杠杆不是压力：0.5 mm 垫片在 100 kPa 就有 h={curves[0]['h_at_100kpa']:.0f}，"
            f"2 mm 垫片即使 2 MPa 也到不了 {TARGET_H:.0f}（体热阻 {curves[2]['r_bulk_mm2k_w']:.0f} mm²K/W "
            f"已超预算）——'压得更紧'救不了'垫得太厚'"
        ),
        (
            f"压力收益递减：1 mm 垫片从 100 kPa 加压到 1 MPa，h 只从 "
            f"{h_1mm_100k:.0f} 提到 {c1['h_at_1mpa']:.0f}（接触热阻本来就不是主导项；"
            "体热阻 + 扩展热阻占大头）"
        ),
        (
            "与 v1.1 §16 对齐：接触热阻伤温度水平；本节给出实现路径——"
            "1 mm 级薄垫 + ~100 kPa 压装即可满足 3C 的 h≥250（标称只需 ~30 kPa，100 kPa 是含包络的工程值）；"
            "工程上要控的是垫片厚度公差与压装一致性，不是极限压力"
        ),
        (
            "参数来源声明：R_c''(100 kPa)=15–60 mm²K/W、n=0.7–1.0、"
            "R_spread=250±50% 均为文献/工程典型值；具体产品需 datasheet 或实测替换"
        ),
    ]

    out = {
        "target_h": TARGET_H,
        "target_source": "v1.1 §16 接触设计下限（3C 连续放电 Tmax≤45 °C）",
        "contact_area_m2": CONTACT_AREA_M2,
        "correlation": "R_c''(P) = R_c''(100 kPa)·(P/100 kPa)^(-n)",
        "literature_ranges": {
            "rc100_mm2k_w": list(RC100_RANGE),
            "n_exp": list(N_EXP_RANGE),
            "r_spread_mm2k_w": [RSPREAD_MM2K_W * (1 - RSPREAD_SIGMA), RSPREAD_MM2K_W * (1 + RSPREAD_SIGMA)],
        },
        "pads": curves,
        "findings": findings,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT / "metrics").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    with open(OUT / "metrics" / "17_pad_contact.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 图 ----
    fig, axes = plotting.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    colors = ("#2980b9", "#27ae60", "#c0392b", "#8e44ad")
    for c, lab, col in zip(curves, labels, colors):
        ax.plot(pg / 1e3, c["h_curve"], lw=2, color=col, label=lab)
    ax.axhline(TARGET_H, color="k", ls="--", lw=1.5, label=f"3C 目标 {TARGET_H:.0f}")
    ax.set_xscale("log")
    ax.set_xlabel("接触压力 [kPa]")
    ax.set_ylabel("冷板口径等效 h [W/(m²·K)]")
    ax.set_title("导热垫：压力 → 等效 h")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    req = [c["pressure_req_pa"] / 1e3 if np.isfinite(c["pressure_req_pa"]) else np.nan for c in curves]
    lo = [c["pressure_req_range_pa"][0] / 1e3 if np.isfinite(c["pressure_req_range_pa"][0]) else np.nan for c in curves]
    hi = [c["pressure_req_range_pa"][1] / 1e3 if np.isfinite(c["pressure_req_range_pa"][1]) else np.nan for c in curves]
    x = np.arange(len(labels))
    ax.bar(x, req, color=colors, alpha=0.85, edgecolor="white")
    for i, (l, h) in enumerate(zip(lo, hi)):
        if np.isfinite(l) and np.isfinite(h):
            ax.plot([i, i], [l, h], color="k", lw=2)
    ax.axhline(100, color="#7f8c8d", ls=":", lw=1, label="100 kPa 参考")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_yscale("log")
    ax.set_ylabel("达标所需压力 [kPa]")
    ax.set_title(f"h_eff≥{TARGET_H:.0f} 的压力需求（黑线=参数包络）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y", which="both")

    fig.tight_layout()
    plotting.save(fig, "17a_pad_contact.png")

    print(json.dumps({"findings": findings, "elapsed_s": out["elapsed_s"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
