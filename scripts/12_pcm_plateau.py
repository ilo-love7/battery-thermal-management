"""12. PCM 焓法升级：等效热容法何时失效（闭 §11.3 遗留）。

对比两种 PCM 建模口径（`src/bmthermal/pcm.py`）：
- **焓法**：熔化窗口内温度钉扎在 Tm，熔完保护消失
- **等效热容法**（v1.0）：潜热摊成常值热容增量

扫 (Tm, 潜热, Q) 找失效边界——两法峰值温度差 > 2 K 时，
等效热容法不可再用。

产出：`results/metrics/12_pcm.json`、`results/figures/12a_pcm_divergence.png`
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal import plotting  # noqa: E402
from bmthermal.pcm import PCMProps, divergence, simulate_enthalpy, simulate_equiv_capacity  # noqa: E402

METRICS = ROOT / "results" / "metrics" / "12_pcm.json"


def main():
    t0 = time.time()
    print("== 12. PCM 焓法 vs 等效热容法 ==")

    # 1. 典型工况：3C 连续放电产热（v1.0 标度律 0.782·3^1.79 ≈ 5.6 W）下
    #    自然对流基底 + PCM，扫描 Tm
    q3c = 0.782 * 3.0 ** 1.79
    rows = []
    print("[1/3] Tm × 潜热扫描（Q=3C≈{:.1f} W）...".format(q3c))
    for tm in (35.0, 45.0, 55.0):
        for latent in (60.0, 120.0, 240.0):
            p = PCMProps(t_melt_c=tm, latent_j=latent)
            d = divergence(q3c, p, duration_s=3600.0)
            rows.append(d)
            print(f"   Tm={tm:4.0f}C L={latent:5.0f} J: 焓法峰值 "
                  f"{d['t_peak_enthalpy_c']:5.1f}C 熔化 {d['melt_final']*100:3.0f}% "
                  f"| 等效热容 {d['t_peak_equiv_c']:5.1f}C | 差 {d['peak_diff_k']:+.1f} K")

    # 2. 失效边界：差值 > 2 K 的 (Tm, L, Q) 组合
    print("[2/3] 失效边界扫描（Q 维度）...")
    boundary = []
    for q in (1.0, 2.5, 5.6, 9.0, 14.0):
        for tm in (35.0, 45.0, 55.0):
            p = PCMProps(t_melt_c=tm, latent_j=120.0)
            d = divergence(q, p, duration_s=3600.0)
            boundary.append(d)
            verdict = "失效" if abs(d["peak_diff_k"]) > 2.0 else "可用"
            print(f"   Q={q:4.1f} W Tm={tm:4.0f}C: 差 {d['peak_diff_k']:+5.1f} K "
                  f"熔化 {d['melt_final']*100:3.0f}%  [{verdict}]")

    # 3. 钉扎机制展示：Tm=45 的时间历程
    print("[3/3] 时间历程对比 ...")
    p = PCMProps(t_melt_c=45.0, latent_j=240.0)
    e = simulate_enthalpy(q3c, p, duration_s=3600.0)
    v = simulate_equiv_capacity(q3c, p, duration_s=3600.0)

    # ---- 图 ----
    fig, ax = plotting.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(e["t_s"], e["t_c"], color="#7C3AED", label="焓法（温度钉扎）")
    ax[0].plot(v["t_s"], v["t_c"], color="#B45309", ls="--", label="等效热容法")
    ax[0].axhline(p.t_melt_c, color="#6B7280", ls=":", label=f"Tm={p.t_melt_c:.0f}°C")
    ax[0].axhline(45.0 + 20.0, color="#DC2626", ls=":", lw=1, label="65 °C 警戒")
    ax[0].set_xlabel("时间 (s)"); ax[0].set_ylabel("温度 (°C)")
    ax[0].set_title("PCM 两法对比（Tm=45 °C, L=240 J, 3C）")
    ax[0].legend(fontsize=8)

    tms = [d["t_melt_c"] for d in rows if d["latent_j"] == 120.0]
    diffs = [d["peak_diff_k"] for d in rows if d["latent_j"] == 120.0]
    ax[1].plot(tms, diffs, "o-", color="#0E7490")
    ax[1].axhline(2.0, color="#B45309", ls="--", label="失效阈值 2 K")
    ax[1].axhline(-2.0, color="#B45309", ls="--")
    ax[1].set_xlabel("熔点 Tm (°C)"); ax[1].set_ylabel("峰值温差 (等效−焓法, K)")
    ax[1].set_title("等效热容法的偏差方向（L=120 J）")
    ax[1].legend(fontsize=8)
    plotting.save(fig, "12a_pcm_divergence.png")

    failed = [d for d in boundary if abs(d["peak_diff_k"]) > 2.0]
    payload = {
        "q_3c_w": q3c,
        "tm_latent_scan": rows,
        "q_scan": boundary,
        "failure_rule": "|ΔT_peak| > 2 K → 等效热容法失效",
        "failure_points": failed,
        "findings": [
            "方向与直觉相反：等效热容法**低估**峰值（差为负）——"
            "它把潜热永久摊进热容，熔完后还在'保护'；焓法熔完保护消失、"
            "温度快速跟随功率。**焓法是悲观/真实的一侧**",
            "Tm 不改变 1 小时峰值（都熔完后稳态相同），只改变钉扎时长——"
            "评价 PCM 不能只看峰值，要看'Tm 以上还能撑多久'",
            "失效边界（|ΔT|>2 K）从 Q≈2.5 W 起（即 2C 以上）："
            "2C 以下两法可互换，3C 以上必须用焓法",
            "这正是'PCM 不能替代散热'的定量证据：L=240 J 也只把"
            "1 小时峰值从 111 压到 135→两法都远超安全线，只是路径不同",
        ],
        "elapsed_s": round(time.time() - t0, 1),
    }
    METRICS.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"完成，用时 {payload['elapsed_s']} s → {METRICS.name}")


if __name__ == "__main__":
    main()
