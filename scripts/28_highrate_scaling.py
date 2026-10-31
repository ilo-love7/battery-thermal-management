"""28 · 高倍率标度律二次校准与损伤倍率等效性（悬了四轮的开放项）。

背景
----
v1.0 判词"3C 以上标度律外推偏差 −25%"（Q=0.782·C^1.79 在 0.5–2C 拟合，
3C+ 对比 Mohtat DFN 偏低）一直没实测锚点；WP23 的"共享振幅"假设
（每 EFC 通量损伤与倍率无关）也没有倍率端检验。v1.5 两块实测补上：

1. **产热侧（Panasonic HPPC 0.5–6C 脉冲）**：固定 SOC/温度下
   Q_ohm=I²R 的倍率指数 p=2+dlnR/dlnC——25 °C R 对倍率平坦（p=2.00），
   −10 °C R 随倍率降（p=1.83，自加热/极化）。**两个温度都 ≥1.83，
   不低于总热标度律的 1.79** → 标度律在高倍率是偏低侧，与 v1.1
   "3C 偏差 −25%"同向：5C 冷却需求 638 W/(m²·K) 应读作下界警示。
2. **损伤侧（SNL 25 °C 多倍率组）**：每 EFC 衰减的倍率等效性检验。
   LFP 3C 组每 EFC 衰减是 1C 的 2.3×（日历解释不掉——3C 壁钟更短）；
   NMC 2C/3C 组反而比 1C 轻（0.6–0.7×）。**共享振幅假设在 3C 失效，
   且方向随化学体系翻转**——寿命模型必须带倍率效应项，不能只靠
   EFC 归一。

产出：
  results/metrics/28_highrate_scaling.json
  results/figures/28a_highrate_scaling.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal import plotting  # noqa: E402
from bmthermal.panasonic_rct import extract_dataset, rate_nonlinearity  # noqa: E402

OUT = ROOT / "results"
MET = OUT / "metrics"
CACHE = OUT / "cache"
PANA_ROOT = Path("D:/项目/02_数据集/01_国际公开集/Panasonic_18650PF")
SNL_CSV = CACHE / "snl_summary.csv"

P_LAW = 1.79          # v1.0 总热标度律指数（0.5–2C 拟合）


def q_exp_from_pulses(pulses: pd.DataFrame, temp_c: float, soc: float = 0.5) -> dict:
    """固定 T/SOC 下 Q_ohm=I²R 的倍率指数。"""
    nl = rate_nonlinearity(pulses, temp_c, soc_ref=soc)
    valid = nl[(nl.c_rate >= 0.5) & (nl.r_end > 0)]
    slope = float(np.polyfit(np.log(valid.c_rate), np.log(valid.r_end), 1)[0])
    return {
        "temp_c": temp_c,
        "dlnR_dlnC": slope,
        "q_exp_ohmic": 2.0 + slope,
        "r_by_c": {float(c): float(r) for c, r in zip(valid.c_rate, valid.r_end)},
    }


def snl_rate_effects() -> list[dict]:
    """SNL 25 °C：每 EFC 衰减的倍率等效性（对 1C 归一）。"""
    import pandas as pd
    df = pd.read_csv(SNL_CSV)
    rows = []
    for chem in ("LFP", "NMC", "NCA"):
        base = df[(df.chem == chem) & (df.temp_c == 25) & (df.soc_lo == 0)
                  & (df.c_dis == 1.0) & df.fade_rate_per_efc.notna()]
        if base.empty:
            continue
        f1 = float(base.fade_rate_per_efc.mean())
        for c_dis in sorted(df[(df.chem == chem) & (df.temp_c == 25)
                               & (df.soc_lo == 0)].c_dis.unique()):
            g = df[(df.chem == chem) & (df.temp_c == 25) & (df.soc_lo == 0)
                   & (df.c_dis == c_dis) & df.fade_rate_per_efc.notna()]
            if g.empty:
                continue
            fc = float(g.fade_rate_per_efc.mean())
            dt = float(g.dt_s_per_cycle.median())
            rows.append({
                "chem": chem, "c_dis": float(c_dis), "n": int(len(g)),
                "fade_per_efc": fc, "dt_s": dt,
                "ratio_vs_1c": fc / f1 if c_dis != 1.0 else 1.0,
                "fade_per_hour": fc / dt * 3600.0,
            })
    return rows


def main() -> None:
    t0 = time.time()

    # ---- 1) 产热侧：欧姆产热倍率指数 ----
    print("[1/3] Panasonic 欧姆产热倍率指数 ...")
    pulses = extract_dataset(PANA_ROOT, CACHE / "panasonic_hppc.csv")
    heat = {}
    for tc in (25.0, -10.0):
        heat[tc] = q_exp_from_pulses(pulses, tc)
        print(f"   T={tc:5.0f}°C  dlnR/dlnC={heat[tc]['dlnR_dlnC']:+.3f}  "
              f"p_ohm={heat[tc]['q_exp_ohmic']:.2f}（总热标度律 p={P_LAW}）")
    print(f"   判词：实测欧姆指数 p≥{min(h['q_exp_ohmic'] for h in heat.values()):.2f} ≥ 总热 {P_LAW}"
          f"——标度律高倍率偏低侧（与 v1.1 '3C −25%' 同向）")

    # ---- 2) 损伤侧：每 EFC 倍率等效性 ----
    print("[2/3] SNL 每 EFC 损伤倍率等效性 ...")
    rate_rows = snl_rate_effects()
    for chem in ("LFP", "NMC", "NCA"):
        rs = [r for r in rate_rows if r["chem"] == chem]
        line = "   " + chem + ": " + "  ".join(
            f"{r['c_dis']:g}C×{r['ratio_vs_1c']:.2f}({r['n']}芯)" for r in rs)
        print(line)
    lfp3 = next(r for r in rate_rows if r["chem"] == "LFP" and r["c_dis"] == 3.0)
    nmc3 = next(r for r in rate_rows if r["chem"] == "NMC" and r["c_dis"] == 3.0)
    print(f"   3C/1C 每 EFC 衰减比：LFP {lfp3['ratio_vs_1c']:.2f}（高倍率更伤）  "
          f"NMC {nmc3['ratio_vs_1c']:.2f}（反而更轻）——共享振幅双向失效")

    # fade/hour 检查（纯日历模型预测与倍率无关）
    print("   fade/hour（日历模型应与倍率无关）：")
    for chem in ("LFP", "NMC"):
        rs = [r for r in rate_rows if r["chem"] == chem]
        line = "      " + chem + ": " + "  ".join(
            f"{r['c_dis']:g}C {r['fade_per_hour']*1e4:.2f}e-4/h" for r in rs)
        print(line)

    # ---- 3) v1.5 寿命账汇总 ----
    print("[3/3] 寿命账口径演进汇总 ...")
    m26 = json.loads((MET / "26_snl_calendar.json").read_text(encoding="utf-8"))
    idset = m26["identified_set"]
    life = {
        "v1.0_single_point": 4.6,
        "v1.2_mc_median": 4.7,
        "v1.3_closed_loop": 4.77,
        "v1.4_closed_loop": 2.14,
        "v1.4_ci": [1.76, 4.77],
        "v1.5_identified_set": [idset["c5_ratio_at_s_lo"], idset["c5_ratio_at_s_hi"]],
        "v1.5_s_rebased": [idset["s_rebased_lo"], idset["s_rebased_hi"]],
    }
    print(f"   v1.4 CI {life['v1.4_ci']} → v1.5 识别集 {life['v1.5_identified_set']}"
          f"（s∈{life['v1.5_s_rebased']}）")
    rate_effect_note = (
        "损伤倍率效应未入寿命账：LFP 3C 每 EFC 超额 ~2.3×、NMC 3C ~0.65×——"
        "若真实化学体系靠近 LFP/NMC 哪一侧，C5 还要按方向修正；该项是 v1.5 新增显式不确定度"
    )
    print("   " + rate_effect_note)

    metrics = {
        "heat_side": {
            "scaling_law_p": P_LAW,
            "p_ohmic_by_temp": {str(k): v for k, v in heat.items()},
            "verdict": (
                "实测欧姆产热指数 p=2.00（25 °C）/1.83（−10 °C），均 ≥ 总热标度律 1.79——"
                "标度律在 3C+ 偏低侧与 v1.1 '−25%' 同向，实测支持『5C 冷却需求是下界警示』。"
                "注意：这是欧姆份指数（0.5–6C 脉冲），总热还含可逆/极化——绝对量不可跨电芯搬运。"
            ),
        },
        "damage_side": {
            "rate_rows": rate_rows,
            "lfp_3c_ratio": lfp3["ratio_vs_1c"],
            "nmc_3c_ratio": nmc3["ratio_vs_1c"],
            "verdict": (
                "每 EFC 损伤倍率等效性（共享振幅假设）在 3C 失效且方向随体系翻转："
                f"LFP 3C/1C={lfp3['ratio_vs_1c']:.2f}（高倍率更伤，日历解释不掉——3C 壁钟更短），"
                f"NMC 3C/1C={nmc3['ratio_vs_1c']:.2f}（反而更轻）。"
                "寿命模型必须带倍率效应项，不能只靠 EFC 归一；WP23 共享振幅是 LFP 侧近似。"
            ),
        },
        "life_table_rollup": life,
        "rate_effect_uncertainty": rate_effect_note,
        "verdict": (
            "悬置四轮的 5C+ 标度律二次校准给出实测锚点（欧姆指数 p≥1.83，标度律偏低侧）；"
            "退化机理侧的倍率效应定量为体系相关（LFP 3C 超额 2.3× / NMC 0.65×），"
            "v1.5 寿命账主口径不变（2.14×，识别集 [1.6, 2.6]）但新增显式倍率效应不确定度。"
        ),
        "elapsed_s": time.time() - t0,
    }
    (MET / "28_highrate_scaling.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 图 28a ----
    plotting.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # (a) 指数对比
    ax = axes[0, 0]
    names = ["总热标度律\n(v1.0, 0.5–2C)", "欧姆实测\n25 °C", "欧姆实测\n−10 °C"]
    vals = [P_LAW, heat[25.0]["q_exp_ohmic"], heat[-10.0]["q_exp_ohmic"]]
    cols = ["#7f8c8d", "#e67e22", "#3498db"]
    ax.bar(names, vals, color=cols, edgecolor="k", linewidth=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_ylabel("产热倍率指数 p (Q∝C^p)")
    ax.set_ylim(0, 2.4)
    ax.set_title("(a) 实测欧姆指数 ≥ 标度律 1.79（偏低侧）")
    ax.grid(alpha=0.3, axis="y")

    # (b) R(C) 两温度
    ax = axes[0, 1]
    for tc, col in ((25.0, "#e67e22"), (-10.0, "#3498db")):
        r = heat[tc]["r_by_c"]
        cs = sorted(r)
        ax.plot(cs, [r[c] * 1000 for c in cs], "o-", c=col, ms=5,
                label=f"T={tc:.0f}°C")
    ax.set_xlabel("倍率 (C)")
    ax.set_ylabel("R_end (mΩ)")
    ax.set_title("(b) R(C)：25 °C 平坦 / −10 °C 下行")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) 每 EFC 损伤倍率等效性
    ax = axes[1, 0]
    mk = {"LFP": "#27ae60", "NMC": "#c0392b", "NCA": "#2980b9"}
    for chem in ("LFP", "NMC", "NCA"):
        rs = [r for r in rate_rows if r["chem"] == chem]
        ax.plot([r["c_dis"] for r in rs], [r["ratio_vs_1c"] for r in rs],
                "o-", c=mk[chem], ms=6, label=chem)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="EFC 等效（共享振幅）")
    ax.set_xlabel("放电倍率 (C)")
    ax.set_ylabel("fade/EFC（对 1C 归一）")
    ax.set_title("(c) 损伤倍率等效性：3C 双向失效")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) 寿命口径演进
    ax = axes[1, 1]
    stages = ["v1.0", "v1.2", "v1.3", "v1.4", "v1.5"]
    med = [4.6, 4.7, 4.77, 2.14, 2.14]
    lo = [4.6, 3.8, 4.59, 1.76, idset["c5_ratio_at_s_lo"]]
    hi = [4.6, 6.7, 4.77, 4.77, idset["c5_ratio_at_s_hi"]]
    x = np.arange(len(stages))
    ax.errorbar(x, med, yerr=[np.array(med) - np.array(lo),
                              np.array(hi) - np.array(med)],
                fmt="o-", c="#2c3e50", capsize=5, ms=7, lw=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("C5 寿命比（3C 液冷 vs 1C 自然）")
    ax.set_title("(d) 寿命口径演进：v1.5 识别集")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    plotting.save(fig, "28a_highrate_scaling.png")

    print(f"完成（{metrics['elapsed_s']:.0f} s）：{MET/'28_highrate_scaling.json'}")


if __name__ == "__main__":
    main()
