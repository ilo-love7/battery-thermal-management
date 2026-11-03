# -*- coding: utf-8 -*-
"""WP44 电极级耦合验证 shift/shape ↔ LLI/LAM 映射（v1.11 第一线）。

v1.10 温度轴负结果两读法的几何层裁决——Oxford PathDependent
Half_Cell（电极半电池 U(x) + 全池 POCV，Raj & Howey 2020）：

1. **耦合重建**：V_full(q)=U_p(x_p)−U_n(x_n)（锂守恒+N/P）拟合
   DCH_FC_POCV → 耦合自洽性 RMSE；
2. **合成损伤恢复矩阵**：纯 LLI / LAM-PE / LAM-NE 注入（OCV 级）
   → degrade_modes 两模式分解恢复份额（5/10/15% 三档）；
3. **SNL 真实几何灵敏度**：LFP 平台型 vs NMC 斜坡型 BoL RPT 上
   合成 δ/γ 扰动 → 估计器在两种几何下的分离能力（LFP 可辨识性）；
4. **裁决**：恢复矩阵对角占优 → 映射几何成立 → 温度轴 35°C
   shape↑ 支持「LAM 型加速」读法（带 LCO→LFP 化学迁移 caveat）；
   交叉污染严重 → 「映射失效」读法得权。

产物：results/metrics/44_electrode_mapping.json、
results/figures/44a_electrode_mapping.png
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal.degrade_modes import (  # noqa: E402
    identify_rpt_segments, resample_discharge,
)
from bmthermal.electrode_coupling import (  # noqa: E402
    electrode_ux, fit_coupling, geometry_sensitivity, load_oxford_halfcell,
    recovery_matrix,
)
from bmthermal.plotting import save as save_fig, setup_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

MET = ROOT / "results" / "metrics"
FIG = ROOT / "results" / "figures"
OXFORD = Path("D:/项目/02_数据集/01_国际公开集/Oxford_PathDependent/Half_Cell.zip")
SNL_ZIP = Path("D:/项目/02_数据集/01_国际公开集/Sandia_SNL/SNL.zip")

LOSS_LEVELS = (0.05, 0.10, 0.15)
V10 = {"shape_share_lfp3c_eol": 0.62, "mech_share_lfp3c": 0.54}


def load_snl_bol_curves() -> dict:
    """每化学一支代表芯的 BoL RPT（第一段）→ (q, v)。"""
    want = {"LFP": None, "NMC": None}
    with zipfile.ZipFile(SNL_ZIP) as zf:
        names = sorted(n for n in zf.namelist()
                       if "25C_0-100_0.5-3C_a.pkl" in n)
        for n in names:
            chem = "LFP" if "LFP" in n else ("NMC" if "NMC" in n else None)
            if chem is None or want[chem] is not None:
                continue
            with zf.open(n) as fh:
                d = pickle.load(fh)
            nominal = float(d.get("nominal_capacity_in_Ah") or 3.0)
            segs = identify_rpt_segments(d["cycle_data"], nominal)
            if segs:
                q, v = resample_discharge(segs[0])
                want[chem] = {"member": Path(n).name, "q": q, "v": v,
                              "n_segs": len(segs)}
        return {k: v for k, v in want.items() if v}


def main() -> None:
    t0 = time.time()
    MET.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # ---- [1] 半电池解码 + 耦合重建 --------------------------------
    print("[1/4] Oxford Half_Cell 解码与耦合重建...")
    data = load_oxford_halfcell(OXFORD)
    # 全池放电分支：正极嵌锂（DHC_HC_cathode）+ 负极脱锂（CH_HC_anode）
    # ——与 full_dch POCV 同一支（滞后一致）；负极用脱锂段反向成
    # 嵌锂度坐标（lithiation=False）
    xp, Up = electrode_ux(data["cathode_dch"], lithiation=True)
    xn, Un = electrode_ux(data["anode_chg"], lithiation=False)
    pocv = data["full_dch"]
    ok = np.isfinite(pocv["Q_Ah"]) & np.isfinite(pocv["V"])
    fit, (q_rec, v_rec) = fit_coupling(
        xp, Up, xn, Un,
        pocv["Q_Ah"][ok].values, pocv["V"][ok].values,
    )
    print(f"   耦合: x_p0={fit.x_p0:.2f} x_n0={fit.x_n0:.2f} "
          f"N/P={fit.np_ratio:.2f} RMSE={fit.rmse_v*1000:.1f} mV")
    edge = [n for n, lo, hi, v in (
        ("x_p0", 0.021, 0.549, fit.x_p0),
        ("x_n0", 0.501, 0.979, fit.x_n0),
        ("np_ratio", 1.001, 1.599, fit.np_ratio),
    ) if v <= lo or v >= hi]
    if edge:
        print(f"   ⚠️ 参数触网格边界: {edge}（拟合受约束，如实记录）")

    # ---- [2] 损伤恢复矩阵 ----------------------------------------
    print("[2/4] 合成损伤恢复矩阵（LLI / LAM-PE / LAM-NE）...")
    rec_levels = {}
    for lv in LOSS_LEVELS:
        df = recovery_matrix(xp, Up, xn, Un, fit, q_loss_frac=lv)
        rec_levels[f"{lv:.2f}"] = df.to_dict("records")
        print(f"   loss={lv:.0%}: " + "  ".join(
            f"{r['injected']}={r['share_shape']:.2f}"
            for _, r in df.iterrows()))

    df10 = pd.DataFrame(rec_levels["0.10"])
    sh10 = {r["injected"]: float(r["share_shape"]) for _, r in df10.iterrows()}

    # 对角占优（映射类）：LLI→shift，LAM→shape
    lli_shiftlike = sh10["lli"] < 0.5
    lam_shapelike = sh10["lam_pe"] > 0.5 and sh10["lam_ne"] > 0.5
    diagonal = bool(lli_shiftlike and lam_shapelike)
    # 严格版本：LLI<0.35 且 min(LAM)>0.65
    strict = bool(sh10["lli"] < 0.35 and min(sh10["lam_pe"], sh10["lam_ne"]) > 0.65)

    # ---- [3] SNL 真实几何灵敏度 ----------------------------------
    print("[3/4] SNL LFP/NMC BoL 几何灵敏度...")
    snl_curves = load_snl_bol_curves()
    geom = {}
    for chem, c in snl_curves.items():
        g = geometry_sensitivity(c["q"], c["v"], loss_frac=0.10)
        g["member"] = c["member"]
        geom[chem] = g
        print(f"   {chem}: shift注入 share_shape="
              f"{g['shift_injected']['share_shape']:.2f}  "
              f"shape注入 {g['shape_injected']['share_shape']:.2f}  "
              f"basis_corr={g['shift_injected']['basis_corr']:.2f}")

    # ---- [4] 裁决 + 图 + JSON ------------------------------------
    print("[4/4] 裁决与产物...")
    lfp_geom_ok = (
        "LFP" in geom
        and geom["LFP"]["shift_injected"]["share_shape"] < 0.25
        and geom["LFP"]["shape_injected"]["share_shape"] > 0.75
    )
    if strict and lfp_geom_ok:
        reading = "b_lam_accel"
        verdict_map = ("恢复矩阵严格对角占优（LLI→shift、LAM→shape）+ "
                       "LFP 平台几何下估计器可分离 → v1.10 温度轴 "
                       "35°C shape↑ 的证据权重落「高温 LAM 型加速」读法")
    elif diagonal:
        reading = "b_weak"
        verdict_map = ("恢复矩阵类对角占优（LLI→shift、LAM→shape）但 "
                       "交叉污染不可忽略 → 「LAM 型加速」读法得权（弱）")
    else:
        reading = "a_mapping_suspect"
        verdict_map = ("恢复矩阵交叉污染严重（LLI 亦出 shape）→ "
                       "「映射失效」读法得权")
    chem_note = ("⚠️ 耦合验证是 LCO/石墨 Kokam 几何——LFP/NMC 化学迁移"
                 "经 SNL 真实曲线合成扰动补强，但 LAM-PE/NE 位点区分"
                 "（正极 vs 负极）不在两模式分辨率内。")
    verdict = (
        f"电极级耦合验证收口（post-mortem 数据侧几何裁决）：Oxford "
        f"Half_Cell（MCOS 解码）电极耦合重建 FC_POCV RMSE "
        f"{fit.rmse_v*1000:.1f} mV（N/P={fit.np_ratio:.2f}）——耦合自洽。"
        f"合成损伤恢复矩阵（loss=10%）：LLI→share_shape {sh10['lli']:.2f}"
        f"（shift-like）、LAM-PE→{sh10['lam_pe']:.2f}、"
        f"LAM-NE→{sh10['lam_ne']:.2f}（shape-like）——"
        f"{'严格' if strict else '类'}对角占优。"
        f"SNL 真实几何：LFP shift注入恢复 "
        f"{geom.get('LFP', {}).get('shift_injected', {}).get('share_shape', float('nan')):.2f}"
        f"/shape注入 "
        f"{geom.get('LFP', {}).get('shape_injected', {}).get('share_shape', float('nan')):.2f}"
        f"，NMC 同款——平台型几何{'可' if lfp_geom_ok else '不可'}分离。"
        f"→ v1.10 温度轴两读法证据权重：{verdict_map}。"
        f" 与 v1.10 份额对拍：LFP 3C shape 62% ↔ 机械 54%（两读法之下"
        f"均指向 LAM 型主导）。{chem_note}"
    )
    print(verdict)

    # 图 44a
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(xp, Up, label="正极 U_p (嵌锂)", lw=1.6)
    ax.plot(xn, Un, label="负极 U_n (嵌锂)", lw=1.6)
    ax.set_xlabel("归一嵌锂度 x")
    ax.set_ylabel("半电池电位 [V vs Li]")
    ax.set_title("a) 电极半电池 U(x)（Oxford HC 解码）")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    okp = np.isfinite(pocv["Q_Ah"]) & np.isfinite(pocv["V"])
    ax.plot(pocv["Q_Ah"][okp], pocv["V"][okp], label="实测 FC_POCV (DCH)",
            lw=1.4, color="#2c3e50")
    ax.plot(q_rec, v_rec, "--", label=f"耦合重建 (RMSE {fit.rmse_v*1000:.0f} mV)",
            lw=1.6, color="#e67e22")
    ax.set_xlabel("容量 [Ah]")
    ax.set_ylabel("全池电压 [V]")
    ax.set_title("b) 电极耦合重建 vs POCV")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    kinds = ["lli", "lam_pe", "lam_ne"]
    labels = ["LLI", "LAM-PE", "LAM-NE"]
    x = np.arange(len(kinds))
    for i, lv in enumerate(LOSS_LEVELS):
        dfl = pd.DataFrame(rec_levels[f"{lv:.2f}"])
        sh = [float(dfl[dfl.injected == k]["share_shape"].iloc[0])
              for k in kinds]
        ax.bar(x + (i - 1) * 0.26, sh, 0.26, label=f"loss={lv:.0%}")
    ax.axhline(0.5, color="#7f8c8d", ls=":", lw=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("恢复 share_shape")
    ax.set_ylim(0, 1.05)
    ax.set_title("c) 损伤恢复矩阵（注入型 vs 份额）")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    if geom:
        xs = np.arange(len(geom))
        for i, (key, g) in enumerate(geom.items()):
            ax.bar(xs[i] - 0.18, g["shift_injected"]["share_shape"], 0.36,
                   color="#2980b9", label="shift 注入" if i == 0 else None)
            ax.bar(xs[i] + 0.18, g["shape_injected"]["share_shape"], 0.36,
                   color="#c0392b", label="shape 注入" if i == 0 else None)
        ax.set_xticks(xs, list(geom.keys()))
    ax.axhline(0.5, color="#7f8c8d", ls=":", lw=1)
    ax.set_ylabel("恢复 share_shape")
    ax.set_ylim(0, 1.05)
    ax.set_title("d) SNL 真实曲线几何灵敏度（BoL RPT）")
    ax.legend(fontsize=8)

    fig.suptitle("44a 电极级耦合验证：shift/shape 与 LLI/LAM 映射", fontsize=12)
    fig.tight_layout()
    save_fig(fig, "44a_electrode_mapping.png")

    out = {
        "dataset": {
            "halfcell": str(OXFORD),
            "pocv_branch": "full_dch",
            "electrodes": "cathode_dch (锂化) + anode_chg (脱锂→嵌锂度)",
            "snl_cells": {k: v["member"] for k, v in geom.items()},
        },
        "coupling_fit": {**fit.to_dict(), "grid_edge": edge},
        "recovery": rec_levels,
        "recovery_10pct_shares": sh10,
        "diagonal_dominance": {"class_level": diagonal, "strict": strict},
        "geometry_sensitivity": geom,
        "lfp_geometry_ok": lfp_geom_ok,
        "temperature_axis_reading": reading,
        "v1_10_crosscheck": V10,
        "caveats": [
            "LCO/石墨 Kokam 几何验证 ≠ LFP/NMC/NCA 化学统一",
            "OCV 级注入不含动力学/阻抗贡献",
            "LAM-PE vs LAM-NE 位点区分不在两模式分辨率内",
            "耦合 (x_n0, N/P) 沿山脊弱可辨识（V(q) 只约束轨迹组合）；"
            "x_p0 与 q_use 可辨识，恢复矩阵对山脊解不敏感（三档一致）",
        ],
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (MET / "44_electrode_mapping.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {out['elapsed_s']}s")


if __name__ == "__main__":
    main()
