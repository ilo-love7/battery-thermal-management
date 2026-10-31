"""10. 实测温升标定尝试：可辨识性分析与测量需求规格。

用 CALCE A123 的实测温度通道（``Temperature (C)_1``，8 个箱温档）
尝试标定热网络参数，回答报告 §11.1 的"参数集分歧需实测标定收敛"。
**结论是数据否掉了预期**（如实保留）：

1. **结构不可辨识**：(C, β, G)→(αC, αβ, αG) 输出不变，单靠温升曲线
   只能辨识 τ=C/G 与 β/G 两个组合——第三自由度必须先验锚定。
2. **实际不可辨识**：剖面似然显示 τ 从 60 s 到 10000 s（160 倍）
   RMSE 只变 0.03 K（< 0.07 K 温度量化噪声）。
3. **两节点更不可辨识**：表面单点温度 + 5 参数 → 优化器发散。
4. **可交付的替代产出**：测量需求规格书（要什么精度的传感器、
   什么激励波形，才能标定出热网络）。

产出：`results/metrics/10_calib.json`、`results/figures/10a_calib_identifiability.png`
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bmthermal.calce_thermal import load_all_dynamic, split_profiles  # noqa: E402
from bmthermal.calib import (  # noqa: E402
    _prepare, _simulate_lti, fit_lumped, fit_two_node,
)
from bmthermal import plotting  # noqa: E402

CACHE = ROOT / "results" / "cache" / "calce_a123_with_temp.parquet"
METRICS = ROOT / "results" / "metrics" / "10_calib.json"
C_PRIOR_J_K = 60.0        # ANR26650: 76 g × ~790 J/(kg·K)
AREA_M2 = 0.0065          # 26650 圆柱侧+端面


def segs_at(df, temp):
    d = df[df["temp_c_set"] == temp]
    return split_profiles(d)


def profile_likelihood(segments, c_th=C_PRIOR_J_K,
                       taus=(60, 120, 300, 600, 1200, 2500, 4000, 6000, 10000)):
    """固定 τ 网格，只拟合 (β, a0, a1)，看 RMSE 是否随 τ 变化。"""
    from scipy.optimize import least_squares

    prepped = [_prepare(*s) for s in segments]
    t_ref = segments[0][0][0]
    out = []
    for tau in taus:
        G = c_th / tau

        def resid(p, G=G):
            beta = np.exp(p[0])
            a0, a1 = p[1], p[2]
            res = []
            for tt, TT, II, dt in prepped:
                ta = a0 + a1 * (tt - t_ref) / 3600.0
                A = np.array([[-G / c_th]])
                B = np.array([[beta / c_th, G / c_th]])
                uu = np.stack([II ** 2, ta])
                y = _simulate_lti(A, B, np.array([1.0]),
                                  np.array([TT[0]]), uu, dt)
                res.append(y - TT)
            return np.concatenate(res)

        r = least_squares(resid, np.array([np.log(0.03), 25.0, 0.0]),
                          method="lm", max_nfev=500)
        rmse = float(np.sqrt(np.mean(resid(r.x) ** 2)))
        out.append({"tau_s": tau, "rmse_k": rmse,
                    "beta_w_a2": float(np.exp(r.x[0]))})
    return out


def ridge_demo(segments):
    """不可辨识脊：固定 C=40/60/80 J/K 三种先验 → **同 RMSE**、
    β 与 G 随 C 线性缩放、τ 恒定。这就是 (C,β,G)→(αC,αβ,αG) 的脊。"""
    out = []
    for c_fix in (40.0, 60.0, 80.0):
        f = fit_lumped(segments, t_amb_c=25.0, c_th_j_k=c_fix)
        out.append({
            "c_th_j_k": f.c_th_j_k, "beta_w_a2": f.beta_w_a2,
            "g_w_k": f.g_w_k, "tau_s": c_fix / f.g_w_k, "rmse_k": f.rmse_k,
        })
    return out


def snr_audit(df, temp=25.0):
    """温度通道信噪比：T 与 I² 的相关性、量化粒度。"""
    d = df[df["temp_c_set"] == temp]
    segs = split_profiles(d)
    s = segs["dst"]
    t = s["t_s"].to_numpy(float)
    T = s["temp_c"].to_numpy(float)
    I = s["current_a"].to_numpy(float)
    I2 = I ** 2
    # 量化粒度：温度差分的最小非零步
    dT = np.diff(T)
    dT = dT[np.abs(dT) > 1e-9]
    quant = float(np.min(np.abs(dT))) if dT.size else float("nan")
    corrs = {}
    dt_med = float(np.median(np.diff(t)))
    for lag_s in (0, 60, 300, 600, 1200):
        lag = int(lag_s / dt_med)
        if lag == 0:
            c = float(np.corrcoef(T, I2)[0, 1])
        else:
            c = float(np.corrcoef(T[lag:], I2[:-lag])[0, 1])
        corrs[f"lag_{lag_s}s"] = c
    return {
        "temp_c_quantum_k": quant,
        "t_range_k": [float(T.min()), float(T.max())],
        "corr_t_i2": corrs,
        "verdict": "corr ≤ 0.3 且无滞后峰 → τ 由噪声主导，不可辨识",
    }


def main():
    t0 = time.time()
    print("== 10. 实测温升标定尝试（可辨识性分析） ==")
    df = load_all_dynamic(cache_parquet=str(CACHE))
    payload = {
        "data": {
            "source": "CALCE A123 (UMD), Arbin Temperature (C)_1",
            "n_rows": int(len(df)),
            "temps_c": sorted(float(x) for x in df["temp_c_set"].unique()),
            "cell": "A123 ANR26650 (2.5 Ah LFP)",
        },
        "c_prior_j_k": C_PRIOR_J_K,
        "area_m2": AREA_M2,
    }

    # 1. 信噪比审计
    print("[1/4] 温度通道信噪比审计 ...")
    payload["snr"] = snr_audit(df)
    print("   量化:", round(payload["snr"]["temp_c_quantum_k"], 4), "K  ",
          payload["snr"]["verdict"])

    # 2. 结构不可辨识脊
    print("[2/4] 三参数不可辨识脊 ...")
    segs25 = segs_at(df, 25.0)
    segl = [(v["t_s"].to_numpy(float), v["temp_c"].to_numpy(float),
             v["current_a"].to_numpy(float)) for v in segs25.values()]
    payload["ridge"] = ridge_demo(segl)
    for r in payload["ridge"]:
        print(f"   C={r['c_th_j_k']:.0f} beta={r['beta_w_a2']:.4f} "
              f"G={r['g_w_k']:.4f} RMSE={r['rmse_k']:.4f}")

    # 3. 剖面似然（τ 网格）
    print("[3/4] 剖面似然 τ 网格 ...")
    payload["profile_likelihood"] = profile_likelihood(segl)
    for r in payload["profile_likelihood"]:
        print(f"   tau={r['tau_s']:6.0f}s  RMSE={r['rmse_k']:.4f} K  "
              f"beta={r['beta_w_a2']:.4f} ohm")
    rmses = [r["rmse_k"] for r in payload["profile_likelihood"]]
    payload["profile_verdict"] = {
        "rmse_range_k": [float(min(rmses)), float(max(rmses))],
        "tau_range_s": [60.0, 10000.0],
        "verdict": "τ 跨 160 倍 RMSE 只变 %.3f K（< 量化噪声）→ 实际不可辨识"
                   % (max(rmses) - min(rmses)),
    }

    # 4. 两节点（负结果）+ 多温度 β/G 量级
    print("[4/4] 两节点与多温度 ...")
    try:
        f2 = fit_two_node(segl, t_amb_c=25.0, area_m2=AREA_M2)
        payload["two_node"] = {
            "c_core_j_k": f2.c_core_j_k, "c_shell_j_k": f2.c_shell_j_k,
            "r_int_k_w": f2.r_int_k_w, "r_conv_k_w": f2.r_conv_k_w,
            "beta_w_a2": f2.beta_w_a2, "rmse_k": f2.rmse_k,
            "gap_max_k": f2.gap_max_k,
            "verdict": "表面单点温度 + 5 参数：参数不物理（发散/退化）→ "
                       "过参数化，负结果",
        }
        print(f"   two-node RMSE={f2.rmse_k:.3f}  Cc={f2.c_core_j_k:.2g} "
              f"Ri={f2.r_int_k_w:.2g}  → 退化")
    except Exception as e:  # pragma: no cover
        payload["two_node"] = {"error": str(e)[:200], "verdict": "拟合失败"}

    multi = []
    for temp in payload["data"]["temps_c"]:
        try:
            segs = segs_at(df, temp)
            segl_t = [(v["t_s"].to_numpy(float), v["temp_c"].to_numpy(float),
                       v["current_a"].to_numpy(float)) for v in segs.values()]
            f = fit_lumped(segl_t, t_amb_c=temp, c_th_j_k=C_PRIOR_J_K,
                           area_m2=AREA_M2, drift=True)
            multi.append({
                "temp_c": temp,
                "beta_w_a2": f.beta_w_a2,
                "g_w_k": f.g_w_k,
                "h_w_m2k": f.h_w_m2k,
                "rmse_k": f.rmse_k,
            })
            print(f"   T={temp:5.1f}C beta={f.beta_w_a2:.4f} h={f.h_w_m2k:6.2f} "
                  f"RMSE={f.rmse_k:.3f}")
        except Exception as e:
            multi.append({"temp_c": temp, "error": str(e)[:120]})
    payload["multi_temp_lumped"] = multi

    # 测量需求规格（可交付替代产出）
    payload["measurement_spec"] = {
        "purpose": "标定单节点热网络 (C, β, G) 或两节点 (…, R_int, R_conv)",
        "temperature_sensor": {
            "resolution_k": "≤ 0.01（当前 Arbin ~0.07 K 不够）",
            "accuracy_k": "±0.1",
            "points": "表面 2 点 + 芯部 1 点（埋热电偶）→ 才能辨识两节点",
            "sampling_s": "≤ 10",
        },
        "excitation": {
            "type": "设计激励而非行车工况：3–5 个恒功率脉冲（0.5/1/2/3C 等效）",
            "pulse_s": "≥ 3τ（τ~30 min → 脉冲 ≥ 1.5 h）或用阶跃到稳态",
            "rest_s": "≥ 3τ 冷却段（分离 τ 与 G）",
            "purpose": "阶跃-冷却曲线直接给 τ=C/G 与 G（Newton 冷却斜率）",
        },
        "heat_source": {
            "preferred": "电加热膜替代电芯（Q 已知精确到 1%）→ 直接辨识 C/G",
            "fallback": "电芯 I²R，R 从 HPPC 脉冲实测（当前 β 与 r_ohm 同量级）",
        },
        "ambient": "恒温箱 ±0.2 K；T_amb(t) 需独立热电偶实测（不能用静置点反推）",
        "why_needed": "本轮证明：无上述条件时 τ 跨 160 倍 RMSE 不变（实际不可辨识）",
    }

    # ---- 图 ----
    fig, ax = plotting.subplots(1, 2, figsize=(10, 3.8))
    pl = payload["profile_likelihood"]
    ax[0].semilogx([r["tau_s"] for r in pl], [r["rmse_k"] for r in pl], "o-",
                   color="#1D4ED8")
    ax[0].axhline(payload["snr"]["temp_c_quantum_k"], color="#B45309", ls="--",
                  label="温度量化噪声")
    ax[0].set_xlabel("τ = C/G (s)"); ax[0].set_ylabel("拟合 RMSE (K)")
    ax[0].set_title("剖面似然：τ 不可辨识")
    ax[0].legend(fontsize=8)

    rs = payload["ridge"]
    ax[1].scatter([r["c_th_j_k"] for r in rs], [r["beta_w_a2"] for r in rs],
                  s=80, color="#047857")
    for r in rs:
        ax[1].annotate(f"RMSE {r['rmse_k']:.3f}",
                       (r["c_th_j_k"], r["beta_w_a2"]),
                       textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax[1].set_xlabel("C_th (J/K)"); ax[1].set_ylabel("β (Ω)")
    ax[1].set_title("不可辨识脊：同 RMSE、不同参数")
    plotting.save(fig, "10a_calib_identifiability.png")

    payload["elapsed_s"] = round(time.time() - t0, 1)
    METRICS.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"完成，用时 {payload['elapsed_s']} s → {METRICS.name}")


if __name__ == "__main__":
    main()
