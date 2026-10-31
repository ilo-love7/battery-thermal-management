"""00 · 环境与模型适用边界勘察。

这一步不是形式主义的"跑个 hello world"。它要回答四个会**决定后面所有数字**的问题：

1. 参数集能不能支撑到 5C？（博远的方案里写着 0.5C–5C 扫参）
2. SPM / SPMe / DFN 选哪个？（产热对极化分布敏感）
3. 电芯厚度方向的温差有多大？lumped 假设还成立吗？
4. 可逆热能不能算？（Chen2020 的熵变系数是 0）

结论写进 results/metrics/00_env.json，并在报告 §2 引用。
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pybamm  # noqa: E402

from bmthermal.params import CELL, load_parameter_values, volumetric_heat_capacity  # noqa: E402

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parents[1] / "results" / "metrics"
OUT.mkdir(parents=True, exist_ok=True)


def run(model: str, c_rate: float, v_min: float = 3.0, thermal: str = "lumped",
        opts_extra: dict | None = None, t_amb_c: float = 25.0):
    opts = {"thermal": thermal}
    if opts_extra:
        opts.update(opts_extra)
    pv = load_parameter_values("Chen2020", t_amb_c=t_amb_c, soc0=1.0)
    mdl = getattr(pybamm.lithium_ion, model)(options=opts)
    exp = pybamm.Experiment([f"Discharge at {c_rate:g}C until {v_min} V"])
    t0 = time.time()
    sol = pybamm.Simulation(mdl, parameter_values=pv, experiment=exp).solve()
    elapsed = time.time() - t0
    t = np.asarray(sol["Time [s]"].entries, dtype=float)
    if t.size == 0:
        return {"model": model, "c_rate": c_rate, "ok": False, "elapsed_s": elapsed}
    q = np.asarray(sol["Total heating [W]"].entries, dtype=float)
    temp = np.asarray(sol["Volume-averaged cell temperature [C]"].entries, dtype=float)
    return {
        "model": model, "c_rate": c_rate, "ok": True, "elapsed_s": round(elapsed, 2),
        "t_end_s": float(t[-1]),
        "delivered_ah": float(5.0 * c_rate * t[-1] / 3600.0),
        "q_mean_w": float(np.mean(q)),
        "t_max_c": float(np.max(temp)),
        "thermal": thermal,
    }


def thickness_gradient(c_rate: float = 2.0, h: float = 10.0) -> dict:
    """x-full（穿厚分辨）下的叠片方向温差。

    注意口径：PyBaMM 的 x 是**叠片厚度方向**（~100 µm 量级），
    不是 21700 的卷芯径向。卷芯径向梯度由两节点/解析解给出（R_int）。
    """
    import pybamm as _pb

    from bmthermal.params import apply_x_full_extras

    pv = load_parameter_values("Chen2020", t_amb_c=25.0, soc0=1.0, h=h)
    apply_x_full_extras(pv, h=h)
    mdl = _pb.lithium_ion.DFN(options={"thermal": "x-full"})
    exp = _pb.Experiment([f"Discharge at {c_rate:g}C until 3.0 V"])
    sol = _pb.Simulation(mdl, parameter_values=pv, experiment=exp).solve()
    t = np.asarray(sol["Time [s]"].entries, dtype=float)
    gaps = []
    for tt in t[:: max(1, t.size // 40)]:
        y = np.asarray(sol["Cell temperature [C]"](float(tt)), dtype=float).ravel()
        gaps.append(float(y.max() - y.min()))
    q = np.asarray(sol["Total heating [W]"].entries, dtype=float)
    return {
        "c_rate": c_rate, "h": h, "option": "x-full",
        "dT_stack_max_k": float(np.max(gaps)),
        "note": "x=叠片厚度方向（~100um），非卷芯径向；卷芯径向梯度见两节点模型",
        "q_mean_w": float(np.mean(q)),
    }


def main() -> None:
    t_start = time.time()
    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pybamm": pybamm.__version__,
        "cell_thermal_capacity_j_k": volumetric_heat_capacity("Chen2020") * CELL.volume_m3,
        "cell_volume_m3": CELL.volume_m3,
        "cell_area_m2": CELL.surface_area_m2,
        "r_int_k_w": CELL.conduction_resistance_core_to_surface(),
    }

    # 1. 倍率上限：SPM / SPMe / DFN 在 3C 的交付容量对照
    print("[1/4] 模型 × 倍率的交付容量对照 ...")
    models = []
    for m in ("SPM", "SPMe", "DFN"):
        for c in (1.0, 3.0):
            r = run(m, c)
            models.append(r)
            print(f"   {m:5s} {c}C  t_end={r.get('t_end_s', float('nan')):7.0f}s  "
                  f"Ah={r.get('delivered_ah', float('nan')):5.2f}")
    env["model_rate_matrix"] = models

    # 2. DFN 的倍率天花板
    print("[2/4] DFN 倍率天花板（4C/5C）...")
    ceiling = []
    for c in (4.0, 5.0):
        r = run("DFN", c)
        ceiling.append(r)
        print(f"   DFN {c}C  t_end={r.get('t_end_s', float('nan')):6.0f}s  "
              f"Ah={r.get('delivered_ah', float('nan')):5.2f}  Q={r.get('q_mean_w', float('nan')):5.2f}W")
    env["rate_ceiling"] = ceiling

    # 3. 穿厚（叠片）方向温差
    print("[3/4] 穿厚方向温度梯度（x-full）...")
    env["thickness_gradient"] = thickness_gradient(2.0, 10.0)
    print(f"   ΔT_stack = {env['thickness_gradient']['dT_stack_max_k']:.6f} K "
          f"（叠片方向；卷芯径向梯度另见两节点模型）")

    # 4. 熵变系数
    print("[4/4] 参数集熵变系数 ...")
    ent = {}
    for name in ("Chen2020", "Mohtat2020", "OKane2022"):
        pv = pybamm.ParameterValues(name)
        neg = pv["Negative electrode OCP entropic change [V.K-1]"]
        pos = pv["Positive electrode OCP entropic change [V.K-1]"]
        ent[name] = {
            "negative": "callable" if callable(neg) else float(neg),
            "positive": "callable" if callable(pos) else float(pos),
            "capacity_ah": float(pv["Nominal cell capacity [A.h]"]),
        }
        print(f"   {name:12s} cap={ent[name]['capacity_ah']:.2f}Ah  "
              f"dU/dT_neg={ent[name]['negative']}  dU/dT_pos={ent[name]['positive']}")
    env["entropic_change"] = ent

    env["total_elapsed_s"] = round(time.time() - t_start, 1)
    (OUT / "00_env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n写入 {OUT / '00_env.json'}（{env['total_elapsed_s']} s）")


if __name__ == "__main__":
    main()
