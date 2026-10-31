"""动态工况（US06）输入与仿真实现。

数据来源（真实测量，不是合成波形）
----------------------------------
`data/us06_crate.csv` 的电流剖面来自 **CALCE A123（LFP）25 °C 的 DST-US06-FUDS
动态放电段**（与 A4 项目同一数据源），取其 US06 段的前 600 s（一个完整 US06 循环），
按 A123 标称容量 2.5 Ah 归一成 C 率，再乘到目标电芯的额定容量上。

波形统计（归一后 C 率）：均值 0.21C（放电为正）、峰值 1.46C、回馈（负）峰值 −0.26C。
US06 是 EPA 高速/高加速度工况，比 FUDS/DST 剧烈得多，对热管理是"动态冲击"的代表。

为什么要用动态工况
------------------
恒流扫参回答"稳态能力够不够"，但真实工况的产热是脉冲式的：
峰值时刻产热 ∝ I²，均值只有 0.21C 而峰值 1.46C —— **峰值产热是均值的 48 倍**。
热容决定电芯"扛不扛得住"这几秒到几十秒的脉冲，h 决定"间歇期能不能把热散掉"。
两者的分工只有在动态工况里才看得清，恒流工况里会混为一谈。

两条使用路径
------------
1. **DFN 电化学-热耦合**（`run_dfn_duty`）：把 C 率曲线喂给
   ``pybamm.step.current``，得到真实的 Q(t) 与温度反馈。上限仍是 ≤2C
   （Chen2020 的可信域，见 heat.py）；US06 峰值 1.46C 在域内。
2. **两节点快模型**（`run_network_duty`）：Q(t) 先由 DFN 等温算出（或直接
   按 Q∝C² 缩放），再喂热网络。用于方案矩阵的动态工况一列。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .network import TwoNodeCell, build_cell, simulate_two_node

#: US06 剖面 CSV（time_s, c_rate_discharge；放电为正，回馈为负）
US06_CSV = Path(__file__).resolve().parents[2] / "data" / "us06_crate.csv"
#: 数据出处（写进报告的引用串）
US06_PROVENANCE = (
    "CALCE A123 (LFP) 25 °C DST-US06-FUDS 动态放电段，US06 前 600 s，"
    "按 2.5 Ah 标称容量归一为 C 率（与 A4 项目同源）"
)


def load_us06(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """读 US06 倍率剖面。``scale`` 用于整体加减载（稳健性检查）。

    返回 ``(t_s, c_rate)``，放电为正。
    """
    import pandas as pd

    df = pd.read_csv(US06_CSV)
    t = df["time_s"].to_numpy(dtype=float)
    c = df["c_rate_discharge"].to_numpy(dtype=float) * float(scale)
    return t, c


def crate_to_current(c_rate: np.ndarray, capacity_ah: float = 5.0) -> np.ndarray:
    """C 率 → 电流 A。**放电为正**的约定在这里换符号：
    PyBaMM 的电流约定是**放电为负**，所以喂给 step.current 时要取负号。
    """
    return -np.asarray(c_rate, dtype=float) * float(capacity_ah)


def make_current_callable(t_s: np.ndarray, i_a: np.ndarray):
    """构造 PyBaMM ``step.current`` 需要的 ``f(t) -> I``（放电为负）。"""
    t_s = np.asarray(t_s, dtype=float)
    i_a = np.asarray(i_a, dtype=float)

    def f(t):
        return float(np.interp(t, t_s, i_a, left=i_a[0], right=i_a[-1]))

    return f


def run_dfn_duty(
    scale: float = 1.0,
    t_amb_c: float = 25.0,
    h: float | None = None,
    isothermal: bool = True,
    parameter_set: str = "Chen2020",
    soc0: float = 1.0,
    capacity_ah: float = 5.0,
    duration_s: float | None = None,
    model: str = "DFN",
) -> dict:
    """跑一段 US06 动态工况（DFN），返回时间序列与摘要。

    ``duration_s`` 默认就是剖面全长（600 s）；可传更长以观察"重复冲击"，
    此时剖面按周期拼接。
    """
    import warnings

    import pybamm

    from .params import load_parameter_values

    t_p, c_p = load_us06(scale=scale)
    if duration_s is None:
        duration_s = float(t_p[-1])

    # 两列数组 (t, I) 交给 pybamm 的 drive-cycle 接口：
    # 它会自动包成 Interpolant，且 duration 超过剖面长度时自动周期延拓，
    # 比手写 callable 干净（callable 会在符号 t 上炸掉）。
    i_a = crate_to_current(c_p, capacity_ah)  # 放电为负
    drive = np.column_stack([t_p, i_a])

    if isothermal:
        opts = {"thermal": "isothermal", "calculate heat source for isothermal models": "true"}
    else:
        opts = {"thermal": "lumped"}
    pv = load_parameter_values(parameter_set, t_amb_c=t_amb_c, soc0=soc0, h=h)
    mdl = getattr(pybamm.lithium_ion, model)(options=opts)
    exp = pybamm.Experiment([pybamm.step.current(drive, duration=duration_s)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = pybamm.Simulation(mdl, parameter_values=pv, experiment=exp).solve()

    t = np.asarray(sol["Time [s]"].entries, dtype=float)
    if t.size == 0:
        raise RuntimeError("动态工况返回空解")

    def g(name, default=0.0):
        try:
            return np.asarray(sol[name].entries, dtype=float).ravel()
        except Exception:
            return np.full(t.shape, default)

    cap = g("Discharge capacity [A.h]")
    q = g("Total heating [W]")
    temp = g("Volume-averaged cell temperature [C]")
    v = g("Voltage [V]")
    # 周期延拓的倍率剖面（用于对照/画图）
    period = float(t_p[-1] + (t_p[1] - t_p[0]))
    c_series = np.interp(np.mod(t, period), t_p, c_p, left=c_p[0], right=c_p[-1])
    return {
        "t_s": t,
        "c_rate": c_series,
        "q_total_w": q,
        "t_cell_c": temp,
        "voltage_v": v,
        "soc": np.clip(1.0 - cap / capacity_ah, 0.0, 1.0),
        "q_mean_w": float(np.mean(q)),
        "q_peak_w": float(np.max(q)),
        "t_max_c": float(np.max(temp)),
        "delivered_ah": float(cap[-1]) if cap.size else 0.0,
        "source": "US06 (CALCE A123 25C, normalized)",
        "scale": scale,
    }


def run_network_duty(
    t_s: np.ndarray,
    q_w: np.ndarray,
    t_amb_c: float,
    cell: TwoNodeCell,
    t0_c: float | None = None,
) -> dict:
    """把动态 Q(t) 喂给两节点网络（复用 network 的求解器）。"""
    return simulate_two_node(t_s, q_w, t_amb_c, cell, t0_c=t0_c)


def duty_matrix(
    scheme_keys=("natural", "forced_air", "cold_plate", "immersion", "pcm"),
    ambients=(25.0, 35.0, 45.0),
    q_from: dict | None = None,
    cache: dict | None = None,
) -> list[dict]:
    """动态工况 × 冷却方案 × 环境温度 的选型矩阵行。

    ``q_from``：可选，``{scale: (t, q)}`` 的外部产热缓存；默认用
    ``q_duty_mean`` 的 Q∝C² 恒流等效（保守）或由调用方先跑
    ``run_dfn_duty`` 再传入。
    """
    from .params import SCHEMES

    t_p, c_p = load_us06()
    rows = []
    for t_amb in ambients:
        for key in scheme_keys:
            scheme = SCHEMES[key]
            net = build_cell(
                h=scheme.h_mid,
                pcm_extra_capacity_j_k=scheme.pcm_extra_capacity_j_k,
            )
            if q_from is not None:
                t_q, q_arr = q_from
            else:
                raise ValueError("需要提供 (t, q) 产热序列：先用 run_dfn_duty 取得")
            out = run_network_duty(t_q, q_arr, t_amb, net)
            tc, ts = out["t_core_c"], out["t_surface_c"]
            rows.append({
                "duty": "US06",
                "scheme": key,
                "scheme_name": scheme.name,
                "h": scheme.h_mid,
                "t_amb_c": t_amb,
                "t_max_c": float(max(tc.max(), ts.max())),
                "t_rise_k": float(max(tc.max(), ts.max()) - t_amb),
                "core_surface_gap_k": float(tc.max() - ts.max()),
                "pass_45c": bool(max(tc.max(), ts.max()) <= 45.0),
            })
    return rows
