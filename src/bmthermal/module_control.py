"""模组级热管理控制：单点传感的布点代价（scripts/24 用）。

问题
----
BMS 在模组上通常只有 1–2 个 NTC。scripts/21 的芯温估计解决的是"表面
遮芯部"（时间维度的遮蔽）；模组级的第二个遮蔽是**空间维度**：冷却液沿程
升温 + 邻接传热 → 上游探针读数低于最热芯。本模块量化"NTC 放哪里"：

1. **上游探针**（冷端）——不补偿：漏报下游过温；
2. **下游探针**（热端）——天然最坏点：保守但过冷（能耗代价）；
3. **单点 + 梯度补偿**——探针在上游，按内部稳态模型加偏移估计最热芯；
4. **逐芯理想**（每芯 NTC）——上界对照。

模型
----
沿用 module.py 的准稳态冷板 + 邻接热导网络，扩展两处：

- **时变 h**（控制档位）：显式 RK4 定步推进（dt=1 s；最热模态
  λ≈h·A_eff/C≈0.07 1/s，显式稳定裕度充足）；
- **逐芯产热散布** ``q_spread``（制造离散，N(1, 5%) 口径）。

梯度补偿的内部模型：给定当前 q 与 h，冷板网络稳态解（定点迭代冷却液
耦合）给出"探针 ↔ 最热芯"偏移——工程上就是 BMS 里的查表/线性化模型。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .actuator import AFFINITY, PUMP_TYPICAL, energy_norm_from_trace
from .module import CP_COOLANT, ModuleSpec


@dataclass
class ModuleControlResult:
    t_s: np.ndarray
    h_cmd: np.ndarray
    temps_c: np.ndarray            # (n_cells, n_t)
    probe_idx: int
    estimator: str
    signal_c: np.ndarray           # 控制律看到的信号（估计最热芯温度）
    mode_trace: np.ndarray
    energy_norm_affinity: float
    energy_norm_pump: float
    duty_high: float
    n_switch: int
    t_max_c: float
    delta_t_max_k: float
    damage_max_s: float
    time_over_limit_s: float
    q_spread: np.ndarray
    meta: dict = field(default_factory=dict)


def _neighbors(spec: ModuleSpec) -> list[list[tuple[int, float]]]:
    pos = {(r, c): r * spec.n_cols + c
           for r in range(spec.n_rows) for c in range(spec.n_cols)}
    out: list[list[tuple[int, float]]] = [[] for _ in range(spec.n_cells)]
    for (r, c), i in pos.items():
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            j = pos.get((r + dr, c + dc))
            if j is not None:
                out[i].append((j, spec.g_neighbor_w_k))
    return out


def _coolant_temps(spec: ModuleSpec, t_cells: np.ndarray, hA: float, mcp: float) -> np.ndarray:
    t_cols = np.empty(spec.n_cols)
    t_cool = spec.t_in_c
    for c in range(spec.n_cols):
        t_cols[c] = t_cool
        heat = 0.0
        for r in range(spec.n_rows):
            i = r * spec.n_cols + c
            heat += (t_cells[i] - t_cool) * hA
        if mcp > 0:
            t_cool = t_cool + heat / mcp
    out = np.empty(spec.n_cells)
    for r in range(spec.n_rows):
        for c in range(spec.n_cols):
            out[r * spec.n_cols + c] = t_cols[c]
    return out


def steady_state_cells(spec: ModuleSpec, q_w: np.ndarray, h: float,
                       max_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """内部模型稳态解（梯度补偿用）：定点迭代冷却液耦合。"""
    c_th = spec.cell.thermal_capacity_j_k + spec.pcm_extra_capacity_j_k  # noqa: F841
    hA = h * spec.a_eff_m2
    mcp = spec.m_dot_kg_s * CP_COOLANT
    nb = _neighbors(spec)
    t = np.full(spec.n_cells, spec.t_in_c)
    for _ in range(max_iter):
        t_cool = _coolant_temps(spec, t, hA, mcp)
        t_new = np.empty_like(t)
        for i in range(spec.n_cells):
            gsum = sum(g for _, g in nb[i])
            rhs = q_w[i] + hA * t_cool[i] + sum(g * t[j] for j, g in nb[i])
            t_new[i] = rhs / (hA + gsum)
        if np.max(np.abs(t_new - t)) < tol:
            return t_new
        t = t_new
    return t


def simulate_module_control(
    t_s: np.ndarray,
    q_w: np.ndarray,
    spec: ModuleSpec,
    law,
    h_lo: float,
    h_hi: float,
    probe_idx: int = 0,
    estimator: str = "probe",     # probe | probe_comp | worst
    q_spread: np.ndarray | None = None,
    noise_k: float = 0.05,
    min_dwell_s: float = 20.0,
    initial_mode: int = 0,
    seed: int = 0,
    dt_max: float = 1.0,
) -> ModuleControlResult:
    """模组闭环：植株（逐芯网络）+ 单点传感 + 控制律。

    ``estimator``：
    - ``probe``：直接用探针读数（布点即策略，probe_idx 选上游/下游）；
    - ``probe_comp``：探针 + 内部稳态模型梯度偏移（估计最热芯）；
    - ``worst``：真值最热芯（逐芯 NTC 上界对照）。
    """
    t = np.asarray(t_s, float)
    q = np.asarray(q_w, float)
    if q.size == 1:
        q = np.full_like(t, float(q))
    n_t = t.size
    n = spec.n_cells
    rng = np.random.default_rng(seed)
    if q_spread is None:
        q_spread = np.ones(n)
    q_spread = np.asarray(q_spread, float)
    if q_spread.size != n:
        raise ValueError("q_spread 长度必须 = 电芯数")

    c_th = spec.cell.thermal_capacity_j_k + spec.pcm_extra_capacity_j_k
    nb = _neighbors(spec)
    mode = int(initial_mode)
    last_switch_t = float(t[0])
    x = np.full(n, spec.t_in_c)
    temps = np.empty((n, n_t))
    temps[:, 0] = x
    h_trace = np.empty(n_t)
    h_trace[0] = h_hi if mode else h_lo
    mode_trace = np.empty(n_t)
    mode_trace[0] = mode
    sig_trace = np.empty(n_t)
    sig_trace[0] = spec.t_in_c
    n_switch = 0

    def rhs(tt: np.ndarray, x_cur: np.ndarray, h: float, q_now: np.ndarray) -> np.ndarray:
        mcp = spec.m_dot_kg_s * CP_COOLANT
        hA = h * spec.a_eff_m2
        t_cool = _coolant_temps(spec, x_cur, hA, mcp)
        d = np.empty(n)
        for i in range(n):
            cond = sum(g * (x_cur[i] - x_cur[j]) for j, g in nb[i])
            d[i] = (q_now[i] - (x_cur[i] - t_cool[i]) * hA - cond) / c_th
        return d

    k = 0
    t_cur = float(t[0])
    while k < n_t - 1:
        # 子步推进到下一个控制采样点（t_cur 是独立游标，防止重复积分）
        t_end = min(t[k + 1], t_cur + dt_max)
        dt = t_end - t_cur
        if dt <= 0:
            raise ValueError("时间轴必须严格递增")
        h = h_hi if mode else h_lo
        q_now = q[k] * q_spread
        # RK4
        k1 = rhs(t_cur, x, h, q_now)
        k2 = rhs(t_cur + dt / 2, x + dt / 2 * k1, h, q_now)
        k3 = rhs(t_cur + dt / 2, x + dt / 2 * k2, h, q_now)
        k4 = rhs(t_cur + dt, x + dt * k3, h, q_now)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t_cur = t_end

        if t_cur >= t[k + 1] - 1e-9:
            k += 1
            temps[:, k] = x
            h_trace[k] = h
            # 测量与估计
            z = float(x[probe_idx]) + float(rng.normal(0.0, noise_k))
            if estimator == "probe":
                sig = z
            elif estimator == "probe_comp":
                ss = steady_state_cells(spec, q[k] * q_spread, h)
                sig = z + float(ss.max() - ss[probe_idx])
            elif estimator == "worst":
                sig = float(x.max())
            else:
                raise ValueError(f"未知 estimator: {estimator}")
            sig_trace[k] = sig
            new_mode = int(law(float(t[k]), sig, mode))
            if new_mode != mode and (t[k] - last_switch_t) >= min_dwell_s:
                mode = new_mode
                last_switch_t = float(t[k])
                n_switch += 1
            mode_trace[k] = mode

    # 能耗（双口径：理想相似定律 + 泵典型曲线）
    e_aff = energy_norm_from_trace(t, h_trace, h_lo, h_hi, AFFINITY)
    e_pump = energy_norm_from_trace(t, h_trace, h_lo, h_hi, PUMP_TYPICAL)

    from .aging import damage_integral
    dmg = [damage_integral(t, temps[i]) for i in range(n)]
    over = temps.max(axis=0) > 45.0
    dt_arr = np.diff(t, prepend=t[0])
    dt_arr[0] = 0.0
    dt_series = temps.max(axis=0) - temps.min(axis=0)

    return ModuleControlResult(
        t_s=t, h_cmd=h_trace, temps_c=temps,
        probe_idx=probe_idx, estimator=estimator,
        signal_c=sig_trace, mode_trace=mode_trace,
        energy_norm_affinity=e_aff, energy_norm_pump=e_pump,
        duty_high=float(np.mean(mode_trace)), n_switch=n_switch,
        t_max_c=float(temps.max()), delta_t_max_k=float(dt_series.max()),
        damage_max_s=float(np.max(dmg)),
        time_over_limit_s=float(dt_arr[over].sum()) if over.any() else 0.0,
        q_spread=q_spread,
        meta={
            "h_lo": h_lo, "h_hi": h_hi, "min_dwell_s": min_dwell_s,
            "noise_k": noise_k, "limit_c": 45.0,
            "damage_cell_max_idx": int(np.argmax(dmg)),
        },
    )
