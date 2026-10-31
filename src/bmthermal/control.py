"""BMS 热管理控制闭环：观测器 → 滞环控制 → 冷却档位。

环路
----
    真值植株（两节点，h 随控制切换）
      → 表面温度测量 z（+噪声）
      → Kalman 芯温估计 x_hat（模型知道当前命令的 h）
      → 滞环控制器（按 T_surface 或 T_core_est 门限切换 h_lo/h_hi）
      → 冷却档位回植株

对比三种策略（scripts/21）：
1. 恒满冷（h_hi 常开）——能耗上限基准；
2. 表面温度触发——§16 的掩蔽风险路径（表面滞后芯部）；
3. **芯温估计触发**——观测器价值的直接兑现。

能耗口径
--------
风机/泵功率按相似定律 P ∝ h³（转速-压头-功率）；输出归一化能量
（以恒满冷 = 1.0 计），不冒充绝对瓦时。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .calib import _zoh
from .network import TwoNodeCell
from .observer import run_kalman


@dataclass
class ControlResult:
    t_s: np.ndarray
    h_cmd: np.ndarray               # 每步冷却档位 h(t)
    t_core_true_c: np.ndarray
    t_surface_true_c: np.ndarray
    t_core_hat_c: np.ndarray        # 观测器估计（控制闭环用）
    mode_trace: np.ndarray          # 0=低档 1=高档
    energy_norm: float              # 归一化冷却能耗（恒满冷=1）
    duty_high: float                # 高档时间占比
    n_switch: int
    t_max_core_c: float
    damage_s: float                 # 芯温损伤积分（s，25 °C 等效）
    core_err_rmse_k: float
    meta: dict = field(default_factory=dict)


def bang_bang(t_on_c: float, t_off_c: float, use_estimate: bool = True):
    """滞环控制律工厂：高于 t_on 开满冷，低于 t_off 关到低档。

    ``use_estimate``：True=按芯温估计触发（BMS 用观测器输出），
    False=按表面实测触发。
    """
    if t_off_c > t_on_c:
        raise ValueError("滞环要求 t_off < t_on")

    def law(t: float, z_surface_c: float, x_hat: np.ndarray, mode: int) -> int:
        sig = float(x_hat[0]) if use_estimate else float(z_surface_c)
        if sig >= t_on_c:
            return 1
        if sig <= t_off_c:
            return 0
        return mode   # 滞环保持

    return law


def always_on(mode: int = 1):
    def law(t, z_surface_c, x_hat, cur):
        return mode
    return law


def simulate_control_loop(
    t_s: np.ndarray,
    q_w: np.ndarray,
    t_amb_c: float,
    cell_true: TwoNodeCell,
    cell_model: TwoNodeCell,
    h_lo: float,
    h_hi: float,
    law: Callable[[float, float, np.ndarray, int], int],
    noise_k: float = 0.05,
    seed: int = 0,
    q_process_w2: float = 0.05**2,
    r_meas_k2: float | None = None,
    initial_mode: int = 0,
    q_true_scale: float = 1.0,
    min_dwell_s: float = 0.0,
) -> ControlResult:
    """闭环仿真（逐步 ZOH）：植株 + Kalman + 控制律。

    植株用 ``cell_true``（可与模型失配）；观测器与控制用 ``cell_model``
    及当前命令 h。``q_true_scale`` 只作用于植株侧产热（模型/观测器仍用
    名义 q）——BMS 的产热估计失配场景。``min_dwell_s`` 是执行器最短
    保持时间（泵/阀不能按采样率切换）：距上次切换不足该时长时保持原
    档位。表面节点热容小（~9 J/K）+ 强档位时表面触发会以采样率抖振，
    这是 BMS 实现必须加最短保持的原因。返回 ControlResult。
    """
    from .observer import bias_matrices  # noqa: F401  (保持依赖显式)
    from .observer import two_node_matrices

    t = np.asarray(t_s, float)
    q = np.asarray(q_w, float)
    if q.size == 1:
        q = np.full_like(t, float(q))
    n = t.size
    if n < 2:
        raise ValueError("时间轴太短")
    r_meas_k2 = noise_k**2 if r_meas_k2 is None else r_meas_k2
    rng = np.random.default_rng(seed)

    # 每个档位的离散矩阵缓存
    def matrices(h: float):
        cell = cell_true.with_h(h)
        return two_node_matrices(cell)

    cache: dict[float, tuple] = {}

    def zoh_of(h: float, dt: float):
        key = (h, dt)
        if key not in cache:
            A, B, _ = matrices(h)
            cache[key] = _zoh(A, B, dt)
        return cache[key]

    # 初始状态
    x_true = np.array([t_amb_c, t_amb_c], dtype=float)
    x_hat = np.array([t_amb_c, t_amb_c], dtype=float)
    P = np.eye(2) * 1.0
    _, _, H = two_node_matrices(cell_model)
    m_cache: dict[tuple[float, float], tuple] = {}

    mode = int(initial_mode)
    last_switch_t = float(t[0])
    xh = np.empty((2, n))
    xh[:, 0] = x_hat
    xt = np.empty((2, n))
    xt[:, 0] = x_true
    h_trace = np.empty(n)
    h_trace[0] = h_hi if mode else h_lo
    mode_trace = np.empty(n)
    mode_trace[0] = mode
    z_trace = np.empty(n)
    z_trace[0] = t_amb_c
    n_switch = 0

    for k in range(n - 1):
        dt = float(t[k + 1] - t[k])
        if dt <= 0:
            raise ValueError("时间轴必须严格递增")
        h_cmd = h_hi if mode else h_lo
        h_trace[k + 1] = h_cmd

        # 植株传播（真值；产热可失配）
        Ad, Bd = zoh_of(h_cmd, dt)
        u = np.array([q[k], t_amb_c], dtype=float)
        u_true = np.array([q[k] * q_true_scale, t_amb_c], dtype=float)
        x_true = Ad @ x_true + Bd @ u_true
        z = float(x_true[1]) + float(rng.normal(0.0, noise_k))
        z_trace[k + 1] = z

        # 观测器：模型用同一命令 h（BMS 知道自己输出什么）
        if (h_cmd, dt) not in m_cache:
            Am, Bm, _ = two_node_matrices(cell_model.with_h(h_cmd))
            m_cache[(h_cmd, dt)] = _zoh(Am, Bm, dt)
        Adm, Bdm = m_cache[(h_cmd, dt)]
        x_hat = Adm @ x_hat + Bdm @ u
        P = Adm @ P @ Adm.T + np.eye(2) * (q_process_w2 * dt)
        S = (H @ P @ H.T).item() + r_meas_k2
        K = (P @ H.T) / S
        y = z - (H @ x_hat).item()
        x_hat = x_hat + K.ravel() * y
        P = (np.eye(2) - K @ H) @ P

        # 控制律（最短保持时间内保持原档位）
        new_mode = int(law(float(t[k + 1]), z, x_hat, mode))
        if new_mode != mode and (t[k + 1] - last_switch_t) >= min_dwell_s:
            n_switch += 1
            mode = new_mode
            last_switch_t = float(t[k + 1])
        mode_trace[k + 1] = mode
        xh[:, k + 1] = x_hat
        xt[:, k + 1] = x_true

    # 能耗：P ∝ h³，归一化到恒满冷
    p_rel = (h_trace / h_hi) ** 3
    p_full = np.ones_like(t)
    dt_arr = np.diff(t, prepend=t[0])
    dt_arr[0] = 0.0
    energy_norm = float(np.sum(p_rel * dt_arr) / np.sum(p_full * dt_arr))

    # 损伤积分（芯温真值）
    from .aging import damage_integral
    damage = damage_integral(t, xt[0])

    err = xh[0] - xt[0]
    return ControlResult(
        t_s=t,
        h_cmd=h_trace,
        t_core_true_c=xt[0],
        t_surface_true_c=xt[1],
        t_core_hat_c=xh[0],
        mode_trace=mode_trace,
        energy_norm=energy_norm,
        duty_high=float(np.mean(mode_trace)),
        n_switch=n_switch,
        t_max_core_c=float(xt[0].max()),
        damage_s=damage,
        core_err_rmse_k=float(np.sqrt(np.mean(err**2))),
        meta={
            "h_lo": h_lo, "h_hi": h_hi,
            "noise_k": noise_k,
            "min_dwell_s": min_dwell_s,
            "q_true_scale": q_true_scale,
            "energy_model": "P ∝ h³（相似定律），恒满冷=1.0",
        },
    )
