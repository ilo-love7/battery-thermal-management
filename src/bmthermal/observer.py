"""BMS 芯温估计器：两节点热网络 + Kalman 观测器（只用表面温度）。

问题
----
车端只有表面热电偶（v1.1 §16 的有效性结论：gap 只随倍率，5C 达 15.9 K，
瞬态可到 34.6 K——表面读数不能直接做安全判据）。芯温要么埋传感器
（成本/可靠性），要么**估**。本模块把 v1.1 §16 的"BMS 必须输出芯温估计"
变成一个可运行的观测器，并给出它的适用域（参数失配多大时芯温误差还可信）。

模型
----
两节点（network.py 同款）是线性定常系统::

    x = [T_core, T_surface],  u = [Q, T_amb]
    C·dx/dt = A x + B u,     z = H x = T_surface

用 ZOH 精确离散（calib._zoh）后跑离散 Kalman：

    预测: x̂⁻ = Ad x̂ + Bd u,   P⁻ = Ad P Adᵀ + Q_w
    更新: K = P⁻Hᵀ(HP⁻Hᵀ+R_w)⁻¹,  x̂ = x̂⁻ + K(z−Hx̂⁻),  P = (I−KH)P⁻

输入 Q(t) 来自电流平方的模型（β·I²，β 可吸收内阻）或直接给产热曲线。
真实 BMS 里 Q 用 I²R 估计——估计器对 Q 的失配鲁棒性因此是必测项
（script 14 扫 q_scale 失配）。

为什么不直接用单节点
--------------------
单节点只有 T_core=T_surface 一个状态，结构上无法表达 gap；
而 gap 恰恰是高倍率下寿命/安全判断的关键（v1.1 §13 两节点负结果说的是
**从单点表面数据辨识不出两节点参数**，不是两节点没用——参数从
按层核算/解析解来（network.build_cell），估计器只补"当前状态"，
不补"模型参数"。这个分工是刻意的）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .calib import _zoh
from .network import TwoNodeCell


def two_node_matrices(cell: TwoNodeCell) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """连续时间状态空间 (A, B, H)，x=[T_core,T_surface], u=[Q, T_amb]。"""
    c_core = cell.c_core
    c_shell = cell.c_shell
    r_int = cell.r_int_k_w
    r_conv = cell.r_conv_k_w
    a11 = -1.0 / (r_int * c_core)
    a12 = 1.0 / (r_int * c_core)
    a21 = 1.0 / (r_int * c_shell)
    a22 = -(1.0 / (r_int * c_shell) + 1.0 / (r_conv * c_shell))
    A = np.array([[a11, a12], [a21, a22]], dtype=float)
    B = np.array(
        [[1.0 / c_core, 0.0], [0.0, 1.0 / (r_conv * c_shell)]], dtype=float
    )
    H = np.array([[0.0, 1.0]], dtype=float)  # 只测表面
    return A, B, H


def simulate_state_space(
    cell: TwoNodeCell,
    t_s: np.ndarray,
    q_w: np.ndarray,
    t_amb_c: float,
    t0_c: float | None = None,
) -> np.ndarray:
    """ZOH 精确积分两节点，返回状态轨迹 shape (2, n)。

    与 ``network.simulate_two_node`` 同一物理（测试里对拍），
    但走矩阵指数路径，供 Kalman 与 Monte Carlo 复用（快）。
    """
    t = np.asarray(t_s, float)
    q = np.asarray(q_w, float)
    if q.size == 1:
        q = np.full_like(t, float(q))
    A, B, _ = two_node_matrices(cell)
    t0 = float(t_amb_c if t0_c is None else t0_c)
    x = np.array([t0, t0], dtype=float)
    n = t.size
    xs = np.empty((2, n), dtype=float)
    xs[:, 0] = x
    cache: dict[float, tuple] = {}
    for k in range(n - 1):
        dt = float(t[k + 1] - t[k])
        if dt <= 0:
            raise ValueError("时间轴必须严格递增")
        if dt not in cache:
            cache[dt] = _zoh(A, B, dt)
        Ad, Bd = cache[dt]
        u = np.array([q[k], t_amb_c], dtype=float)
        x = Ad @ x + Bd @ u
        xs[:, k + 1] = x
    return xs


@dataclass
class KalmanResult:
    t_s: np.ndarray
    t_core_hat_c: np.ndarray
    t_surface_hat_c: np.ndarray
    t_core_truth_c: np.ndarray | None
    innovation_k: np.ndarray
    gain_trace: np.ndarray
    core_err_rmse_k: float
    core_err_max_k: float
    within_3k_share: float
    diverged: bool

    def summary(self) -> dict:
        return {
            "core_err_rmse_k": self.core_err_rmse_k,
            "core_err_max_k": self.core_err_max_k,
            "within_3k_share": self.within_3k_share,
            "diverged": self.diverged,
            "innovation_rms_k": float(np.sqrt(np.mean(self.innovation_k**2))),
        }


def run_kalman(
    t_s: np.ndarray,
    q_w: np.ndarray,
    z_surface_c: np.ndarray,
    t_amb_c: float,
    model_cell: TwoNodeCell,
    q_process_w2: float = 0.05**2,
    r_meas_k2: float = 0.05**2,
    t0_c: float | None = None,
    truth_core_c: np.ndarray | None = None,
    x0_c: float | None = None,
) -> KalmanResult:
    """离散 Kalman 估计芯温。

    参数
    ----
    q_process_w2: 过程噪声协方差（对状态的等效扰动，按 (K/s)² 口径调）。
        实现里对离散 A 相当于每步注入 ``Q_w = q_process_w2 * dt * I``。
    r_meas_k2: 表面测量噪声方差（K²）。
    """
    t = np.asarray(t_s, float)
    q = np.asarray(q_w, float)
    if q.size == 1:
        q = np.full_like(t, float(q))
    z = np.asarray(z_surface_c, float)
    A, B, H = two_node_matrices(model_cell)

    x0 = float(t_amb_c if x0_c is None else x0_c)
    x = np.array([x0, x0], dtype=float)
    P = np.eye(2) * 1.0  # 初始不确定度 ~1 K²
    n = t.size
    xh = np.empty((2, n))
    xh[:, 0] = x
    innov = np.zeros(n)
    gtrace = np.zeros(n)
    cache: dict[float, tuple] = {}
    for k in range(n - 1):
        dt = float(t[k + 1] - t[k])
        if dt <= 0:
            raise ValueError("时间轴必须严格递增")
        if dt not in cache:
            cache[dt] = _zoh(A, B, dt)
        Ad, Bd = cache[dt]
        u = np.array([q[k], t_amb_c], dtype=float)
        # 预测
        x = Ad @ x + Bd @ u
        P = Ad @ P @ Ad.T + np.eye(2) * (q_process_w2 * dt)
        # 更新（测表面）
        S = (H @ P @ H.T).item() + r_meas_k2
        K = (P @ H.T) / S
        y = z[k + 1] - (H @ x).item()
        x = x + K.ravel() * y
        P = (np.eye(2) - K @ H) @ P
        xh[:, k + 1] = x
        innov[k + 1] = y
        gtrace[k + 1] = float(K[0, 0])  # 芯温通道增益

    core_hat = xh[0]
    surf_hat = xh[1]
    truth = None if truth_core_c is None else np.asarray(truth_core_c, float)
    if truth is not None:
        err = core_hat - truth
        rmse = float(np.sqrt(np.mean(err**2)))
        emax = float(np.max(np.abs(err)))
        within = float(np.mean(np.abs(err) < 3.0))
        diverged = bool(rmse > 5.0 or not np.all(np.isfinite(core_hat)))
    else:
        rmse = emax = within = np.nan
        diverged = bool(not np.all(np.isfinite(core_hat)))
    return KalmanResult(
        t_s=t,
        t_core_hat_c=core_hat,
        t_surface_hat_c=surf_hat,
        t_core_truth_c=truth,
        innovation_k=innov,
        gain_trace=gtrace,
        core_err_rmse_k=rmse,
        core_err_max_k=emax,
        within_3k_share=within,
        diverged=diverged,
    )
