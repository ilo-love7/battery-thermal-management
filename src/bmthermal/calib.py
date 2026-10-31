"""热网络参数标定：从实测温升曲线辨识 R/C/h。

方法
----
热网络是线性定常系统，逐段常值输入下有精确离散解（矩阵指数），
**不数值积分 ODE**——拟合快且无积分误差。

- :func:`fit_lumped` 单节点：C·dT/dt = β·I² − G·(T−T_amb)
  未知量 (C, β, G)；β 吸收内阻（W/A² = Ω），G = h·A（W/K）。
- :func:`fit_two_node` 芯-壳两节点：5 参数 (C_core, C_shell, R_int, R_conv, β)。

三条口径：

1. **多段联合拟合**：同一电芯同一箱温下的 DST/US06/FUDS 三段共享参数，
   残差叠加——单段温升只有 ~2 K，各自拟合欠约束（参数会跑飞）。
2. **轨迹 NLLS**（对 T(t) 逐步残差）而不是 dT/dt 微分拟合——
   Arbin 温度有 ~0.07 K 量化，微分会放大噪声。
3. **≥5 s 降采样**：热时间常数是分钟级，1 s 采样对慢动力学是冗余的，
   降采样后拟合快 10 倍且不损失辨识信息。

参数辨识用 log 域参数化保正 + ``scipy.optimize.least_squares``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

MIN_DT_S = 5.0


def ambient_from_rest(
    t_s: np.ndarray,
    temp_c: np.ndarray,
    current_a: np.ndarray,
    rest_tol_a: float = 0.05,
    smooth_s: float = 600.0,
) -> np.ndarray:
    """从静置点（|I| < ``rest_tol_a``）估计时变环境温度 T_amb(t)。

    恒温箱的设定值不等于电芯实际感受到的环境（箱体本身漂 ~0.6 K/10 h，
    且有强制循环的空间不均匀）。静置时电芯温度收敛到当地环境温度，
    故取静置点做中值平滑后按时间插值到全时轴。
    """
    t = np.asarray(t_s, float)
    T = np.asarray(temp_c, float)
    I = np.asarray(current_a, float)
    rest = np.abs(I) < rest_tol_a
    if rest.sum() < 5:
        return np.full_like(t, float(np.median(T)))
    tr, Tr = t[rest], T[rest]
    # 时间窗中值平滑（抗量化噪声）
    out = np.empty_like(tr)
    half = smooth_s / 2.0
    for k, tk in enumerate(tr):
        m = np.abs(tr - tk) <= half
        out[k] = np.median(Tr[m])
    return np.interp(t, tr, out)


# ---------------------------------------------------------------------------
# 精确离散化（ZOH：逐采样段内输入视为常值）
# ---------------------------------------------------------------------------


def _zoh(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """x_{k+1} = Ad x_k + Bd u_k，对 LTI 系统精确离散。"""
    n, m = B.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


def _simulate_lti(
    A: np.ndarray,
    B: np.ndarray,
    C_out: np.ndarray,
    x0: np.ndarray,
    u: np.ndarray,
    dt: np.ndarray,
) -> np.ndarray:
    """逐步 ZOH 模拟，返回输出 y（一维）。

    ``u`` 形状 ``(m, n)``（m 个输入 × n 个采样），``B`` 为 ``(n_state, m)``；
    ``dt`` 为相邻采样间隔（长度 n-1）。
    """
    u = np.atleast_2d(np.asarray(u, float))   # (m, n)
    x = np.asarray(x0, float).ravel()
    n = u.shape[1]
    y = np.empty(n)
    y[0] = float(C_out @ x)
    cache: dict[float, tuple] = {}
    for k in range(n - 1):
        d = float(dt[k])
        if d not in cache:
            cache[d] = _zoh(A, B, d)
        Ad, Bd = cache[d]
        x = Ad @ x + Bd @ u[:, k]
        y[k + 1] = float(C_out @ x)
    return y


def _prepare(
    t_s: np.ndarray,
    temp_c: np.ndarray,
    current_a: np.ndarray,
    min_dt: float = MIN_DT_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """清洗时间轴（严格递增）并降到 ≥ ``min_dt`` 采样。返回 (t, T, I, dt)。"""
    t = np.asarray(t_s, float)
    T = np.asarray(temp_c, float)
    I = np.asarray(current_a, float)
    ok = np.diff(t) > 0
    t = np.r_[t[0], t[1:][ok]]
    T = np.r_[T[0], T[1:][ok]]
    I = np.r_[I[0], I[1:][ok]]
    keep = [0]
    acc = 0.0
    for k in range(1, t.size):
        acc += t[k] - t[k - 1]
        if acc >= min_dt:
            keep.append(k)
            acc = 0.0
    if keep[-1] != t.size - 1:
        keep.append(t.size - 1)
    idx = np.asarray(keep)
    t, T, I = t[idx], T[idx], I[idx]
    return t, T, I, np.diff(t)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class LumpedFit:
    c_th_j_k: float
    beta_w_a2: float          # 等效内阻 Ω（Q = β·I²）；q 给定时为 nan
    g_w_k: float              # h·A
    h_w_m2k: float
    area_m2: float
    t_amb_c: float
    rmse_k: float
    n_points: int
    n_segments: int
    success: bool
    traj: list[dict] = field(repr=False, default_factory=list)
    # 每段: {"t_s", "t_meas_c", "t_sim_c"}


@dataclass
class TwoNodeFit:
    c_core_j_k: float
    c_shell_j_k: float
    r_int_k_w: float
    r_conv_k_w: float
    beta_w_a2: float
    h_w_m2k: float
    area_m2: float
    t_amb_c: float
    rmse_k: float
    gap_max_k: float
    n_points: int
    n_segments: int
    success: bool
    traj: list[dict] = field(repr=False, default_factory=list)


# ---------------------------------------------------------------------------
# 单节点集总（多段共享参数）
# ---------------------------------------------------------------------------


def fit_lumped(
    segments: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    t_amb_c: float,
    area_m2: float = 0.00531,
    c_th_j_k: Optional[float] = None,
    q_w: Optional[Sequence[np.ndarray]] = None,
    drift: bool = False,
    t_ref_s: float = 0.0,
) -> LumpedFit:
    """单节点拟合，多段共享参数。

    ``segments`` 为 ``[(t_s, temp_c, current_a), ...]``（t_s 用**绝对
    测试时间**，漂移项才对得上）；``q_w`` 给定时用外部热源（β 不辨识），
    否则用 β·I²。

    ``drift=True`` 时环境温度取线性漂移 T_amb(t)=a0+a1·(t−t_ref)，
    (a0, a1) 作为干扰参数联合拟合。**不能用静置点温度直接当 T_amb**：
    2 min 的静置远不够 ~4000 s 的热时间常数平衡，静置点温度是"带残余
    自热的电芯温度"，用它会把 G 拟合到 600 W/(m²·K) 量级的荒唐值。
    快动态（I²，秒–分钟）与慢漂移（小时）时标正交，联合拟合良定。

    **结构可辨识性**：(C, β, G) → (αC, αβ, αG) 时输出 T(t) 不变，
    单靠温升曲线只能辨识 τ=C/G 与 β/G 两个组合。故 ``c_th_j_k``
    给定时固定 C、只辨识 (β, G)（well-posed）；不给定时三参数自由拟合，
    返回值仅用于展示不可辨识脊（ridge），**不可当物理参数用**。
    """
    fixed_c = c_th_j_k is not None
    fixed_q = q_w is not None
    prepped = []
    for i, seg in enumerate(segments):
        t, T, I = seg[0], seg[1], seg[2]
        tt, TT, II, dt = _prepare(t, T, I)
        u = np.asarray(q_w[i], float)[
            np.searchsorted(np.asarray(t, float), tt)
        ] if fixed_q else II ** 2
        prepped.append((tt, TT, II, dt, u))

    def unpack(p):
        off = 0
        if fixed_c:
            C = float(c_th_j_k)
            n_th = 0
        else:
            C = float(np.exp(p[0]))
            n_th = 1
        if fixed_q:
            G = float(np.exp(p[n_th]))
            beta = float("nan")
            n_th += 1
        else:
            beta, G = float(np.exp(p[n_th])), float(np.exp(p[n_th + 1]))
            n_th += 2
        if drift:
            a0, a1 = float(p[n_th]), float(p[n_th + 1])
        else:
            a0, a1 = t_amb_c, 0.0
        return C, beta, G, a0, a1

    def sim_all(p):
        C, beta, G, a0, a1 = unpack(p)
        outs = []
        for tt, TT, II, dt, u in prepped:
            ta = a0 + a1 * (tt - t_ref_s) / 3600.0
            if fixed_q:
                A = np.array([[-G / C]])
                B = np.array([[1.0 / C]])
                uu = (u + G * ta).reshape(1, -1)
            else:
                A = np.array([[-G / C]])
                B = np.array([[beta / C, G / C]])
                uu = np.stack([u, ta])
            y0 = np.array([TT[0]])
            outs.append((tt, TT, _simulate_lti(A, B, np.array([1.0]), y0, uu, dt)))
        return outs

    def resid(p):
        return np.concatenate([y - TT for _, TT, y in sim_all(p)])

    base = ([50.0] if not fixed_c else []) + (
        [0.05] if fixed_q else [0.03, 0.05])
    p0 = np.log(base).tolist()
    if drift:
        p0 += [t_amb_c, 0.0]
    res = least_squares(resid, np.array(p0), method="lm", max_nfev=1500)
    outs = sim_all(res.x)
    C, beta, G, _, _ = unpack(res.x)
    rmse = float(np.sqrt(np.mean(np.concatenate([y - TT for _, TT, y in outs]) ** 2)))
    return LumpedFit(
        c_th_j_k=C, beta_w_a2=beta, g_w_k=G,
        h_w_m2k=float(G / area_m2), area_m2=area_m2, t_amb_c=t_amb_c,
        rmse_k=rmse,
        n_points=int(sum(TT.size for _, TT, _ in outs)),
        n_segments=len(outs), success=bool(res.success),
        traj=[{"t_s": tt, "t_meas_c": TT, "t_sim_c": y} for tt, TT, y in outs],
    )


# ---------------------------------------------------------------------------
# 两节点芯-壳（多段共享参数）
# ---------------------------------------------------------------------------


def fit_two_node(
    segments: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    t_amb_c: float,
    area_m2: float = 0.00531,
) -> TwoNodeFit:
    """两节点（芯-壳）拟合，输出取壳（表面）温度，多段共享 5 参数。

    C_c·dTc/dt = β·I² − (Tc−Ts)/R_int
    C_s·dTs/dt = (Tc−Ts)/R_int − (Ts−T_amb)/R_conv
    """
    prepped = [_prepare(t, T, I) for t, T, I in segments]

    def matrices(p):
        Cc, Cs, Ri, Rc, beta = np.exp(p)
        A = np.array([
            [-1.0 / (Ri * Cc), 1.0 / (Ri * Cc)],
            [1.0 / (Ri * Cs), -1.0 / (Ri * Cs) - 1.0 / (Rc * Cs)],
        ])
        B = np.array([
            [beta / Cc, 0.0],
            [0.0, 1.0 / (Rc * Cs)],
        ])
        return A, B

    def sim_all(p, out_row: int):
        A, B = matrices(p)
        C_out = np.zeros(2)
        C_out[out_row] = 1.0
        outs = []
        for tt, TT, II, dt in prepped:
            vv = np.stack([II ** 2, np.full_like(II, t_amb_c)])
            y0 = np.array([TT[0], TT[0]])
            outs.append((tt, TT, _simulate_lti(A, B, C_out, y0, vv, dt)))
        return outs

    def resid(p):
        return np.concatenate([y - TT for _, TT, y in sim_all(p, 1)])

    p0 = np.log([40.0, 20.0, 2.0, 20.0, 0.03])
    res = least_squares(resid, p0, method="lm", max_nfev=2500)
    outs_surf = sim_all(res.x, 1)
    outs_core = sim_all(res.x, 0)
    Cc, Cs, Ri, Rc, beta = np.exp(res.x)
    rmse = float(np.sqrt(np.mean(
        np.concatenate([y - TT for _, TT, y in outs_surf]) ** 2)))
    gap = float(np.max(np.concatenate(
        [yc - ys for (_, _, yc), (_, _, ys) in zip(outs_core, outs_surf)])))
    return TwoNodeFit(
        c_core_j_k=float(Cc), c_shell_j_k=float(Cs),
        r_int_k_w=float(Ri), r_conv_k_w=float(Rc),
        beta_w_a2=float(beta),
        h_w_m2k=float(1.0 / (Rc * area_m2)), area_m2=area_m2,
        t_amb_c=t_amb_c, rmse_k=rmse, gap_max_k=gap,
        n_points=int(sum(TT.size for _, TT, _ in outs_surf)),
        n_segments=len(outs_surf), success=bool(res.success),
        traj=[{"t_s": tt, "t_meas_c": TT,
               "t_sim_c": ys, "t_core_c": yc}
              for (tt, TT, ys), (_, _, yc) in zip(outs_surf, outs_core)],
    )
