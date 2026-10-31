"""日历-循环老化分解：把"每 EFC 衰减"拆成通量机制与时间机制。

为什么必须拆（scripts/19 的负结果根源）
--------------------------------------
scripts/19 观察到 CS2 0.5C 组每 EFC 衰减**快于** 1.0C 组（3.44e-4 vs
2.79e-4 /EFC）——自加热只有 1–3 K，热加速解释不了。原因：每 EFC 壁钟
时长 0.5C 是 1.0C 的 ~2×，**日历老化混进了循环老化**。

而 scripts/05/07/15/18 的损伤口径 D = ∫exp(−Ea/R(1/T−1/T_ref))dt 是纯
「热秒数」——它隐含的机制是纯日历（∝时间）。该口径会**给高倍率打折**
（EFC 短 → 损伤少），但通量驱动的循环损伤（SEI 开裂等）每 EFC 应大致
恒定、与 EFC 时长无关。分解告诉我们折打错了多少。

两机制模型（每 EFC 衰减）
--------------------------
    fade/N = cyc_per_efc + cal_per_efc
    f(N)   = a_i·N^b + k_cal·t(N)^p        （t = 循环壁钟时间，s）

- 共享 (b, k_cal, p)，逐电芯 a_i（电芯个体差异吸收在振幅）；
- p=0.5 主口径（SEI √t 文献），0.75 敏感性；
- **识别性**：同为 1 EFC，两倍率组 t 不同 → 组间差直接识别 k_cal。
  两组均值口径下有闭式解（``two_group_share``），曲线拟合做细化。

对 feedback 闭环的修正（calendar_bracket）
-----------------------------------------
设 s_cal = 日历占每 EFC 损伤份额（25 °C，工作点 f≈0.1），t_ref = 参考
（1C）EFC 时长，t_s = 场景 EFC 时长。旧 d(u)（热秒数）修正为：

    d_total(u) = d_old(u) · [s_cal + (1 − s_cal)·t_ref/t_s]

- 括号=1 当 t_s=t_ref（基准场景不变）；
- 高倍率 t_s<t_ref → 括号>1：通量损伤不再被打折 → 每 EFC 损伤↑ →
  **方案间寿命比向 1 收缩**（高倍率端不再"白捡"短 EFC）。
- 近似声明：两机制共用 Ea（s_cal 的温度标度未分机制）；p≠1 时
  fade 份额≠损伤份额（p=1 时严格相等），敏感性随 p 档给出。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


def two_group_share(
    fade_per_efc_slow: float,
    fade_per_efc_fast: float,
    t_ratio: float,
) -> float:
    """两组均值闭式解：s_cal = (r − 1)/(t_ratio − 1)。

    组 A（慢倍率，EFC 时长 = t_ratio× 组 B）；r = A 组每 EFC 衰减 / B 组。
    r>1 只能由日历（时长差）解释的部分即 s_cal——**定义在 B 组（快组）
    的 EFC 时长上**。约束 [0, 1]，越界返回截断值并由调用方标注
    （可能是组间非热差异）。
    """
    if fade_per_efc_fast <= 0 or t_ratio <= 1.0:
        raise ValueError("fade_per_efc_fast>0 且 t_ratio>1 才可识别")
    r = fade_per_efc_slow / fade_per_efc_fast
    s = (r - 1.0) / (t_ratio - 1.0)
    return float(np.clip(s, 0.0, 1.0))


def rebase_share(s_cal: float, t_ref: float, t_at: float) -> float:
    """份额换参考时长：s 定义在 ``t_at``，换算到 ``t_ref``。

    由 D_c/k = t_at·(1−s)/s 不变：s2 = s·t_ref / (s·t_ref + (1−s)·t_at)。
    CS2 每循环 3–4 h，场景每 EFC 0.3–1 h——不 rebase 会把日历份额错搬。
    """
    if t_ref <= 0 or t_at <= 0:
        raise ValueError("时长必须为正")
    num = s_cal * t_ref
    den = s_cal * t_ref + (1.0 - s_cal) * t_at
    return float(np.clip(num / den, 0.0, 1.0)) if den > 0 else 0.0


def calendar_bracket(s_cal: float, t_s: float, t_ref: float) -> float:
    """旧损伤口径的修正乘子（见模块 docstring）。

    ``s_cal`` 必须已 rebase 到 ``t_ref``（``rebase_share``）。
    """
    if t_s <= 0 or t_ref <= 0:
        raise ValueError("时长必须为正")
    return float(s_cal + (1.0 - s_cal) * t_ref / t_s)


def bootstrap_two_group_share(
    fade_by_cell: dict[str, float],
    group_slow: list[str],
    group_fast: list[str],
    t_ratio: float,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """闭式解的按电芯 bootstrap CI（组内有放回）。

    主估计器是组间**水平差**（two_group_share）；电芯散布→组均值散布→
    s_cal 分布。形状拟合（fit_decomposition）在 7 芯噪声下 CI 含 0，
    不能作主估计——CI 以此函数为准。
    """
    rng = np.random.default_rng(seed)
    slow = [fade_by_cell[c] for c in group_slow if c in fade_by_cell]
    fast = [fade_by_cell[c] for c in group_fast if c in fade_by_cell]
    if len(slow) < 2 or len(fast) < 2:
        return {"median": np.nan, "lo68": np.nan, "hi68": np.nan, "n": 0}
    vals = []
    for _ in range(n_boot):
        m_slow = float(np.mean(rng.choice(slow, size=len(slow), replace=True)))
        m_fast = float(np.mean(rng.choice(fast, size=len(fast), replace=True)))
        if m_fast <= 0:
            continue
        try:
            vals.append(two_group_share(m_slow, m_fast, t_ratio))
        except ValueError:
            continue
    if not vals:
        return {"median": np.nan, "lo68": np.nan, "hi68": np.nan, "n": 0}
    v = np.array(vals)
    return {
        "median": float(np.median(v)),
        "lo68": float(np.quantile(v, 0.16)),
        "hi68": float(np.quantile(v, 0.84)),
        "n": int(v.size),
    }


def calibrate_from_share(
    s_cal: float,
    f_work: float,
    n_work: float,
    dt_s: float,
    b: float,
    p_cal: float = 0.5,
) -> tuple[float, float]:
    """由工作点份额闭式校准 (a, k)：f_cyc=(1−s)·f_work，f_cal=s·f_work。

    用于分解示意（图 23a）与份额曲线——**不是拟合**，是把闭式 s_cal
    映射进 (a, k) 参数化，形状由固定 (b, p) 决定。
    """
    if not (0.0 < s_cal < 1.0):
        raise ValueError("s_cal 必须在 (0,1)")
    a = (1.0 - s_cal) * f_work / n_work**b
    k = s_cal * f_work / (n_work * dt_s)**p_cal
    return float(a), float(k)


@dataclass
class FadeDecomp:
    b: float                       # 通量幂指数（共享）
    k_cal: float                   # 日历系数（共享），fade/s^p
    p_cal: float                   # 日历幂指数（固定档位）
    a_by_cell: dict = field(default_factory=dict)
    r2: float = np.nan
    n_pts: int = 0
    share_at: dict = field(default_factory=dict)   # {N: s_cal(N)}
    meta: dict = field(default_factory=dict)

    def predict(self, n, t_s: float | np.ndarray, cell: str):
        n = np.asarray(n, float)
        t = np.asarray(t_s, float)
        a = self.a_by_cell[cell]
        f_cyc = a * n**self.b
        f_cal = self.k_cal * t**self.p_cal
        return f_cyc, f_cal


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or x.size < win:
        return x
    k = np.ones(win) / win
    pad = np.pad(x, (win // 2, win - 1 - win // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")[: x.size]


def fit_decomposition(
    cells: dict[str, tuple[np.ndarray, np.ndarray, float]],
    p_cal: float = 0.5,
    f_lo: float = 0.01,
    f_hi: float = 0.25,
    smooth_win: int = 25,
    b_fixed: float | None = None,
    amplitude: str = "shared",
) -> FadeDecomp:
    """联合拟合 f_i(N) = a·N^b + k_cal·(N·dt_i)^p。

    ``cells``: {cell_id: (n, f, dt_s)}，dt_s 为该芯每循环壁钟时长中位数。
    点集限定 f∈(f_lo, f_hi)（EOL 相关窗，与 scripts/19 同口径），
    逐芯平滑（窗口 smooth_win）后进入拟合。

    **可辨识性护栏**（v1.4 踩坑记录）：
    - b 必须固定（``b_fixed``；缺省按逐芯单幂律取中位）：N^b 与 (N·dt)^p
      在观测窗内近共线，b 自由会退化（实测 b→5.75、k 吸收全部曲率）；
    - 振幅默认**全体共享**（``amplitude="shared"``）：同体系的通量损伤
      每 EFC 与倍率无关 → a 必须共享，组间水平差才能归给日历项 k。
      逐芯 a_i 或分组 a_g 都会把组间差吸收掉（k_cal→0 塌缩，两次踩坑）。
    """
    pooled = []
    for cid, (n, f, dt) in cells.items():
        n = np.asarray(n, float)
        f = _smooth(np.asarray(f, float), smooth_win)
        m = (f > f_lo) & (f < f_hi) & (n > 0)
        if m.sum() < 5:
            continue
        pooled.append((cid, n[m], f[m], float(dt)))
    if len(pooled) < 2:
        raise ValueError("有效电芯不足（≥2）")

    if b_fixed is None:
        bs = []
        for cid, n, f, dt in pooled:
            x = np.log(n)
            y = np.log(f)
            bs.append(float(np.polyfit(x, y, 1)[0]))
        b = float(np.clip(np.median(bs), 0.3, 1.2))
        b_source = f"pooled single-power median (clipped to [0.3,1.2]): {np.median(bs):.3f}"
    else:
        b = float(b_fixed)
        b_source = "fixed by caller"

    # 振幅分组
    med_dt = float(np.median([p[3] for p in pooled]))
    if amplitude == "shared":
        key_of = {p[0]: 0 for p in pooled}
        n_amp = 1
    elif amplitude == "group":
        key_of = {p[0]: (0 if p[3] <= med_dt else 1) for p in pooled}
        n_amp = 2
    else:
        key_of = {p[0]: i for i, p in enumerate(pooled)}
        n_amp = len(pooled)

    ids = [p[0] for p in pooled]
    n_pts = sum(p[1].size for p in pooled)
    # 参数：log a_g（振幅）, log k_cal
    def unpack(x):
        la = x[:n_amp]
        lk = x[n_amp]
        return np.exp(la), np.exp(lk)

    def resid(x):
        a_arr, k = unpack(x)
        out = []
        for cid, n, f, dt in pooled:
            a = a_arr[key_of[cid]]
            pred = a * n**b + k * (n * dt)**p_cal
            out.append(pred - f)
        return np.concatenate(out)

    x0 = np.concatenate([np.full(n_amp, np.log(0.01)), [np.log(1e-4)]])
    sol = least_squares(resid, x0, method="trf", max_nfev=20000)
    a_arr, k = unpack(sol.x)
    pred = resid(sol.x)
    ss_res = float(np.sum(pred**2))
    allf = np.concatenate([p[2] for p in pooled])
    ss_tot = float(np.sum((allf - allf.mean())**2))

    fit = FadeDecomp(
        b=b, k_cal=float(k), p_cal=float(p_cal),
        a_by_cell={cid: float(a_arr[key_of[cid]]) for cid in ids},
        r2=1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        n_pts=int(n_pts),
    )
    # 工作点份额：各芯在 f≈0.1 对应 N 处的 s_cal 中位
    for f0 in (0.05, 0.10, 0.15):
        shares = []
        for cid, n, f, dt in pooled:
            # 解 a·N^b + k·(N·dt)^p = f0 数值反解 N（扫描）
            grid = np.geomspace(1, 2e5, 2000)
            a = fit.a_by_cell[cid]
            tot = a * grid**fit.b + fit.k_cal * (grid * dt)**fit.p_cal
            idx = np.searchsorted(tot, f0)
            if idx >= grid.size:
                continue
            N = float(grid[idx])
            f_c = a * N**fit.b
            f_k = fit.k_cal * (N * dt)**fit.p_cal
            shares.append(f_k / (f_c + f_k))
        if shares:
            fit.share_at[f0] = float(np.median(shares))
    fit.meta = {
        "dt_by_cell": {p[0]: p[3] for p in pooled},
        "b_source": b_source,
        "amplitude": amplitude,
        "note": f"p_cal={p_cal} 固定；b 固定；k_cal 共享，振幅按 {amplitude}",
    }
    return fit


def bootstrap_share(
    cells: dict[str, tuple[np.ndarray, np.ndarray, float]],
    n_boot: int = 400,
    p_cal: float = 0.5,
    seed: int = 0,
) -> dict:
    """按电芯重采样的两组份额 s_cal 分布（组内有放回，组间结构保持）。"""
    groups: dict[str, list[str]] = {}
    for cid, (_, _, dt) in cells.items():
        key = "slow" if dt > np.median([v[2] for v in cells.values()]) else "fast"
        groups.setdefault(key, []).append(cid)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sub = {}
        for key, members in groups.items():
            picks = rng.choice(members, size=len(members), replace=True)
            for cid in picks:
                sub[f"{key}_{cid}_{rng.integers(1e9)}"] = cells[cid]
        try:
            fit = fit_decomposition(sub, p_cal=p_cal)
        except Exception:
            continue
        s = fit.share_at.get(0.10)
        if s is not None:
            vals.append(s)
    if not vals:
        return {"median": np.nan, "lo68": np.nan, "hi68": np.nan, "n": 0}
    v = np.array(vals)
    return {
        "median": float(np.median(v)),
        "lo68": float(np.quantile(v, 0.16)),
        "hi68": float(np.quantile(v, 0.84)),
        "n": int(v.size),
    }
