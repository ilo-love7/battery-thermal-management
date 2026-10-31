"""产热提取与分解：用 PyBaMM DFN 算 Q(t)，并拆成欧姆 / 不可逆 / 可逆三部分。

参数集与倍率可信域（这份文件最重要的部分）
------------------------------------------
博远的方案写的是"0.5C–5C 扫参"。实际勘察（scripts/00、§2.4）发现：**这套区间
无法由单一参数集支撑**，必须分工，否则会得到物理上不可能的数：

===========  ==============  ======================  ==========================
参数集        电芯形态         1C/2C/3C 交付容量       结论
===========  ==============  ======================  ==========================
Chen2020     21700 5 Ah      93% / 81% / **38%**      3C 起电解液传输受限，失效
Mohtat2020   软包 5 Ah       96% / 94% / 91%（5C 86%） 高倍率可用，但几何完全不同
===========  ==============  ======================  ==========================

所以本项目的口径是：

* **热学层（几何、热容、面积、$R_{int}$）一律用 Chen2020 的 21700**：53 cm² 表面积、
  42.8 J/K 热容，与真实 21700 一致，也是散热设计要面对的形态。
* **产热层：≤2C 用 Chen2020 实测；3C/5C 用 $Q=kC^{\\alpha}$ 标度律外推**，
  并用 Mohtat2020 的实测值交叉验证（3C 差 24%、5C 差 0.4%，见 §2.4）。
* 3C/5C 的 Q(t) **形态**取自 Mohtat2020（那里模型仍有效），**幅值**归一到标度律预测值。
  这比"拿 2C 曲线压缩"更接近真实，因为不同倍率的产热曲线形态确实不同。

为什么分解用"等温"、温升用"耦合"
--------------------------------
分解要回答"3C 时欧姆热和极化热各占多少"，这个占比应反映**倍率**，不该混进自热反馈；
所以分解走 ``thermal="isothermal"`` + ``calculate heat source for isothermal models="true"``。
温升必须走 ``thermal="lumped"``，让产热与温度互相反馈。两者在 h 很大时应当一致
（实测 2C 下 2.7228 W vs 2.7227 W，差 0.003%），可作为一致性自检。

Chen2020 的熵变系数为 0
-----------------------
``Negative/Positive electrode OCP entropic change`` 在 Chen2020 与 OKane2022 里都是
0.0——参数集缺口，不是代码 bug，因此 Chen2020 下可逆热恒为 0。可逆热的量级用
Mohtat2020 单独评估（报告 §2.5）：1C 时 1.75 W，占其总产热 86%；这个比例明显高于
文献常见的 10–40%，说明该参数集的熵变系数偏大，**只作量级参考**。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .params import CELL, load_parameter_values

#: 主参数集（热学层）
PRIMARY_SET = "Chen2020"
#: 高倍率交叉验证参数集
HIGH_RATE_SET = "Mohtat2020"
#: Chen2020 的可信倍率上限（交付容量 ≥80%）
C_RATE_TRUSTED = 2.0


@dataclass
class HeatResult:
    """一次放电仿真的产热结果。"""

    c_rate: float
    t_amb_c: float
    parameter_set: str
    h: float | None
    isothermal: bool
    time_s: np.ndarray
    soc: np.ndarray
    q_total_w: np.ndarray
    q_ohmic_w: np.ndarray
    q_irrev_w: np.ndarray
    q_rev_w: np.ndarray
    t_cell_c: np.ndarray
    voltage_v: np.ndarray
    delivered_ah: float
    meta: dict = field(default_factory=dict)

    @property
    def q_mean_w(self) -> float:
        return float(np.mean(self.q_total_w))

    @property
    def q_peak_w(self) -> float:
        return float(np.max(self.q_total_w))

    @property
    def heat_energy_j(self) -> float:
        """整段放电的产热总量（梯形积分）。"""
        return float(np.trapezoid(self.q_total_w, self.time_s))

    @property
    def heat_per_ah_j(self) -> float:
        """每交付 1 Ah 的产热，J/Ah。跨倍率比较时比 Q_mean 更公平。"""
        return float(self.heat_energy_j / self.delivered_ah) if self.delivered_ah > 0 else np.nan

    @property
    def t_rise_k(self) -> float:
        return float(np.max(self.t_cell_c) - self.t_amb_c)

    def shares(self) -> dict[str, float]:
        """三部分占 Q_total 的比例（按能量积分，不按瞬时值平均）。"""
        tot = self.heat_energy_j
        if tot <= 0:
            return {"ohmic": np.nan, "irreversible": np.nan, "reversible": np.nan, "other": np.nan}
        out = {
            "ohmic": float(np.trapezoid(self.q_ohmic_w, self.time_s) / tot),
            "irreversible": float(np.trapezoid(self.q_irrev_w, self.time_s) / tot),
            "reversible": float(np.trapezoid(self.q_rev_w, self.time_s) / tot),
        }
        out["other"] = 1.0 - sum(out.values())
        return out

    def summary(self) -> dict:
        s = self.shares()
        return {
            "c_rate": self.c_rate,
            "t_amb_c": self.t_amb_c,
            "parameter_set": self.parameter_set,
            "h": self.h,
            "isothermal": self.isothermal,
            "delivered_ah": self.delivered_ah,
            "duration_s": float(self.time_s[-1]) if self.time_s.size else np.nan,
            "q_mean_w": self.q_mean_w,
            "q_peak_w": self.q_peak_w,
            "heat_energy_j": self.heat_energy_j,
            "heat_per_ah_j": self.heat_per_ah_j,
            "t_rise_k": self.t_rise_k,
            "share_ohmic": s["ohmic"],
            "share_irreversible": s["irreversible"],
            "share_reversible": s["reversible"],
            "share_other": s["other"],
            "v_mean": float(np.mean(self.voltage_v)) if self.voltage_v.size else np.nan,
        }


def _entries(sol, name: str) -> np.ndarray:
    try:
        return np.asarray(sol[name].entries, dtype=float).ravel()
    except Exception:
        return np.zeros(len(sol["Time [s]"].entries))


def run_dfn_heat(
    c_rate: float,
    t_amb_c: float = 25.0,
    v_min: float = 3.0,
    h: float | None = None,
    isothermal: bool = False,
    parameter_set: str = PRIMARY_SET,
    soc0: float = 1.0,
    model: str = "DFN",
) -> HeatResult:
    """跑一段恒流放电，返回产热分解。

    Parameters
    ----------
    c_rate : 倍率（相对额定容量）。
    v_min : 放电截止电压，默认 3.0 V（工程"可用容量"截止；参数集本身的下限是 2.5/2.8 V）。
    h : 换热系数；``isothermal=True`` 时被忽略。
    """
    import warnings

    import pybamm

    if isothermal:
        opts = {"thermal": "isothermal", "calculate heat source for isothermal models": "true"}
    else:
        opts = {"thermal": "lumped"}
    pv = load_parameter_values(parameter_set, t_amb_c=t_amb_c, soc0=soc0, h=h)
    mdl = getattr(pybamm.lithium_ion, model)(options=opts)
    exp = pybamm.Experiment([f"Discharge at {c_rate:g}C until {v_min} V"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol = pybamm.Simulation(mdl, parameter_values=pv, experiment=exp).solve()

    t = _entries(sol, "Time [s]")
    if t.size == 0:
        raise RuntimeError(
            f"{model} {c_rate}C 在 {parameter_set} 上返回空解——检查初始状态是否可行"
            f"（soc0={soc0} 时开路电压是否已越过截止电压）"
        )

    cap = _entries(sol, "Discharge capacity [A.h]")
    nominal = float(pv["Nominal cell capacity [A.h]"])
    return HeatResult(
        c_rate=c_rate,
        t_amb_c=t_amb_c,
        parameter_set=parameter_set,
        h=h,
        isothermal=isothermal,
        time_s=t,
        soc=np.clip(1.0 - cap / nominal, 0.0, 1.0),
        q_total_w=_entries(sol, "Total heating [W]"),
        q_ohmic_w=_entries(sol, "Ohmic heating [W]"),
        q_irrev_w=_entries(sol, "Irreversible electrochemical heating [W]"),
        q_rev_w=_entries(sol, "Reversible heating [W]"),
        t_cell_c=_entries(sol, "Volume-averaged cell temperature [C]"),
        voltage_v=_entries(sol, "Voltage [V]"),
        delivered_ah=float(cap[-1]) if cap.size else 0.0,
        meta={"model": model, "v_min": v_min},
    )


def sweep_heat(
    rates: list[float] | tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    t_amb_c: float = 25.0,
    v_min: float = 3.0,
    isothermal: bool = True,
    parameter_set: str = PRIMARY_SET,
) -> list[HeatResult]:
    """倍率扫描（分解口径）。"""
    return [
        run_dfn_heat(c, t_amb_c=t_amb_c, v_min=v_min, isothermal=isothermal, parameter_set=parameter_set)
        for c in rates
    ]


def fit_power_law(rates, q_values) -> dict:
    """拟合 ``Q = k · C^α``（对数最小二乘）。

    α 的含义：焦耳热 ∝ I² ⇒ α=2；Tafel 区的极化热 ∝ I ⇒ α≈1。
    实测 α≈1.8，说明这个区间内产热几乎全是"平方项"，
    这正是"高倍率只能靠 h、不能靠材料"的定量依据。
    """
    c = np.asarray(rates, dtype=float)
    q = np.asarray(q_values, dtype=float)
    mask = (c > 0) & (q > 0) & np.isfinite(q)
    if mask.sum() < 2:
        return {"alpha": np.nan, "k": np.nan, "r2": np.nan, "n": int(mask.sum())}
    x, y = np.log(c[mask]), np.log(q[mask])
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "alpha": float(a),
        "k": float(np.exp(b)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "n": int(mask.sum()),
    }


def predict_q(c_rate: float, law: dict) -> float:
    """按标度律外推产热，W。"""
    return float(law["k"] * float(c_rate) ** law["alpha"])


def heat_at_soc(res: HeatResult, soc_grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """把 Q(SOC) 重采样到统一网格。

    按 SOC 而不是按时间对齐：不同倍率放电时长差 20 倍，按时间对齐会把
    "1C 的最后一小时"和"3C 的前三分钟"放在一起；按 SOC 对齐才是同一电化学状态。
    """
    if soc_grid is None:
        soc_grid = np.linspace(1.0, 0.0, 101)
    s = np.asarray(res.soc, dtype=float)[::-1]
    q = np.asarray(res.q_total_w, dtype=float)[::-1]
    uniq, idx = np.unique(s, return_index=True)
    return soc_grid, np.interp(soc_grid, uniq, q[idx], left=q[idx][0], right=q[idx][-1])


# --------------------------------------------------------------------------- #
# 统一的 Q(t) 提供者：下游（冷却/模组/寿命/安全）只认这一个入口
# --------------------------------------------------------------------------- #
_LAW_CACHE: dict[str, dict] = {}


def q_law(t_amb_c: float = 25.0, v_min: float = 3.0, rates=(0.5, 1.0, 1.5, 2.0)) -> dict:
    """在 ≤2C 的实测点上标定 ``Q = k C^α``（带缓存）。"""
    key = f"{t_amb_c:.1f}|{v_min:.2f}"
    if key not in _LAW_CACHE:
        qs = []
        for c in rates:
            r = run_dfn_heat(c, t_amb_c=t_amb_c, v_min=v_min, isothermal=True)
            qs.append(r.q_mean_w)
        law = fit_power_law(list(rates), qs)
        law["measured_q"] = dict(zip([float(c) for c in rates], qs))
        law["t_amb_c"] = t_amb_c
        _LAW_CACHE[key] = law
    return _LAW_CACHE[key]


def q_profile(
    c_rate: float,
    t_amb_c: float = 25.0,
    v_min: float = 3.0,
    cache: dict | None = None,
    cross_check: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """返回可直接喂给热网络的 ``(t, Q(t))`` 与说明信息。

    * ``c_rate ≤ 2C``：直接用 Chen2020 实测曲线（模型在此区间可信）。
    * ``c_rate > 2C``：形态取 Mohtat2020 实测（那里模型仍有效），幅值归一到
      ``k C^α`` 外推值；若 ``cross_check``，同时返回 Mohtat 的实测均值用于对照。
    """
    key = (round(float(c_rate), 3), round(float(t_amb_c), 1))
    if cache is not None and key in cache:
        return cache[key]

    law = q_law(t_amb_c=t_amb_c, v_min=v_min)
    q_pred = predict_q(c_rate, law)

    if c_rate <= C_RATE_TRUSTED + 1e-9:
        res = run_dfn_heat(c_rate, t_amb_c=t_amb_c, v_min=v_min, isothermal=True)
        info = {
            "source": "Chen2020 measured",
            "q_mean_measured": res.q_mean_w,
            "q_used": res.q_mean_w,
            "scale": 1.0,
            "delivered_ah": res.delivered_ah,
        }
        out = (res.time_s, res.q_total_w, info)
    else:
        res = run_dfn_heat(c_rate, t_amb_c=t_amb_c, v_min=v_min, isothermal=True, parameter_set=HIGH_RATE_SET)
        scale = q_pred / res.q_mean_w
        info = {
            "source": "Mohtat2020 shape, amplitude scaled to k*C^alpha",
            "q_mean_measured": res.q_mean_w,
            "q_used": res.q_mean_w * scale,
            "scale": scale,
            "delivered_ah": res.delivered_ah,
            "q_law_pred": q_pred,
            "cross_check_dev_pct": (q_pred / res.q_mean_w - 1.0) * 100.0 if cross_check else np.nan,
        }
        out = (res.time_s, res.q_total_w * scale, info)

    if cache is not None:
        cache[key] = out
    return out
