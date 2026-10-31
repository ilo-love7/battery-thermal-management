"""电芯几何、热物性与冷却方案库。

所有数字的来源分三类，代码里严格区分
------------------------------------
1. **PyBaMM 参数集**（Chen2020，DOI 10.1149/1945-7111/ab9050，5 Ah LGM50/21700）：
   `Cell volume [m3]`、`Cell cooling surface area [m2]`，以及各层材料的
   密度/比热/厚度。热容由这些量现场算出（`volumetric_heat_capacity`），
   不硬编码，避免"书上抄一个数"和模型内部值打架。
   实测两者一致到 0.03%（1.768e6 vs 1.7676e6 J/(m³·K)）。
2. **几何常识**（21700 = Ø21 mm × 70 mm）：半径、高、质量。仅用于把体积生成率
   q''' 换算成温升、以及圆柱径向导热的解析解。
3. **工程经验值**（换热系数分区、卷芯径向导热系数）：带明确区间，用于扫参设计，
   不当文献引用。凡此类数字，docstring 里标〔工程经验值〕。

为什么换热系数按"区间"而不是"点"
--------------------------------
文献里"液冷 h=500"和"液冷 h=2000"都能找到，差 4 倍。本项目的做法是：
每个方案给一个区间 + 一个代表值，扫参扫代表值，稳健性检查扫区间端点。
选型结论若在区间两端翻转，报告里必须写出来。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

#: 通用气体常数
R_GAS = 8.31446261815324  # J/(mol·K)
#: 寿命折算的参考温度（25 °C）
T_REF_K = 298.15


# --------------------------------------------------------------------------- #
# 电芯
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CellSpec:
    """单电芯的几何与热物性。

    默认值是 21700 / 5 Ah LGM50（Chen2020 参数集对应的电芯）。
    """

    name: str = "LGM50 21700 (Chen2020)"
    capacity_ah: float = 5.0
    radius_m: float = 0.0105          # Ø21 mm
    height_m: float = 0.070           # 70 mm
    volume_m3: float = 2.42e-5        # PyBaMM `Cell volume [m3]`
    surface_area_m2: float = 0.00531  # PyBaMM `Cell cooling surface area [m2]`
    mass_kg: float = 0.070            # 21700 典型 ~68-72 g〔工程经验值〕
    #: 卷芯径向导热系数。层叠结构径向热阻大，文献区间 0.5–2 W/(m·K)〔工程经验值〕。
    k_radial_w_mk: float = 1.0
    #: 轴向（沿极耳方向）导热系数，铜/铝箔连续导热，明显高于径向〔工程经验值〕。
    k_axial_w_mk: float = 20.0

    @property
    def lateral_area_m2(self) -> float:
        """圆柱侧面面积（不含两端）。"""
        import math

        return 2.0 * math.pi * self.radius_m * self.height_m

    def conduction_resistance_core_to_surface(self, k_radial: float | None = None) -> float:
        """卷芯中心到表面的径向导热热阻，K/W。

        均匀内热源圆柱的稳态解给出 ``T_core - T_surface = q''' R² / (4 k)``，
        总量 ``Q = q''' π R² L``，代入得 ``ΔT = Q / (4 π k L)``，
        因此等效热阻 ``R = 1 / (4 π k L)``。这条式子是本项目"核心-表面两节点"
        模型里 ``R_int`` 的物理来源，不是拍脑袋的分流系数。
        """
        k = self.k_radial_w_mk if k_radial is None else k_radial
        import math

        return 1.0 / (4.0 * math.pi * k * self.height_m)

    @property
    def thermal_capacity_j_k(self) -> float:
        """电芯总热容 J/K（按各层材料体积加权，见 ``volumetric_heat_capacity``）。"""
        return volumetric_heat_capacity("Chen2020") * self.volume_m3


CELL = CellSpec()


def volumetric_heat_capacity(parameter_set: str = "Chen2020") -> float:
    """体积热容 J/(m³·K)，按"五层材料厚度加权"算，不抄常数。

    ``ρc_p = Σ(ρ_i c_p,i L_i) / Σ L_i``，层为：负极 / 隔膜 / 正极 / 负极集流体 / 正极集流体。

    Chen2020 得 1.768e6 J/(m³·K)，与 PyBaMM 模型内部变量
    ``Volume-averaged effective heat capacity`` 的 1.7676e6 一致（相对差 0.03%），
    说明这个手算口径和求解器用的是同一套数。乘体积 2.42e-5 m³ → 42.78 J/K。

    换参数集时这个值必须重算：Mohtat2020 是 3.09e6 J/(m³·K) × 3.92e-5 m³ = 121 J/K，
    比 21700 大 2.8 倍——**热容不能跨参数集借用**。
    """
    try:
        import pybamm

        pv = pybamm.ParameterValues(parameter_set)
        num = 0.0
        den = 0.0
        for layer in ("Negative electrode", "Separator", "Positive electrode"):
            rho = float(pv[f"{layer} density [kg.m-3]"])
            cp = float(pv[f"{layer} specific heat capacity [J.kg-1.K-1]"])
            thick = float(pv[f"{layer} thickness [m]"])
            num += rho * cp * thick
            den += thick
        for cc in ("Negative", "Positive"):
            rho = float(pv[f"{cc} current collector density [kg.m-3]"])
            cp = float(pv[f"{cc} current collector specific heat capacity [J.kg-1.K-1]"])
            thick = float(pv[f"{cc} current collector thickness [m]"])
            num += rho * cp * thick
            den += thick
        return num / den
    except Exception:  # pragma: no cover - 无 PyBaMM 时的兜底（Chen2020 实测值）
        return 1.768e6


# --------------------------------------------------------------------------- #
# 冷却方案
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CoolingScheme:
    """一种冷却方案，统一折算为等效对流换热系数 h（W/(m²·K)）。

    这是本项目的方法论核心：**风冷/液冷/浸没式的差别，在设计前期可以统一表达为 h**。
    代价是丢掉流场信息（冷板温度分布、局部死区），所以模组级另建冷却液沿程温升模型
    （`module.py`）补回"沿流动方向的一致性"这一维度。
    """

    key: str
    name: str
    h_lo: float
    h_mid: float
    h_hi: float
    #: 参与换热的表面积占比。同样的 h，作用面积差 4 倍，温升就差 4 倍。
    #: 圆柱电芯贴底部冷板时，真正贴合的只有端面 + 局部侧面〔工程经验值〕。
    area_fraction: float = 1.0
    #: PCM 层等效热容增量，J/K（仅 PCM 方案非零）
    pcm_extra_capacity_j_k: float = 0.0
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "h_lo": self.h_lo,
            "h_mid": self.h_mid,
            "h_hi": self.h_hi,
            "area_fraction": self.area_fraction,
            "pcm_extra_capacity_j_k": self.pcm_extra_capacity_j_k,
            "note": self.note,
        }


#: 方案库。h 区间为〔工程经验值〕：自然对流 5–10、强制风冷 25–100、
#: 底部液冷板 300–800、浸没式 1000–2000 W/(m²·K)。
SCHEMES: dict[str, CoolingScheme] = {
    s.key: s
    for s in [
        CoolingScheme(
            "natural", "自然对流", 5.0, 8.0, 12.0, area_fraction=1.0,
            note="无风扇，靠外壳自然对流+辐射；h 随安装姿态变化大",
        ),
        CoolingScheme(
            "forced_air", "强制风冷", 25.0, 50.0, 100.0, area_fraction=0.85,
            note="风扇强制对流，风道设计决定上限；电芯间气流可达性约 85%",
        ),
        CoolingScheme(
            "cold_plate", "底部液冷板", 300.0, 500.0, 800.0, area_fraction=0.25,
            note="液冷板贴底；圆柱电芯只有端面+局部侧面贴合，有效面积仅约 1/4",
        ),
        CoolingScheme(
            "immersion", "浸没式", 1000.0, 1500.0, 2000.0, area_fraction=1.0,
            note="单相浸没，全表面浸润，h 高且均匀",
        ),
        CoolingScheme(
            "pcm", "PCM 复合（自然对流基底）", 5.0, 8.0, 12.0, area_fraction=1.0,
            pcm_extra_capacity_j_k=120.0,
            note="自然对流 + 相变层；只增加热容不增加散热能力",
        ),
    ]
}


def schemes_enabled(keys: Iterable[str] | None = None) -> list[CoolingScheme]:
    if keys is None:
        return list(SCHEMES.values())
    return [SCHEMES[k] for k in keys]


# --------------------------------------------------------------------------- #
# 参数集
# --------------------------------------------------------------------------- #
def load_parameter_values(
    parameter_set: str = "Chen2020",
    t_amb_c: float = 25.0,
    soc0: float = 1.0,
    h: float | None = None,
):
    """取一套 PyBaMM 参数值，并把环境温度/初始 SOC/换热系数写进去。

    几个必须显式处理的坑
    --------------------
    * ``set_initial_state(soc0)`` 不能省。Mohtat2020 默认初始浓度使得满电电压高于
      4.2 V 上截止，实验步会被判 "infeasible at initial conditions" 而静默跳过，
      结果是 ``EmptySolution``——不是报错，是空结果，很容易误判成"模型跑通了"。
    * 初始温度与环境温度必须同时设。只改 ``Ambient temperature`` 会让电芯从 25 °C
      开始、向 45 °C 环境"升温"，前 10 min 的温升方向是反的。
    * ``h`` 为 None 时保留参数集自带的 ``Total heat transfer coefficient``
      （Chen2020 = 10，Mohtat2020 = 5），便于复现原作者的边界条件。
    """
    import pybamm

    pv = pybamm.ParameterValues(parameter_set)
    pv.set_initial_state(soc0)
    t_k = t_amb_c + 273.15
    pv.update(
        {
            "Ambient temperature [K]": t_k,
            "Initial temperature [K]": t_k,
        }
    )
    if h is not None:
        pv.update({"Total heat transfer coefficient [W.m-2.K-1]": float(h)})
    return pv


def ambient_k(t_amb_c: float) -> float:
    return t_amb_c + 273.15


#: x-full（穿厚分辨）热模型需要、而 Chen2020/Mohtat2020 参数集没有提供的键。
#: 数值取〔工程经验值〕：表面换热沿用同一 h，极耳几何按 21700 典型值。
#: 这些键只影响 x-full 的边界细节，不影响 lumped 主线（见报告 §2.3）。
X_FULL_EXTRA: dict[str, float] = {
    "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 10.0,
    "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 10.0,
    "Negative tab heat transfer coefficient [W.m-2.K-1]": 10.0,
    "Positive tab heat transfer coefficient [W.m-2.K-1]": 10.0,
    "Edge heat transfer coefficient [W.m-2.K-1]": 10.0,
    "Negative tab width [m]": 2e-3,
    "Positive tab width [m]": 2e-3,
}


def apply_x_full_extras(pv, h: float | None = None) -> None:
    """把 x-full 缺的键补进 ParameterValues（就地修改）。

    PyBaMM 26.10 的 ``thermal: "x-full"`` 会索引一批极耳/边缘换热参数，
    Chen2020 没有提供；不补就会 KeyError。补的值只用于 x-full 勘察，
    lumped 主线不经过这里。
    """
    upd = dict(X_FULL_EXTRA)
    if h is not None:
        for k in list(upd):
            if "heat transfer" in k:
                upd[k] = float(h)
    pv.update(upd)
