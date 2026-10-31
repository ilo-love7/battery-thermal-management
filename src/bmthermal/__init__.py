"""bmthermal —— 电芯/模组热管理与冷却方案仿真评估。

模块划分
--------
params.py    电芯几何与热物性、冷却方案库、参数集封装
heat.py      PyBaMM DFN 产热提取与分解（欧姆/不可逆/可逆）、Q∝C^α 标度律
network.py   集总热网络（1/2/N 节点）求解与参数辨识
cooling.py   冷却方案扫描、选型 map、PCM 等效热容
module.py    模组级热网络（电芯链 + 冷却液沿程温升）
aging.py     Arrhenius 损伤积分与寿命折算
safety.py    极端工况与预警裕度
duty.py      动态工况（US06）输入与仿真
plotting.py  统一出图风格
"""
from __future__ import annotations

__version__ = "1.0.0"

from .params import (
    CELL,
    SCHEMES,
    CellSpec,
    CoolingScheme,
    load_parameter_values,
    volumetric_heat_capacity,
)
from .heat import (
    C_RATE_TRUSTED,
    HIGH_RATE_SET,
    PRIMARY_SET,
    HeatResult,
    fit_power_law,
    heat_at_soc,
    predict_q,
    q_law,
    q_profile,
    run_dfn_heat,
    sweep_heat,
)
from .network import (
    TwoNodeCell,
    analytic_core_surface_gap,
    build_cell,
    identify_two_node,
    simulate_two_node,
    steady_state_rise,
)
from .aging import damage_integral, equivalent_temperature, life_ratio, mean_temperature_bias
from .cooling import (
    T_LIMIT_C,
    cell_steady_rise,
    pcm_buffer,
    required_h,
    sweep_schemes_dfn,
    sweep_schemes_network,
)
from .module import (
    CP_COOLANT,
    ModuleSpec,
    consistency_report,
    required_flow_lpm,
    simulate_module,
)
from .safety import Scenario, degradation_map, run_scenario
from .duty import US06_PROVENANCE, load_us06, run_dfn_duty, run_network_duty

__all__ = [
    "__version__",
    "CELL",
    "SCHEMES",
    "CellSpec",
    "CoolingScheme",
    "load_parameter_values",
    "volumetric_heat_capacity",
    "PRIMARY_SET",
    "HIGH_RATE_SET",
    "C_RATE_TRUSTED",
    "HeatResult",
    "run_dfn_heat",
    "sweep_heat",
    "heat_at_soc",
    "fit_power_law",
    "predict_q",
    "q_law",
    "q_profile",
    "TwoNodeCell",
    "build_cell",
    "simulate_two_node",
    "identify_two_node",
    "steady_state_rise",
    "analytic_core_surface_gap",
    "damage_integral",
    "equivalent_temperature",
    "life_ratio",
    "mean_temperature_bias",
    "T_LIMIT_C",
    "cell_steady_rise",
    "required_h",
    "pcm_buffer",
    "sweep_schemes_dfn",
    "sweep_schemes_network",
    "CP_COOLANT",
    "ModuleSpec",
    "consistency_report",
    "required_flow_lpm",
    "simulate_module",
    "Scenario",
    "run_scenario",
    "degradation_map",
    "US06_PROVENANCE",
    "load_us06",
    "run_dfn_duty",
    "run_network_duty",
]
