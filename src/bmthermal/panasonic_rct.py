"""Panasonic 18650PF HPPC 脉冲解析：多温度 R(T,SOC,C) 实测标定。

为什么用它（scripts/25 的开放项）
---------------------------------
scripts/25 的低温 R(T) 只有双口径带：A123 DCIR 下界（1.24×）+ 文献典型
（2.2×，标注非实测）——口径带 3.9–41.8 K 宽度来自"没有实测脉冲"。
Kollmeyer（Wisconsin-Madison/McMaster）Panasonic 18650PF 数据集有
HPPC 脉冲（25/10/0/−10/−20 °C）× 5 倍率（0.5/1/2/4/6C）× 13 个 SOC 站，
是**受控温箱脉冲实测**——正是 scripts/25 判词里"温箱脉冲实测收口"
的那批数据。文献带由此可换成实测带。

数据形态（.mat，meas 结构体）：
  Time[s], Current[A]（放电为负）, Voltage[V], Ah（累计放电，负向）,
  Battery_Temp_degC, Chamber_Temp_degC
脉冲协议：每个 SOC 站 5 个 10 s 脉冲（0.5/1/2/4/6C），站间长静置。

口径：
- R_inst = (V_pre − V_1s)/ΔI：脉冲起始 1 s 的欧姆+快 RC；
- R_end  = (V_pre − V_end)/ΔI：10 s 末端（含极化）；
- SOC = 1 + Ah/2.9（PF 标称 2.9 Ah，放电 Ah 为负）；
- V_pre = 脉冲前 30 s 静置中位（每个脉冲前都有长静置）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

TEMP_RE = re.compile(r"(?P<neg>n?)(?P<val>\d+)degC", re.IGNORECASE)
NOMINAL_AH = 2.9
# 脉冲档位（A，放电负向）：0.5/1/2/4/6C @2.9 Ah
PULSE_LEVELS_A = (-1.45, -2.9, -5.8, -11.6, -17.4)
LEVEL_TOL_A = 0.15


def parse_temp(fname: str) -> float:
    m = TEMP_RE.search(Path(fname).name)
    if not m:
        raise ValueError(f"无法从文件名解析温度: {fname}")
    v = float(m.group("val"))
    return -v if m.group("neg").lower() == "n" else v


def list_hppc_files(root: str | Path) -> list[tuple[Path, float]]:
    """目录下全部 HPPC 脉冲文件 → (path, 温度)。"""
    out = []
    for p in sorted(Path(root).glob("*.mat")):
        name = p.name.lower()
        if "hppc" not in name and "pulse" not in name:
            continue
        try:
            out.append((p, parse_temp(p.name)))
        except ValueError:
            continue
    return sorted(out, key=lambda x: x[1])


def load_hppc(path: str | Path) -> pd.DataFrame:
    """meas 结构体 → DataFrame(t, i, v, ah, t_batt)。"""
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["meas"]
    return pd.DataFrame({
        "t": np.asarray(m.Time, float).ravel(),
        "i": np.asarray(m.Current, float).ravel(),
        "v": np.asarray(m.Voltage, float).ravel(),
        "ah": np.asarray(m.Ah, float).ravel(),
        "t_batt": np.asarray(m.Battery_Temp_degC, float).ravel(),
    })


def extract_pulses(df: pd.DataFrame, temp_c: float) -> pd.DataFrame:
    """逐脉冲提取 R_inst/R_end 与 SOC。"""
    t, i, v, ah, tb = (df[c].to_numpy(float) for c in ("t", "i", "v", "ah", "t_batt"))
    on = np.abs(i) > 0.1
    d = np.diff(on.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends = list(np.where(d == -1)[0] + 1)
    if on[0]:
        starts.insert(0, 0)
    if on[-1]:
        ends.append(len(i))
    rows = []
    for s, e in zip(starts, ends):
        seg = i[s:e]
        if seg.size < 3:
            continue
        i_med = float(np.median(seg))
        # 匹配最近档位
        level = min(PULSE_LEVELS_A, key=lambda L: abs(L - i_med))
        if abs(level - i_med) > LEVEL_TOL_A:
            continue
        # 脉冲前静置（起点前 30 s 内取 |I|<0.05 的样本）
        pre = slice(max(0, s - 30), max(1, s - 1))
        pre_mask = np.abs(i[pre]) < 0.05
        if pre_mask.sum() >= 3:
            v_pre = float(np.median(v[pre][pre_mask]))
        else:
            v_pre = float(v[max(0, s - 1)])
        dt = np.median(np.diff(t[s:e])) if e - s > 2 else 1.0
        n1 = int(max(1, round(1.0 / max(dt, 0.1))))
        v_1s = float(np.median(v[s:s + n1]))
        v_end = float(np.median(v[e - n1:e]))
        di = abs(i_med)
        soc = float(1.0 + np.median(ah[s:e]) / NOMINAL_AH)
        rows.append({
            "temp_c": temp_c, "i_level": i_med, "c_rate": di / NOMINAL_AH,
            "soc": min(max(soc, 0.0), 1.0),
            "v_pre": v_pre,
            "r_inst": (v_pre - v_1s) / di,
            "r_end": (v_pre - v_end) / di,
            "t_mid": float(np.median(t[s:e])),
            "t_batt": float(np.median(tb[s:e])),
            "dur_s": float(t[e - 1] - t[s]),
        })
    return pd.DataFrame(rows)


def extract_dataset(root: str | Path, cache_csv: str | Path | None = None) -> pd.DataFrame:
    """全部温度文件 → 逐脉冲长表（可 CSV 缓存）。"""
    if cache_csv is not None and Path(cache_csv).exists():
        return pd.read_csv(cache_csv)
    frames = [extract_pulses(load_hppc(p), tc) for p, tc in list_hppc_files(root)]
    df = pd.concat(frames, ignore_index=True)
    if cache_csv is not None:
        Path(cache_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
    return df


@dataclass
class RIntTFit:
    """R(T) 拟合结果：Arrhenius + 逐温度实测比值。"""

    t_ref_c: float
    ea_j_mol: float
    r_ref: float
    r2: float
    r_abs_by_temp: dict[float, float]
    r_ratio_by_temp: dict[float, float]
    soc_ref: float
    c_ref: float
    metric: str
    monotonic: bool

    def ratio(self, t_c: float) -> float:
        """Arrhenius 外推 g(T)=R(T)/R(T_ref)。"""
        return float(np.exp(self.ea_j_mol / 8.314462618
                            * (1.0 / (t_c + 273.15) - 1.0 / (self.t_ref_c + 273.15))))


def fit_r_int_t(
    pulses: pd.DataFrame,
    soc_ref: float = 0.5,
    c_ref: float = 2.0,
    t_ref_c: float = 25.0,
    metric: str = "r_end",
    soc_tol: float = 0.08,
) -> RIntTFit:
    """固定 (SOC, 倍率) 窗内 R 对 T 的 Arrhenius 拟合。

    温箱脉冲协议受控（同一文件内逐温度同协议）——与 A123 动态工况
    DCIR 列不同，这里 T 是唯一自由变量，单调性成立才拟合。
    """
    m = (
        ((pulses["soc"] - soc_ref).abs() <= soc_tol)
        & ((pulses["c_rate"] - c_ref).abs() <= 0.3)
        & pulses[metric].notna()
        & (pulses[metric] > 0)
    )
    g = pulses[m]
    if g["temp_c"].nunique() < 3:
        raise ValueError(f"温度点不足: {sorted(g['temp_c'].unique())}")
    # 每温度中位
    per_t = g.groupby("temp_c")[metric].median()
    temps = per_t.index.to_numpy(float)
    r_vals = per_t.to_numpy(float)
    # 单调性检查（R 应随 T 升而降）
    order = np.argsort(temps)
    temps_s, r_s = temps[order], r_vals[order]
    mono = bool(np.all(np.diff(r_s) < 0) or np.all(np.diff(r_s) > 0))
    x = 1.0 / (temps + 273.15) - 1.0 / (t_ref_c + 273.15)
    y = np.log(r_vals)
    slope, b = np.polyfit(x, y, 1)
    y_hat = slope * x + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # 约定与 a123_thermal 对齐：ea_r_j_mol = slope·R_GAS，
    # ratio() = exp(ea/R·Δ(1/T))；比值按实测 T_ref 中位归一
    r_gas = 8.314462618
    r_ref = float(per_t.get(t_ref_c, np.exp(b)))
    ratios = {float(T): float(r / r_ref) for T, r in per_t.items()}
    return RIntTFit(
        t_ref_c=t_ref_c, ea_j_mol=float(slope * r_gas), r_ref=r_ref, r2=float(r2),
        r_abs_by_temp={float(T): float(r) for T, r in per_t.items()},
        r_ratio_by_temp=ratios,
        soc_ref=soc_ref, c_ref=c_ref, metric=metric, monotonic=mono,
    )


def rate_nonlinearity(pulses: pd.DataFrame, temp_c: float, soc_ref: float = 0.5,
                      soc_tol: float = 0.08) -> pd.DataFrame:
    """固定温度/SOC 下 R 随脉冲倍率的变化（WP28 的高倍率锚点）。"""
    m = (
        (pulses["temp_c"] == temp_c)
        & ((pulses["soc"] - soc_ref).abs() <= soc_tol)
    )
    g = pulses[m]
    return g.groupby("c_rate")[["r_inst", "r_end"]].median().reset_index()
