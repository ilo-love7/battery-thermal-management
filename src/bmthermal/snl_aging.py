"""Sandia SNL 多化学体系/多倍率老化解析 + 跨体系日历份额估计。

为什么用 SNL 收 s_cal（scripts/23 的开放项）
--------------------------------------------
scripts/23 的日历份额 s_cal 只来自 CS2（1.1 Ah LFP），迁移到基准 NMC
电芯是跨体系近似——C5 寿命比 CI [1.76, 4.77] 的宽度主要由它贡献。
SNL（Preger 2020, JES）镜像有 NCA/NMC/LFP 三种化学体系 × 多放电倍率
（0.5/1/2/3C，0.5C 恒流充电，25 °C 全窗口 0–100%），与 CS2 两组结构
同构：倍率差 → 每 EFC 壁钟时长差 → 可复用 ``two_group_share`` 闭式解。
跨体系开放项由此可关：NMC/NCA 直接给出自己的 s_cal，LFP 与 CS2 对拍。

文件名即元数据（batteryarchive 统一格式）：
    SNL_18650_{chem}_{temp}C_{lo}-{hi}_{chg}-{dis}C_{rep}.pkl
pkl 为 dict：cell_id / cathode_material / nominal_capacity_in_Ah /
cycle_data（逐循环 dict：cycle_number, current_in_A, voltage_in_V,
charge/discharge_capacity_in_Ah, time_in_s, temperature_in_C）。

口径：
- 每循环放电容量 = discharge_capacity_in_Ah 序列最大值（段内累计）；
- 每循环壁钟时长 = time_in_s 跨度（含步内静置，与 CS2 段间 dt 口径
  略不同——两组同协议，t_ratio 比值消掉系统偏差）；
- 0–100% DOD → 1 EFC/循环，fade/循环 = fade/EFC；
- fade 率 = 1–25% 衰减窗线性斜率（与 ``cs2_aging.analyze_cell`` 同口径）。
"""
from __future__ import annotations

import pickle
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NAME_RE = re.compile(
    r"SNL_18650_(?P<chem>[A-Z]+)_(?P<temp>-?\d+)C_"
    r"(?P<lo>\d+)-(?P<hi>\d+)_(?P<chg>[\d.]+)-(?P<dis>[\d.]+)C_(?P<rep>\w+)\.pkl"
)


@dataclass
class SnlCell:
    cell_id: str
    chem: str
    temp_c: float
    soc_lo: float
    soc_hi: float
    c_chg: float
    c_dis: float
    rep: str
    n_cycles: int
    cap0_ah: float
    fade_rate_per_efc: float
    dt_s_per_cycle: float
    fade_final: float

    @property
    def efc_per_cycle(self) -> float:
        return 1.0  # 0–100% DOD


def parse_name(fname: str) -> dict:
    """文件名 → 元数据。"""
    m = NAME_RE.search(Path(fname).name)
    if not m:
        raise ValueError(f"无法解析 SNL 文件名: {fname}")
    d = m.groupdict()
    return {
        "chem": d["chem"],
        "temp_c": float(d["temp"]),
        "soc_lo": float(d["lo"]),
        "soc_hi": float(d["hi"]),
        "c_chg": float(d["chg"]),
        "c_dis": float(d["dis"]),
        "rep": d["rep"],
    }


def fade_window_rate(n: np.ndarray, f: np.ndarray) -> float:
    """1–25% 衰减窗内 fade 对循环数的线性斜率（每循环=每 EFC）。"""
    m = (f > 0.01) & (f < 0.25)
    if m.sum() >= 10:
        return float(np.polyfit(n[m], f[m], 1)[0])
    if len(n) >= 4:
        third = len(n) // 3
        tail = slice(max(third, 5), None)
        dn = n[tail][-1] - n[tail][0]
        return float((f[tail][-1] - f[tail][0]) / dn) if dn > 0 else np.nan
    return np.nan


def summarize_cycles(cycles: list, nominal_ah: float) -> dict:
    """逐循环列表 → 容量序列与逐循环壁钟时长。

    早期条目有重复导出（同签名连续出现，CS2 同款坑）——按
    (时长跨度, 点数, 容量) 签名去重后再统计。
    """
    n_list, cap_list, dt_list = [], [], []
    last_sig = None
    for c in cycles:
        dis = np.asarray(c["discharge_capacity_in_Ah"], dtype=float)
        t = np.asarray(c["time_in_s"], dtype=float)
        if dis.size == 0 or t.size < 2:
            continue
        cap = float(np.max(dis))
        dt = float(t[-1] - t[0])
        sig = (round(dt, 3), t.size, round(cap, 5))
        if sig == last_sig:
            continue
        last_sig = sig
        n_list.append(int(c["cycle_number"]))
        cap_list.append(cap)
        dt_list.append(dt)
    n = np.asarray(n_list, float)
    cap = np.asarray(cap_list, float)
    dt = np.asarray(dt_list, float)
    good = np.isfinite(cap) & (cap > 0.2 * nominal_ah)
    n, cap, dt = n[good], cap[good], dt[good]
    if n.size < 10:
        raise ValueError(f"有效循环不足: {n.size}")
    base = (n >= 1) & (n <= 20)
    cap0 = float(np.median(cap[base])) if base.sum() >= 3 else float(np.median(cap[:5]))
    f = 1.0 - cap / cap0
    # 逐循环 dt 的段内中位（相邻差分剔除异常）
    d_ok = dt[(dt > 60) & (dt < 5 * 24 * 3600)]
    dt_med = float(np.median(d_ok)) if d_ok.size else float(np.median(dt))
    return {
        "n": n,
        "f": f,
        "cap0_ah": cap0,
        "fade_rate_per_efc": fade_window_rate(n, f),
        "dt_s_per_cycle": dt_med,
        "n_cycles": int(n.size),
        "fade_final": float(f[-1]),
    }


def load_cell(zip_path: str | Path, member: str) -> SnlCell:
    """从 SNL.zip 读一支电芯并汇总。"""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            d = pickle.load(fh)
    meta = parse_name(member)
    nominal = float(d.get("nominal_capacity_in_Ah") or 3.0)
    s = summarize_cycles(d["cycle_data"], nominal)
    return SnlCell(
        cell_id=str(d.get("cell_id") or Path(member).stem),
        chem=meta["chem"],
        temp_c=meta["temp_c"],
        soc_lo=meta["soc_lo"],
        soc_hi=meta["soc_hi"],
        c_chg=meta["c_chg"],
        c_dis=meta["c_dis"],
        rep=meta["rep"],
        n_cycles=s["n_cycles"],
        cap0_ah=s["cap0_ah"],
        fade_rate_per_efc=s["fade_rate_per_efc"],
        dt_s_per_cycle=s["dt_s_per_cycle"],
        fade_final=s["fade_final"],
    )


def list_members(zip_path: str | Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        return [n for n in zf.namelist() if n.endswith(".pkl")]


def load_dataset(zip_path: str | Path, cache_csv: str | Path | None = None) -> pd.DataFrame:
    """全量电芯汇总表（可 CSV 缓存——解压+pickle 读 60 支较慢）。"""
    if cache_csv is not None and Path(cache_csv).exists():
        return pd.read_csv(cache_csv)
    rows = []
    for m in list_members(zip_path):
        try:
            c = load_cell(zip_path, m)
        except Exception as e:  # 单芯失败不拖垮全集
            rows.append({"cell_id": Path(m).stem, "error": str(e)})
            continue
        rows.append({
            "cell_id": c.cell_id, "chem": c.chem, "temp_c": c.temp_c,
            "soc_lo": c.soc_lo, "soc_hi": c.soc_hi,
            "c_chg": c.c_chg, "c_dis": c.c_dis, "rep": c.rep,
            "n_cycles": c.n_cycles, "cap0_ah": c.cap0_ah,
            "fade_rate_per_efc": c.fade_rate_per_efc,
            "dt_s_per_cycle": c.dt_s_per_cycle, "fade_final": c.fade_final,
        })
    df = pd.DataFrame(rows)
    if cache_csv is not None:
        Path(cache_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
    return df


def group_stats(
    df: pd.DataFrame, chem: str, temp_c: float, c_dis: float,
    soc_lo: float = 0.0, soc_hi: float = 100.0,
) -> dict:
    """某化学体系/温度/SOC 窗/放电倍率组的 fade 与时长均值。

    SOC 窗必须进过滤——SNL 同目录混有 20–80% 窗电芯，混组会把
    每 EFC 时长算错（0.6 EFC/循环）。
    """
    m = (
        (df["chem"] == chem) & (df["temp_c"] == temp_c) & (df["c_dis"] == c_dis)
        & (df["soc_lo"] == soc_lo) & (df["soc_hi"] == soc_hi)
    )
    g = df[m & df["fade_rate_per_efc"].notna()]
    if g.empty:
        raise ValueError(f"空组: {chem} {temp_c}C {soc_lo}-{soc_hi} {c_dis}C")
    return {
        "cells": g["cell_id"].tolist(),
        "n_cells": int(len(g)),
        "fade_per_efc": float(g["fade_rate_per_efc"].mean()),
        "fade_per_efc_std": float(g["fade_rate_per_efc"].std(ddof=1)) if len(g) > 1 else 0.0,
        "dt_s": float(g["dt_s_per_cycle"].median()),
    }


def pair_share(
    df: pd.DataFrame, chem: str, temp_c: float,
    c_slow: float, c_fast: float,
    soc_lo: float = 0.0, soc_hi: float = 100.0,
) -> dict:
    """一对倍率组 → t_ratio、r、s_cal（定义在快组 EFC 时长）。"""
    from bmthermal.calendar import two_group_share

    slow = group_stats(df, chem, temp_c, c_slow, soc_lo, soc_hi)
    fast = group_stats(df, chem, temp_c, c_fast, soc_lo, soc_hi)
    t_ratio = slow["dt_s"] / fast["dt_s"]
    r = slow["fade_per_efc"] / fast["fade_per_efc"]
    if t_ratio > 1.0:
        s = two_group_share(slow["fade_per_efc"], fast["fade_per_efc"], t_ratio)
    else:
        s = float("nan")  # 时长不随倍率单调——资格否决
    s_raw = (r - 1.0) / (t_ratio - 1.0) if t_ratio != 1.0 else float("nan")
    k_cal = (slow["fade_per_efc"] - fast["fade_per_efc"]) / (slow["dt_s"] - fast["dt_s"]) \
        if slow["dt_s"] != fast["dt_s"] else float("nan")
    c_flux = fast["fade_per_efc"] - k_cal * fast["dt_s"] if np.isfinite(k_cal) else float("nan")
    return {
        "chem": chem, "temp_c": temp_c,
        "c_slow": c_slow, "c_fast": c_fast,
        "t_ratio": float(t_ratio), "fade_ratio": float(r),
        "s_cal_at_fast": s, "s_raw": float(s_raw),
        "k_cal_per_s": float(k_cal), "c_flux_per_efc": float(c_flux),
        "eligible": bool(t_ratio > 1.0 and np.isfinite(s) and 0.0 < s <= 1.0
                         and np.isfinite(s_raw) and s_raw > 0
                         and np.isfinite(c_flux) and c_flux > 0),
        "slow": slow, "fast": fast,
    }
