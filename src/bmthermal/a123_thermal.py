"""A123 多温度动态工况数据：R_int(T) 与容量(T) 实测标定（scripts/25 用）。

数据
----
CALCE A123 系列（ANR26650 型 LFP）：DST/US06/FUDS 动态工况在
−10/0/10/20/25/30/40/50 °C 八个温度点各测两支电芯（A1-007/A1-008），
单次导出 xlsx（Arbin 格式，`Channel_1-*` sheet）：

* ``Internal_Resistance(Ohm)``——Arbin 全程 DCIR 估计（逐行），取
  正值中位数作为该温度点的 R_int 代理；
* ``Discharge_Capacity(Ah)``——容量 vs 温度；
* ``Temperature (C)_1``——实测温度（与设定点互验）。

用途：全项目此前 R_int 为常数（25 °C 口径）。低温下 R 通常翻倍以上
→ I²R 产热与稳态 gap 同比放大 → 报告的有效性边界缺一根**温度轴**。
本模块给出 R(T)/R(25) 归一化形状（锚定基准电芯绝对值使用）。

**v1.4 负结果**：Arbin DCIR 列被协议/SOC/电流幅度调制，跨温度非单调
（30–50 °C 上翘）——该通道**标不了** R(T)，fit_r_int_t 的全带拟合
r²<0，只能作下界口径（−10 °C ~1.2×，文献典型 2–2.5×）；容量(T) 可靠。
"""
from __future__ import annotations

import glob
import io
import os
import re
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_ROOT = r"D:/项目/02_数据集/01_国际公开集/CALCE_UMD"
USECOLS = [
    "Test_Time(s)", "Current(A)", "Voltage(V)",
    "Discharge_Capacity(Ah)", "Internal_Resistance(Ohm)", "Temperature (C)_1",
]


def _temp_from_name(name: str) -> float:
    """A123_DST-US06-FUDS-N10.zip → −10；A123_DST-US06-FUDS-25.zip → 25。"""
    m = re.search(r"FUDS-(N?\d+)\.zip$", os.path.basename(name))
    if not m:
        return np.nan
    token = m.group(1)
    return -float(token[1:]) if token.startswith("N") else float(token)


def list_a123_files(root: str | None = None) -> list[dict]:
    root = root or os.environ.get("CALCE_ROOT", DEFAULT_ROOT)
    out = []
    for path in sorted(glob.glob(os.path.join(root, "A123_DST-US06-FUDS-*.zip"))):
        t_set = _temp_from_name(path)
        if not np.isfinite(t_set):
            continue
        with zipfile.ZipFile(path) as z:
            members = sorted(n for n in z.namelist() if n.lower().endswith(".xlsx"))
        for m in members:
            out.append({"zip": path, "member": m, "t_set_c": t_set})
    return out


def _extract_one(args: tuple) -> dict:
    zip_path, member, t_set = args
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(member)
    try:
        # 伪 xlsx（老格式）混存：read_excel 自动选引擎可读，ExcelFile/openpyxl 会炸
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
        sheet_key = next(s for s in sheets if str(s).startswith("Channel"))
        df = sheets[sheet_key]
        df = df[[c for c in USECOLS if c in df.columns]]
    except Exception:
        return {}
    ir = pd.to_numeric(df["Internal_Resistance(Ohm)"], errors="coerce")
    cap = pd.to_numeric(df["Discharge_Capacity(Ah)"], errors="coerce")
    temp = pd.to_numeric(df["Temperature (C)_1"], errors="coerce")
    ir_pos = ir[ir > 0]
    # SOC 对齐取样：R 随 SOC 强调制，全程中位会把温度效应淹没
    # （v1.4 教训：首版全程中位得非单调 R(T)、Ea 拟合 r²<0）
    soc = cap / cap.max() if cap.notna().any() and cap.max() > 0 else cap
    mid = (soc >= 0.4) & (soc <= 0.6) & (ir > 0)
    return {
        "cell": os.path.basename(member).split("-DST")[0],
        "t_set_c": t_set,
        "r_ohm_mid_soc": float(ir[mid].median()) if mid.sum() > 20 else np.nan,
        "n_mid_soc": int(mid.sum()),
        "r_ohm": float(ir_pos.median()) if len(ir_pos) else np.nan,
        "cap_ah": float(cap.max()) if cap.notna().any() else np.nan,
        "t_meas_c": float(temp.mean()) if temp.notna().any() else np.nan,
        "n_rows": int(len(df)),
    }


def extract_a123_dataset(root: str | None = None,
                         cache_csv: str | None = None) -> pd.DataFrame:
    """解析全部温度点 → 每 (温度, 电芯) 一行。可 CSV 缓存。"""
    if cache_csv and os.path.exists(cache_csv):
        return pd.read_csv(cache_csv)
    files = list_a123_files(root)
    rows = []
    for f in files:
        row = _extract_one((f["zip"], f["member"], f["t_set_c"]))
        if row:
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["t_set_c", "cell"]).reset_index(drop=True)
    if cache_csv:
        df.to_csv(cache_csv, index=False)
    return df


@dataclass
class RIntTFit:
    ea_r_j_mol: float          # 表观活化能（R(T) 形状）
    r2: float
    r_ratio_by_temp: dict      # {T_c: R(T)/R(25)}
    r_ohm_by_temp: dict
    cap_by_temp: dict
    t_ref_c: float = 25.0
    meta: dict | None = None

    def ratio(self, t_c: float) -> float:
        """R(T)/R(T_ref)（Arrhenius 外推；插值区间内与实测一致）。"""
        from .params import R_GAS
        t_k = t_c + 273.15
        t_ref_k = self.t_ref_c + 273.15
        return float(np.exp(self.ea_r_j_mol / R_GAS * (1.0 / t_k - 1.0 / t_ref_k)))


def fit_r_int_t(df: pd.DataFrame, t_ref_c: float = 25.0) -> RIntTFit:
    """按温度聚合（两芯中位，SOC 中窗）→ R(T)/R(25) → ln-ratio vs 1/T 拟合 Ea_R。"""
    from .params import R_GAS
    grp = df.groupby("t_set_c").agg(
        r_ohm=("r_ohm_mid_soc", "median"),
        cap_ah=("cap_ah", "median"),
        t_meas_c=("t_meas_c", "median"),
        n=("cell", "count"),
    ).reset_index()
    if t_ref_c not in set(grp["t_set_c"]):
        # 取最近温度点为参考
        t_ref_c = float(grp.iloc[(grp["t_set_c"] - t_ref_c).abs().argmin()]["t_set_c"])
    r_ref = float(grp.loc[grp["t_set_c"] == t_ref_c, "r_ohm"].iloc[0])
    grp["r_ratio"] = grp["r_ohm"] / r_ref

    # ln(ratio) = Ea/R (1/T − 1/Tref)
    t_k = grp["t_set_c"].to_numpy(float) + 273.15
    t_ref_k = t_ref_c + 273.15
    x = 1.0 / t_k - 1.0 / t_ref_k
    y = np.log(grp["r_ratio"].to_numpy(float))
    good = np.isfinite(x) & np.isfinite(y) & (grp["t_set_c"].to_numpy(float) != t_ref_c)
    ea, _ = np.polyfit(x[good], y[good], 1)
    pred = ea * x[good]
    ss_res = float(np.sum((y[good] - pred) ** 2))
    ss_tot = float(np.sum((y[good] - y[good].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return RIntTFit(
        ea_r_j_mol=float(ea * R_GAS),
        r2=r2,
        r_ratio_by_temp={float(t): float(r)
                         for t, r in zip(grp["t_set_c"], grp["r_ratio"])},
        r_ohm_by_temp={float(t): float(r)
                       for t, r in zip(grp["t_set_c"], grp["r_ohm"])},
        cap_by_temp={float(t): float(c)
                     for t, c in zip(grp["t_set_c"], grp["cap_ah"])},
        t_ref_c=float(t_ref_c),
        meta={"note": "R_int 为 Arbin DCIR 代理（SOC 40–60% 窗中位）；形状锚定 25 °C",
              "n_rows_by_temp": {float(t): int(n)
                                 for t, n in zip(grp["t_set_c"], grp["n"])}},
    )
