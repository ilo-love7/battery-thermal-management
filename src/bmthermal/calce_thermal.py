"""CALCE A123 实测数据读取（带温度通道）。

与 A4 的 ``bmsoc.io_calce`` 同源数据，但保留 Arbin 的
``Temperature (C)_1`` 实测电芯温度列——A5 的热模型标定靠它。

列约定（Arbin 导出，充电为正/放电为负）：

===============  ==========================================
内部名            Arbin 原名
===============  ==========================================
t_s              Test_Time(s)
current_a        Current(A)
voltage_v        Voltage(V)
q_chg_ah         Charge_Capacity(Ah)
q_dis_ah         Discharge_Capacity(Ah)
temp_c           Temperature (C)_1        ← 实测电芯温度
===============  ==========================================
"""
from __future__ import annotations

import glob
import os
import re
import tempfile
import zipfile
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_ROOT = r"D:/项目/02_数据集/01_国际公开集/CALCE_UMD"

_COLMAP = {
    "Data_Point": "data_point",
    "Test_Time(s)": "t_s",
    "Step_Index": "step",
    "Cycle_Index": "cycle",
    "Current(A)": "current_a",
    "Voltage(V)": "voltage_v",
    "Charge_Capacity(Ah)": "q_chg_ah",
    "Discharge_Capacity(Ah)": "q_dis_ah",
    "Temperature (C)_1": "temp_c",
}
_KEEP = list(_COLMAP.keys())

_TEMP_RE = re.compile(r"(?P<sign>-)?(?P<n>N)?-?(?P<num>\d+)", re.I)


def data_root(root: str | None = None) -> str:
    return root or os.environ.get("CALCE_ROOT", DEFAULT_ROOT)


def parse_package(path: str) -> dict:
    """包名 → 测试类型与温度。

    命名规则与 A4 相同且**两族不一致**，必须分开处理：

    ==========================  ==================  ==================
    包名                        含义              解析
    ==========================  ==================  ==================
    ``A123_OCV-10-20120629``    −10 °C            ``-`` 是负号
    ``A123_OCV25-20120905``     +25 °C            无分隔符
    ``A123_DST-US06-FUDS-10``   +10 °C            ``-`` 是分隔符
    ``A123_DST-US06-FUDS-N10``  −10 °C            ``N`` 表示负
    ==========================  ==================  ==================

    若按「``-`` 一律当负号」解析，``FUDS-25`` 会变成 −25 °C（本项目踩过）。
    """
    base = os.path.basename(path)[:-4]
    kind = "ocv" if "OCV" in base.upper() else "dst_us06_fuds"
    if kind == "ocv":
        rest = base.split("OCV", 1)[1]
    else:
        rest = base.split("FUDS", 1)[1].lstrip("-")  # DST 的 "-" 只是分隔
    tm = _TEMP_RE.match(rest)
    if tm is None:
        temp_c = float("nan")
    else:
        neg = bool(tm.group("sign")) or bool(tm.group("n"))
        temp_c = float(-int(tm.group("num")) if neg else int(tm.group("num")))
    return {"kind": kind, "temp_c": temp_c, "path": path}


def _pick_member(zip_path: str) -> str:
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".xlsx")]
        names = [n for n in names if not os.path.basename(n).startswith("~$")]
        if not names:
            raise FileNotFoundError(f"{zip_path} 内无 xlsx")
        return sorted(names)[0]


def _extract_member(zip_path: str, member: str, cache_dir: Optional[str] = None) -> str:
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "calce_xlsx")
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(
        cache_dir,
        f"{os.path.basename(zip_path)[:-4]}__{os.path.basename(member)}",
    )
    with zipfile.ZipFile(zip_path) as z:
        size = z.getinfo(member).file_size
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return dst
    with zipfile.ZipFile(zip_path) as z, z.open(member) as f, open(dst, "wb") as out:
        out.write(f.read())
    return dst


def read_sheet_with_temp(path: str, sheet: str = "Channel_1-006",
                         limit: Optional[int] = None) -> pd.DataFrame:
    """读 Arbin xlsx 数据表，保留温度列。

    表头之前有测试报告行，用「包含 Data_Point 的行」定位表头。
    """
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        def _has_cells(title: str) -> bool:
            return hasattr(wb[title], "iter_rows")

        if sheet not in wb.sheetnames or not _has_cells(sheet):
            pref = [s for s in wb.sheetnames
                    if s.startswith("Channel") and _has_cells(s)]
            if not pref:
                raise ValueError(f"{path} 无可读数据表（{wb.sheetnames}）")
            sheet = pref[0]
        ws = wb[sheet]
        rows = ws.iter_rows(values_only=True)
        header = None
        for row in rows:
            if row and any(isinstance(v, str) and v.strip() == "Data_Point"
                           for v in row if v):
                header = [str(v).strip() if v is not None else "" for v in row]
                break
        if header is None:
            raise ValueError(f"{path} 未找到表头")
        idx = {name: header.index(name) for name in _KEEP if name in header}
        for req in ("Test_Time(s)", "Current(A)", "Voltage(V)"):
            if req not in idx:
                raise ValueError(f"{path} 缺少必要列 {req}")
        cols = {_COLMAP[c]: [] for c in idx}
        order = list(idx.items())
        n = 0
        for row in rows:
            if row is None:
                continue
            if limit is not None and n >= limit:
                break
            if len(row) <= max(idx.values()):
                continue
            for name, pos in order:
                cols[_COLMAP[name]].append(row[pos])
            n += 1
    finally:
        wb.close()

    df = pd.DataFrame(cols)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["t_s", "current_a", "voltage_v"]).reset_index(drop=True)
    return df


def load_package(zip_path: str, limit: Optional[int] = None) -> pd.DataFrame:
    """读一个温度包（第一个电芯），附元信息。"""
    meta = parse_package(zip_path)
    member = _pick_member(zip_path)
    local = _extract_member(zip_path, member)
    df = read_sheet_with_temp(local, limit=limit)
    df["temp_c_set"] = meta["temp_c"]
    df["kind"] = meta["kind"]
    df["source"] = os.path.basename(zip_path)
    return df


def list_packages(root: str | None = None) -> list[dict]:
    """列出全部 DST-US06-FUDS 动态工况包（含温度元数据）。"""
    root = data_root(root)
    out = []
    for path in sorted(glob.glob(os.path.join(root, "A123_DST-US06-FUDS*.zip"))):
        out.append(parse_package(path))
    return out


def load_all_dynamic(cache_parquet: str | None = None,
                     root: str | None = None,
                     limit: Optional[int] = None) -> pd.DataFrame:
    """加载全部温度档的动态工况数据（带温度列），可落盘缓存。

    缓存存在则直接读缓存。注意：**缓存必须带 ``temp_c`` 列**，
    A4 的旧缓存没有温度通道，不能复用。
    """
    if cache_parquet and os.path.exists(cache_parquet):
        df = pd.read_parquet(cache_parquet)
        if "temp_c" in df.columns:
            return df
    frames = []
    for meta in list_packages(root):
        df = load_package(meta["path"], limit=limit)
        frames.append(df)
        print(f"  loaded {os.path.basename(meta['path'])}: "
              f"{len(df)} rows  T_set={meta['temp_c']}")
    out = pd.concat(frames, ignore_index=True)
    if cache_parquet:
        os.makedirs(os.path.dirname(cache_parquet) or ".", exist_ok=True)
        out.to_parquet(cache_parquet, index=False)
    return out


def split_profiles(df: pd.DataFrame, min_dur_s: float = 1200.0,
                   min_abs_i: float = 1.0) -> dict[str, pd.DataFrame]:
    """切出三段动态放电 ``{"dst", "us06", "fuds"}``。

    判据与 A4 的 ``calce.segments`` 相同：**持续 ≥ 20 min、峰值电流 ≥ 1 A
    且均值为负（放电）** 的 step，包内恰好三个，按时间顺序对应
    包名里的 DST / US06 / FUDS。只看时长会把低温下 6.9 h 的 CC-CV
    充电段一起选进来（A4 踩过），故必须加 ``i_mean < 0``。
    """
    g = df.groupby("step").agg(
        n=("t_s", "size"), t0=("t_s", "min"), t1=("t_s", "max"),
        i_mean=("current_a", "mean"),
        i_absmax=("current_a", lambda s: float(np.abs(s).max())),
    )
    g["dur_s"] = g["t1"] - g["t0"]
    pick = g[(g["dur_s"] >= min_dur_s) & (g["i_absmax"] >= min_abs_i)
             & (g["i_mean"] < 0)].sort_values("t0")
    if len(pick) != 3:
        raise ValueError(f"期望 3 段动态放电，实际 {len(pick)} 段（{list(pick.index)}）")
    labels = ["dst", "us06", "fuds"]
    out = {}
    for label, step in zip(labels, pick.index):
        seg = df[df["step"] == step].copy().reset_index(drop=True)
        out[label] = seg
    return out
