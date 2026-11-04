"""Oxford Guide xlsx → 块映射表（v1.14 抽出，50/52 复用）。

Guide 结构：Group 标题行 → Cell 行（单元格号，后续行省略）→
每行 TPGx.y 步号 + RPT/Profile/EoL 标签。返回统一表
(tpg, step, cell, group, label, guide)。
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# 列名（Maccor 导出，元素流 utf8 负载解码，v1.12）：
# Cyc, Step, TestTime, StepTime, Amphr, Watthr, Amps, Volts, Temp1, VARx1-15
# ⚠️ TestTime/StepTime 单位是**分钟**（d(Amphr)/(I·dt)=1/60 实测标定，
# 1 Hz 采样）——容量积分必须 /60。


def load_guide_rows(guide_specs: list[tuple[Path, int, int]]) -> pd.DataFrame:
    """Guide xlsx 清单 → 统一表（tpg, step, cell, group, label, guide）。

    guide_specs: [(path, group_lo, group_hi), ...]——同组号可出现在
    多个 Part 的 Guide（如组 1–4 同时有 Part 1 与 Part 2 步段）。
    """
    rows = []
    for path, g_lo, g_hi in guide_specs:
        if not path.exists():
            print(f"    ! guide 缺 {path.name}")
            continue
        df = pd.read_excel(path, sheet_name=0, header=None)
        group = None
        cell = None
        for _, r in df.iterrows():
            v0 = str(r[0]) if pd.notna(r[0]) else ""
            v1 = str(r[1]) if pd.notna(r[1]) else ""
            v2 = str(r[2]) if pd.notna(r[2]) else ""
            if v0.startswith("Group"):
                m = re.search(r"Group\s*(\d+)", v0)
                group = int(m.group(1)) if m else None
                continue
            if v0.strip() == "Cell" or not v1:
                continue
            if v0.strip().isdigit():
                cell = int(v0.strip())
            m = re.match(r"TPG(\d+)(?:\.(\d+))?$", v1.strip())
            if m is None or cell is None or group is None:
                continue
            if not (g_lo <= group <= g_hi):
                continue
            step = int(m.group(2)) if m.group(2) else 1
            rows.append({"tpg": int(m.group(1)), "step": step,
                         "cell": cell, "group": group,
                         "label": v2.strip(),
                         "guide": path.name})
    return pd.DataFrame(rows)


def is_rpt(label: str) -> bool:
    """标签含 RPT 即真（RPT/EoL 终测不在内时由调用方判）。"""
    return "RPT" in label.upper()
