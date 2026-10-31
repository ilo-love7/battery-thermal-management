"""CALCE CS2 循环老化数据：解析与模型对标。

数据
----
CALCE CS2 系列（1.1 Ah 方形 LFP，Arbin 循环老化）：每个 zip 是一支电芯
的多次数据导出（xlsx，按日期分批）。关键列：

* ``Discharge_Capacity(Ah)``——**全程累计**（每循环 ≈ +1.04 Ah），逐循环差分得容量；
* ``Internal_Resistance(Ohm)``——脉冲法 DC 内阻（>0 的行）；
* ``Current(A)``——放电约 −1.1 A（1C）。

本模块做三件事（为 scripts/19 服务，也可测试）：

1. **逐循环抽取 + 缓存**（parquet，一次解析反复用）；
2. **实测衰减定律**：f = a·N^b 的幂律拟合（b≈0.5 SEI 型 / b≈1 线性）；
3. **实测内阻增长 β**：(R/R0 − 1) vs f 的斜率——直接检验 feedback.py
   假设的 β∈[1.5, 3.5]。
"""
from __future__ import annotations

import glob
import io
import os
import re
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

DEFAULT_ROOT = r"D:/项目/02_数据集/01_国际公开集/CALCE_UMD"

USECOLS = [
    "Test_Time(s)", "Step_Index", "Cycle_Index", "Current(A)",
    "Voltage(V)", "Discharge_Capacity(Ah)", "Internal_Resistance(Ohm)",
]


def list_cs2_cells(root: str | None = None) -> list[dict]:
    """列出含 xlsx 的 CS2 电芯包（txt 格式的 CS2_21 等不解析）。"""
    root = root or os.environ.get("CALCE_ROOT", DEFAULT_ROOT)
    out = []
    for path in sorted(glob.glob(os.path.join(root, "CS2_*.zip"))):
        with zipfile.ZipFile(path) as z:
            n_xlsx = sum(1 for n in z.namelist() if n.lower().endswith(".xlsx"))
        if n_xlsx == 0:
            continue
        out.append({
            "cell": os.path.basename(path).replace(".zip", ""),
            "path": path,
            "n_files": n_xlsx,
        })
    return out


def _extract_one_file(args: tuple) -> list[dict]:
    """子进程：解析单个 xlsx → 每循环一行汇总（段内差分）。

    每份 xlsx 是一段独立导出（Test_Time 从 30 起、容量段内累计），
    因此**段内**差分累计容量得逐循环容量——跨文件差分是错的
    （会把段边界算进容量，产生 0.2 Ah 假容量和负衰减）。
    """
    zip_path, member = args
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(member)
    try:
        df = pd.read_excel(io.BytesIO(raw), sheet_name=1, usecols=USECOLS)
    except Exception:
        return []
    if "Cycle_Index" not in df.columns or df.empty:
        return []
    rows = []
    prev_end = 0.0
    for cyc, g in df.groupby("Cycle_Index"):
        cap_end = float(g["Discharge_Capacity(Ah)"].max())
        r = g["Internal_Resistance(Ohm)"]
        r_pos = r[r > 0]
        i_dis = g.loc[g["Current(A)"] < -0.2, "Current(A)"]
        rows.append({
            "member": os.path.basename(member),
            "cycle_in_file": int(cyc),
            "t_end_s": float(g["Test_Time(s)"].max()),
            "cap_end_ah": cap_end,
            "cap_ah": cap_end - prev_end,   # 段内差分
            "r_ohm": float(r_pos.median()) if len(r_pos) else np.nan,
            "i_dis_a": float(-i_dis.mean()) if len(i_dis) else np.nan,
            "n_rows": int(len(g)),
        })
        prev_end = cap_end
    return rows


def _member_date(member: str) -> tuple:
    """从文件名解析导出日期 CS2_xx_M_D_YY → (y, m, d)，用于时序拼接。"""
    m = re.search(r"_(\d{1,2})_(\d{1,2})_(\d{2})\.xlsx$", os.path.basename(member))
    if not m:
        return (9999, 12, 31)
    mm, dd, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return (2000 + yy, mm, dd)


def extract_cell(zip_path: str, cache_dir: str | None = None) -> pd.DataFrame:
    """解析一支电芯全部 xlsx → 逐循环汇总 DataFrame（缓存 parquet）。

    处理链：段内差分 → 去重（重复导出）→ 按文件名日期时序拼接 →
    全局循环序。列：cycle、cap_ah（逐循环容量）、cap_end_ah（全局累计）、
    r_ohm、i_dis_a、fade（相对前 5 循环均值）、r_ratio、date（导出日期序）。
    """
    cell = os.path.basename(zip_path).replace(".zip", "")
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"cs2_{cell}_per_cycle.parquet")
        if os.path.exists(cache):
            return pd.read_parquet(cache)

    with zipfile.ZipFile(zip_path) as z:
        members = sorted(n for n in z.namelist() if n.lower().endswith(".xlsx"))
    args = [(zip_path, m) for m in members]
    with ProcessPoolExecutor(max_workers=4) as ex:
        results = ex.map(_extract_one_file, args)

    frames = []
    seen: set = set()
    for member, part in zip(members, results):
        if not part:
            continue
        # 重复导出去重：同 (行数, 累计容量, 总时间) 判为同一段
        sig = (part[0]["n_rows"], round(part[-1]["cap_end_ah"], 3),
               round(part[-1]["t_end_s"], 1))
        if sig in seen:
            continue
        seen.add(sig)
        seg = pd.DataFrame(part)
        y, m, d = _member_date(member)
        seg["date_key"] = f"{y:04d}-{m:02d}-{d:02d}"   # 补零保证字典序=时间序
        frames.append(seg)
    if not frames:
        raise RuntimeError(f"{cell}: 未解析到循环数据")

    # 按导出日期时序拼接（等价于按测试顺序；见 docstring 假设）
    frames.sort(key=lambda f: f["date_key"].iloc[0])
    df = pd.concat(frames, ignore_index=True)
    # 段内伪影过滤：逐循环容量异常小（<0.05 Ah）丢弃
    df = df[df["cap_ah"] > 0.05].copy().reset_index(drop=True)
    df["cycle"] = np.arange(1, len(df) + 1)
    df["cap_end_ah"] = df["cap_ah"].cumsum()
    # 基准：前 5 循环均值（新化成波动）
    df["fade"] = 1.0 - df["cap_ah"] / df["cap_ah"].iloc[:5].mean()
    df["r_ratio"] = df["r_ohm"] / df["r_ohm"].iloc[:5].mean()

    if cache_dir:
        df.to_parquet(cache, index=False)
    return df


def fit_fade_power(cycle: np.ndarray, fade: np.ndarray,
                   f_lo: float = 0.01, f_hi: float = 0.25) -> dict:
    """幂律衰减 f = a·N^b（log-log 回归，限定 [f_lo, f_hi] 窗口）。

    窗口默认 1–25 %——EOL 相关段；深衰减尾段（>25 %）的指数不代表
    正常使用区间，且个别协议（部分循环）尾段畸变。
    """
    n = np.asarray(cycle, float)
    f = np.asarray(fade, float)
    m = (f > f_lo) & (f < f_hi) & (n > 0)
    if m.sum() < 5:
        return {"a": np.nan, "b": np.nan, "r2": np.nan, "n_pts": int(m.sum())}
    x = np.log(n[m])
    y = np.log(f[m])
    b, loga = np.polyfit(x, y, 1)
    pred = loga + b * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "a": float(np.exp(loga)), "b": float(b),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "n_pts": int(m.sum()),
    }


def fit_resistance_growth(fade: np.ndarray, r_ratio: np.ndarray,
                          f_lo: float = 0.01, f_hi: float = 0.25) -> dict:
    """线性模型 R/R0 = 1 + β·f → 斜率即 β（feedback.py 的关键参数）。

    同样限定 [f_lo, f_hi] 窗口：闭环模型的 f 范围就是 [0, 0.2]，
    深衰减尾段的非线性增长不应污染 β。
    """
    f = np.asarray(fade, float)
    r = np.asarray(r_ratio, float)
    m = np.isfinite(f) & np.isfinite(r) & (f > f_lo) & (f < f_hi)
    if m.sum() < 5:
        return {"beta": np.nan, "r2": np.nan, "n_pts": int(m.sum()),
                "r_ratio_at_eol": np.nan}
    beta, intercept = np.polyfit(f[m], r[m], 1)
    pred = intercept + beta * f[m]
    ss_res = float(np.sum((r[m] - pred) ** 2))
    ss_tot = float(np.sum((r[m] - r[m].mean()) ** 2))
    return {
        "beta": float(beta),
        "intercept": float(intercept),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "n_pts": int(m.sum()),
        "r_ratio_at_eol": float(intercept + beta * 0.2),
    }


def implied_delta_t_k(damage_ratio: float, ea_j_mol: float = 45_000.0,
                      t_ref_c: float = 25.0) -> float:
    """损伤速率比 → 等效温差：ΔT 使 exp(−Ea/R(1/(T+ΔT)−1/T)) = ratio。

    用于把"实测快倍率 vs 慢倍率的每 EFC 衰减比"翻译成温度语言，
    再与热模型预测的 ΔT 对比——分离热加速与非热（机械）加速。
    """
    from .params import R_GAS
    if damage_ratio <= 0 or not np.isfinite(damage_ratio):
        return np.nan
    t_k = t_ref_c + 273.15
    inv = 1.0 / t_k - R_GAS / ea_j_mol * np.log(damage_ratio)
    return float(1.0 / inv - t_k)


def analyze_cell(df: pd.DataFrame) -> dict:
    """一支电芯的衰减/内阻/倍率画像。

    基准容量用第 2–20 循环中位数（跳过首循环离群——异构协议里
    首循环常是化成/检查循环，容量口径不同）。
    """
    n = df["cycle"].to_numpy(float)
    cap = df["cap_ah"].to_numpy(float)
    r_raw = df["r_ohm"].to_numpy(float)
    # 稳健基准
    base_idx = (n >= 2) & (n <= 20)
    cap0 = float(np.median(cap[base_idx])) if base_idx.sum() >= 3 else float(np.median(cap[:5]))
    r0 = float(np.nanmedian(r_raw[base_idx])) if base_idx.sum() >= 3 else float(np.nanmedian(r_raw[:5]))
    f = 1.0 - cap / cap0
    r = r_raw / r0 if np.isfinite(r0) and r0 > 0 else np.full_like(r_raw, np.nan)
    # EOL 窗口（1–25% 衰减）内的每 EFC 衰减速率
    m = (f > 0.01) & (f < 0.25)
    if m.sum() >= 10:
        fade_rate = float(np.polyfit(n[m], f[m], 1)[0])
    else:
        third = len(df) // 3
        tail = slice(max(third, 5), None)
        fade_rate = float((f[tail][-1] - f[tail][0]) / (n[tail][-1] - n[tail][0])) \
            if len(n[tail]) > 1 else np.nan
    i_nom = float(np.nanmedian(df["i_dis_a"]))
    return {
        "n_cycles": int(len(df)),
        "cap0_ah": cap0,
        "i_dis_a": i_nom,
        "crate": i_nom / cap0 if cap0 > 0 else np.nan,
        "fade_final": float(f[-1]),
        "fade_rate_per_cycle": fade_rate,
        "r0_ohm": r0,
        "fade_power": fit_fade_power(n, f),
        "resistance_growth": fit_resistance_growth(f, r),
    }
