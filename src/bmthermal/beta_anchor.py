"""beta_anchor · β 跨体系锚：NMC/NCA 内阻老化实测（v1.8, WP35）。

开放项 #5（NMC 内阻老化数据）的收口模块。三条数据路径：

1. **MATR-10 截面**（NCA 21700，94 芯 SOH 62–100%）：
   Arbin 文本，满充（Ah≈0, V≈4.2）C→D 反转提 R(1C)（ΔI=4.84 A），
   R = ΔV/ΔI；容量取电芯清单的 C/97 慢放主容量 → f = 1 − cap/nominal。
   跨芯回归 R = a + b·f → β = b/a（截面口径，bootstrap CI）。
2. **HNEI 纵向**（LCO+NMC 18650，14 芯 >1000 循环）：逐循环放电起始
   反转（CV 尾 → 1.5C）提 R_inst，R/R0 vs f 逐芯拟合 β。
3. **MICH / XJTU 纵向**（NMC111 软包 40 芯 / NCM523 18650 23 芯）：同 2。

统一模型（与 cs2_aging 一致）：R/R0 = 1 + β·f，f∈[0.01, 0.25] 窗口。
教训复用：滚动中位防毛刺（v1.7）；schema 有字段≠有数据（内阻列全零，
全部从 V/I 阶跃提取）；每条测量带 SOC/倍率锚点。

产物：per-cell 提取行（dataset, cell_id, fade, r_ohm, r_ratio…）+ β 汇总。
"""

from __future__ import annotations

import csv
import io
import pickle
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cs2_aging import fit_resistance_growth

# ---- 电芯清单里的标称容量（MATR-10：Panasonic 21700 NCA）----
NOMINAL_AH = 4.84
# 1C 放电电流（A）
RATE_1C_A = 4.84
# 反转提取阈值
DI_MIN_A = 0.3


# ----------------------------------------------------------------- 通用

def smooth_series(x: np.ndarray, w: int = 7) -> np.ndarray:
    """滚动中位（v1.7 毛刺教训）。"""
    x = np.asarray(x, float)
    if x.size < w:
        return x.copy()
    out = np.empty_like(x)
    half = w // 2
    for i in range(x.size):
        lo, hi = max(0, i - half), min(x.size, i + half + 1)
        out[i] = np.nanmedian(x[lo:hi])
    return out


def extract_step_r(i: np.ndarray, v: np.ndarray,
                   di_min: float = DI_MIN_A) -> list:
    """电流阶跃处的 R_inst = ΔV/ΔI。

    返回 [{idx, i_from, i_to, r_ohm, v_from, v_to, into_discharge}]。
    只取 |ΔI| ≥ di_min 的阶跃；into_discharge=阶跃后为放电方向（I<0）。
    """
    i = np.asarray(i, float)
    v = np.asarray(v, float)
    if i.size < 3:
        return []
    di = np.diff(i)
    hits = np.nonzero(np.abs(di) >= di_min)[0]
    out = []
    for j in hits:
        d_i = float(di[j])
        d_v = float(v[j + 1] - v[j])
        if d_i == 0:
            continue
        r = d_v / d_i
        # 方向一致性：放电阶跃（ΔI<0）应伴随 ΔV<0；充电反之
        if not np.isfinite(r) or r <= 0 or r > 0.5:
            continue
        out.append({
            "idx": int(j),
            "i_from": float(i[j]), "i_to": float(i[j + 1]),
            "v_from": float(v[j]), "v_to": float(v[j + 1]),
            "r_ohm": float(r),
            "into_discharge": bool(i[j + 1] < 0),
        })
    return out


def pick_reference_r(hits: list, rate_a: float = RATE_1C_A,
                     tol_frac: float = 0.25) -> float:
    """取 |I_to| 最接近 rate_a 的放电阶跃 R（同 SOC 语境下调用方保证）。"""
    dis = [h for h in hits if h["into_discharge"] and abs(h["i_to"]) > 0]
    if not dis:
        return float("nan")
    best = min(dis, key=lambda h: abs(abs(h["i_to"]) - rate_a))
    if abs(abs(best["i_to"]) - rate_a) > tol_frac * rate_a:
        return float("nan")
    return float(best["r_ohm"])


# ----------------------------------------------------------------- MATR-10

@dataclass
class Matr10Cell:
    cell_id: str
    cap_ah: float
    r_by_rate: dict = field(default_factory=dict)   # rate_a -> r_ohm
    r_1c: float = float("nan")
    fade: float = float("nan")


def _parse_matr10_file(path: str | Path) -> list:
    """Arbin 文本 → 行列表 (amps, volts, state, amp_hr)。"""
    rows = []
    with io.open(path, "r", errors="replace", newline="") as fh:
        hdr = None
        idx = None
        for i, line in enumerate(fh):
            parts = line.rstrip("\r\n").split("\t")
            if i == 1:
                hdr = parts
                idx = {n: j for j, n in enumerate(hdr)}
                continue
            if idx is None or len(parts) < 30:
                continue
            try:
                rows.append((
                    float(parts[idx["Amps"]]),
                    float(parts[idx["Volts"]]),
                    parts[idx["State"]],
                    float(parts[idx["Amp-hr"]]),
                ))
            except ValueError:
                continue
    return rows


def extract_matr10_r(rows: list) -> dict:
    """满充 C→D / R→D 反转 → {rate_a: r_ohm}。

    满充判据：反转前 V≥4.15 且 |I_pre|<0.25（CV 尾或静置），|ΔI|≥DI_MIN。
    注意 Amp-hr 列是步内累积，不能用 ah_pre 判满充（充电步末 ah_pre≈容量，
    会把 C→D 反转全部误杀——实测 11/94 芯假通过）。
    """
    out = {}
    for j in range(1, len(rows)):
        i0, v0, s0, ah0 = rows[j - 1]
        i1, v1, s1, ah1 = rows[j]
        if s1 != "D":
            continue
        d_i = i1 - i0
        if abs(d_i) < DI_MIN_A or i1 >= 0:
            continue
        if not (v0 >= 4.15 and abs(i0) < 0.25):
            continue
        r = (v1 - v0) / d_i
        if not np.isfinite(r) or r <= 0 or r > 0.5:
            continue
        rate = abs(i1)
        # 同倍率重复取中位
        out.setdefault(rate, []).append(float(r))
    return {k: float(np.median(v)) for k, v in out.items()}


def load_matr10_inventory(csv_path: str | Path) -> dict:
    """电芯清单 → {cell_id: cap_ah}（main_cap_ah 列）。"""
    caps = {}
    with io.open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            seq = row["seq"].zfill(6)
            barcode = row["barcode"]
            cell_id = f"ResVal_{seq}_{barcode}"
            try:
                caps[cell_id] = float(row["main_cap_ah"])
            except (KeyError, ValueError):
                continue
    return caps


def scan_matr10(data_dir: str | Path, inventory_csv: str | Path,
                max_files: int = 0) -> list:
    """全量扫描 MATR-10 → [Matr10Cell]。"""
    caps = load_matr10_inventory(inventory_csv)
    cells = []
    files = sorted(Path(data_dir).glob("ResVal_*"))
    if max_files:
        files = files[:max_files]
    for fp in files:
        cell_id = fp.stem  # ResVal_000013_000185
        cap = caps.get(cell_id)
        if cap is None or cap <= 0:
            continue
        rows = _parse_matr10_file(fp)
        r_by_rate = extract_matr10_r(rows)
        if not r_by_rate:
            continue
        r1 = pick_reference_r(
            [{"i_to": -k, "into_discharge": True, "r_ohm": v}
             for k, v in r_by_rate.items()],
            rate_a=RATE_1C_A)
        c = Matr10Cell(cell_id=cell_id, cap_ah=cap,
                       r_by_rate=r_by_rate, r_1c=r1)
        c.fade = float(1.0 - cap / NOMINAL_AH)
        cells.append(c)
    return cells


def fit_beta_cross_sectional(cells: list, fade_lo: float = 0.01,
                             fade_hi: float = 0.38,
                             n_boot: int = 2000, seed: int = 0) -> dict:
    """截面回归 R = a + b·f → β = b/a（bootstrap CI）。

    fade 窗放宽到 [0.01, 0.38]：MATR-10 SOH 62% → f_max≈0.38，
    截面数据没有深衰减尾段污染问题（一次性表征）。
    """
    f = np.array([c.fade for c in cells], float)
    r = np.array([c.r_1c for c in cells], float)
    m = np.isfinite(f) & np.isfinite(r) & (f > fade_lo) & (f < fade_hi) & (r > 0)
    f, r = f[m], r[m]
    if f.size < 8:
        return {"beta": float("nan"), "n_cells": int(f.size),
                "intercept_ohm": float("nan"), "slope_ohm_per_f": float("nan"),
                "r2": float("nan"), "ci95": [],
                "r_at_eol_ohm": float("nan"), "r0_ohm": float("nan")}

    def _beta(ff, rr):
        b, a = np.polyfit(ff, rr, 1)
        return float(b / a) if a > 0 else float("nan")

    beta = _beta(f, r)
    b_all, a_all = np.polyfit(f, r, 1)
    pred = a_all + b_all * f
    ss_res = float(np.sum((r - pred) ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        j = rng.integers(0, f.size, f.size)
        bb = _beta(f[j], r[j])
        if np.isfinite(bb):
            boots.append(bb)
    boots = np.array(boots)
    return {
        "beta": beta,
        "intercept_ohm": float(a_all),
        "slope_ohm_per_f": float(b_all),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "n_cells": int(f.size),
        "ci95": [float(np.percentile(boots, 2.5)),
                 float(np.percentile(boots, 97.5))] if boots.size else [],
        "r_at_eol_ohm": float(a_all + b_all * 0.2),
        "r0_ohm": float(a_all),
    }


# ----------------------------------------------------------------- 纵向 pkl

@dataclass
class LongCell:
    cell_id: str
    chem: str
    n: np.ndarray = field(default_factory=lambda: np.empty(0))
    fade: np.ndarray = field(default_factory=lambda: np.empty(0))
    r_ohm: np.ndarray = field(default_factory=lambda: np.empty(0))
    r_ratio: np.ndarray = field(default_factory=lambda: np.empty(0))


def _fade_from_capacity(cap: np.ndarray) -> np.ndarray:
    cap = np.asarray(cap, float)
    ok = cap[np.isfinite(cap) & (cap > 0)]
    if ok.size == 0:
        return np.full_like(cap, np.nan)
    k = max(3, int(0.05 * ok.size))
    cap0 = float(np.median(ok[:k]))
    return 1.0 - cap / cap0


def extract_longitudinal_pkl(d: dict, dataset: str) -> LongCell | None:
    """batteryarchive pkl → 逐循环放电起始反转 R → LongCell。"""
    cyc = d.get("cycle_data") or []
    if len(cyc) < 10:
        return None
    ns, fades, rs = [], [], []
    for c in cyc:
        i = np.asarray(c.get("current_in_A"), float).ravel()
        v = np.asarray(c.get("voltage_in_V"), float).ravel()
        if i.size < 5:
            continue
        hits = extract_step_r(i, v)
        r = pick_reference_r(hits, rate_a=abs(float(np.min(i))))
        # 放电起始反转：优先 into_discharge 的最大 |ΔI| 阶跃
        dis = [h for h in hits if h["into_discharge"]]
        if dis:
            best = max(dis, key=lambda h: abs(h["i_to"] - h["i_from"]))
            r = float(best["r_ohm"])
        cap = c.get("discharge_capacity_in_Ah")
        cap_v = float(np.nanmax(np.asarray(cap, float).ravel())) if cap is not None else float("nan")
        n = c.get("cycle_number")
        if not np.isfinite(r) or r <= 0:
            continue
        ns.append(float(n if n is not None else len(ns)))
        rs.append(r)
        fades.append(cap_v)
    if len(ns) < 10:
        return None
    ns = np.array(ns)
    rs = np.array(rs)
    fade = _fade_from_capacity(np.array(fades))
    # 毛刺护栏：R 滚动中位
    rs_s = smooth_series(rs, 7)
    k = max(3, int(0.05 * rs_s.size))
    r0 = float(np.nanmedian(rs_s[:k]))
    with np.errstate(invalid="ignore"):
        ratio = rs_s / r0 if r0 > 0 else np.full_like(rs_s, np.nan)
    meta_chem = str(d.get("cathode_material") or "")
    chem = "NMC" if "NMC" in meta_chem.upper() or "Ni" in meta_chem else dataset
    return LongCell(cell_id=str(d.get("cell_id") or dataset),
                    chem=chem, n=ns, fade=fade,
                    r_ohm=rs_s, r_ratio=ratio)


def load_longitudinal_zip(zip_path: str | Path, dataset: str,
                          cache_csv: str | Path | None = None,
                          max_cells: int = 0) -> list:
    """zip 内 pkl 全量提取（可 CSV 缓存）。"""
    if cache_csv is not None and Path(cache_csv).exists():
        cells = []
        by_id: dict = {}
        with io.open(cache_csv, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["dataset"] != dataset:
                    continue
                cid = row["cell_id"]
                g = by_id.setdefault(cid, LongCell(
                    cell_id=cid, chem=row["chem"]))
                g.n = np.append(g.n, float(row["n"]))
                g.fade = np.append(g.fade, float(row["fade"]))
                g.r_ohm = np.append(g.r_ohm, float(row["r_ohm"]))
                g.r_ratio = np.append(g.r_ratio, float(row["r_ratio"]))
        return [g for g in by_id.values() if g.n.size >= 10]

    cells = []
    rows_out = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".pkl")]
        if max_cells:
            names = names[:max_cells]
        for member in names:
            with zf.open(member) as fh:
                d = pickle.load(fh)
            g = extract_longitudinal_pkl(d, dataset)
            if g is None:
                continue
            cells.append(g)
            for k in range(g.n.size):
                rows_out.append({
                    "dataset": dataset, "cell_id": g.cell_id,
                    "chem": g.chem, "n": float(g.n[k]),
                    "fade": float(g.fade[k]), "r_ohm": float(g.r_ohm[k]),
                    "r_ratio": float(g.r_ratio[k]),
                })
    if cache_csv is not None and rows_out:
        with io.open(cache_csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    return cells


def fit_beta_longitudinal(cells: list) -> dict:
    """逐芯 β（R/R0=1+β·f）→ 体系汇总（中位 + 全距 + 资格计数）。"""
    rows = []
    for g in cells:
        fit = fit_resistance_growth(g.fade, g.r_ratio)
        rows.append({"cell_id": g.cell_id, "chem": g.chem, **fit})
    ok = [r for r in rows if np.isfinite(r.get("beta", np.nan))]
    betas = np.array([r["beta"] for r in ok], float)
    return {
        "beta_median": float(np.median(betas)) if betas.size else float("nan"),
        "beta_range": [float(np.min(betas)), float(np.max(betas))]
                      if betas.size else [],
        "beta_ci95": [float(np.percentile(betas, 2.5)),
                      float(np.percentile(betas, 97.5))]
                     if betas.size >= 5 else [],
        "n_cells_fitted": int(betas.size),
        "n_cells_total": len(cells),
        "cell_rows": rows,
    }
