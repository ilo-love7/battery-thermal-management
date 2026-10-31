"""统一出图风格与保存。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体：Windows 优先 Microsoft YaHei / SimHei，找不到就退回默认并告警。
for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "PingFang SC"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f] + plt.rcParams["font.sans-serif"]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

#: 方案配色（定性色板，色盲友好）
SCHEME_COLORS = {
    "natural": "#B45309",
    "forced_air": "#0E7490",
    "cold_plate": "#1D4ED8",
    "immersion": "#047857",
    "pcm": "#7C3AED",
}

FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "figure.autolayout": True,
    })


def save(fig, name: str, out_dir: Path | None = None) -> Path:
    setup_style()
    out = Path(out_dir) if out_dir else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
