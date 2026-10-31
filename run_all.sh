#!/usr/bin/env bash
# A5 全流水线：环境勘察 → 产热 → 冷却 → 梯度/辨识 → 模组 → 寿命 → 安全 → 动态
set -e
PY="${PY:-python}"
cd "$(dirname "$0")"
export PYTHONPATH=src
for s in 00_env_check 01_heat_analysis 02_cooling_sweep 03_gradient_ident \
         04_module 05_aging 06_safety 07_duty; do
  echo "===== $s ====="
  "$PY" "scripts/$s.py"
done
echo "全部完成。指标见 results/metrics/，图见 results/figures/"
