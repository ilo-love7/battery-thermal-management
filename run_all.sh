#!/usr/bin/env bash
# A5 全流水线：环境勘察 → 产热 → 冷却 → 梯度/辨识 → 模组 → 寿命 → 安全 → 动态
#              → v1.1 深化（10–13）→ v1.2 深化（14–17）→ v1.3 深化（18–21）
#              → v1.4 深化（22–25）→ v1.5 深化（26–28）→ v1.6 深化（29–31）
set -e
PY="${PY:-python}"
cd "$(dirname "$0")"
export PYTHONPATH=src
for s in 00_env_check 01_heat_analysis 02_cooling_sweep 03_gradient_ident \
         04_module 05_aging 06_safety 07_duty \
         10_calib_real 11_uncertainty 12_pcm_plateau 13_validity_contact \
         14_core_observer 15_life_uncertainty 16_immersion_boiling 17_pad_contact \
         18_aging_feedback 19_life_validation 20_q_correction 21_thermal_control \
         22_actuator_ident 23_calendar_cycle 24_module_control 25_lowtemp_validity \
         26_snl_calendar 27_lowtemp_rct 28_highrate_scaling \
         29_rate_effect 30_module_life 31_5c_band; do
  echo "===== $s ====="
  "$PY" "scripts/$s.py"
done
echo "全部完成。指标见 results/metrics/，图见 results/figures/"
