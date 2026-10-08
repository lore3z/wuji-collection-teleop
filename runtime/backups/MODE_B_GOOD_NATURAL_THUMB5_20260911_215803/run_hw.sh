#!/usr/bin/env bash
set -e

cd ~/wuji_ftp1_collection_v8\ \(1\)

conda activate geort
source /opt/ros/humble/setup.bash
source ~/linkerhand-telop-ros2/install/setup.bash

# 四指抓握
export V10_GRASP_PREEMPT=0.12
export V10_GRASP_ENTER=0.25
export V10_GRASP_EXIT=0.05
export V10_GRASP_FULL=1.00
export V10_GRASP_FINGER_BEND_RAD=0.36
export V10_GRASP_DWELL_MS=40
export V10_GRASP_RELEASE_MS=140
export V10_GRASP_TAU_S=0.08
export V10_GRASP_ASSIST_TAU_S=0.10
export V10_GRASP_FINGER_ASSIST=0.95
export V10_GRASP_ROOT_MAX=1.33
export V10_GRASP_PIP_MAX=1.75

# grasp不碰拇指
export V10_GRASP_THUMB_ASSIST=0.0

# 拇指尖端：IP模型
export V116_MCP_EXPAND=1.00
export V116_IP_EXPAND=1.20
export V116_THUMB_TAU_S=0.045
export V116_MCP_BLEND=0.0
export V116_IP_BLEND=1.00
export V116_THUMB_TIP_MAX=1.25

# 拇指根部：adduction模型
export V118_ROOT_EXPAND=1.10
export V118_ROOT_TAU_S=0.055
export V118_ROOT_BLEND=1.00
export V118_ROOT_REVERSE=0
export V118_ROOT_Q14_MIN=0.00
export V118_ROOT_Q14_MAX=0.83

# 实体bias
export V94_REAL_THUMB_ROLL_BIAS=0.0
export V94_REAL_THUMB_YAW_BIAS=0.0
export V94_REAL_THUMB_PITCH_BIAS=0.0
export V94_REAL_THUMB_MCP_BIAS=0.0

python -u \
  skeleton_teleop_MODE_B_thumb5.py \
  --hz 120 \
  --arm
