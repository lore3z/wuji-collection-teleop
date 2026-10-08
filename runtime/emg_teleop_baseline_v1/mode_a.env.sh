#!/usr/bin/env bash

cd ~/wuji_ftp1_collection_v8\ \(1\)

conda activate geort
source /opt/ros/humble/setup.bash
source ~/linkerhand-telop-ros2/install/setup.bash

export V94_REAL_THUMB_ROLL_BIAS=0.0
export V94_REAL_THUMB_YAW_BIAS=0.0
export V94_REAL_THUMB_PITCH_BIAS=0.0
export V94_REAL_THUMB_MCP_BIAS=0.0

export V94_PINCH_DETECT_ON_MM=90
export V94_PINCH_LOCK_DWELL_MS=120
export V94_PINCH_DOMINANCE_MM=4

export V94_PINCH_PREP_ON_MM=70
export V94_PINCH_COMMIT_ON_MM=55
export V94_PINCH_ROOT_ONLY_ON_MM=45
export V94_PINCH_CLOSE_DONE_MM=25
export V94_PINCH_RELEASE_MM=100

export V94_PINCH_ANTICIPATE_GAIN=0.15
export V94_PINCH_PREP_GAIN=0.65
export V94_PINCH_ROOT_PREBEND=0.20
export V94_PINCH_ASSIST_SLEW_S=0.18

python -u skeleton_teleop_v94_hw.py --hz 120 --arm
