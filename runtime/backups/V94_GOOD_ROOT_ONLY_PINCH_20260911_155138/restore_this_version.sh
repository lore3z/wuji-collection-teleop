#!/usr/bin/env bash
set -e

BACKUP="/home/lzt/wuji_ftp1_collection_v8 (1)/runtime/backups/V94_GOOD_ROOT_ONLY_PINCH_20260911_155138"

cp -a   "$BACKUP/skeleton_teleop_v94_hw.py"   "/home/lzt/wuji_ftp1_collection_v8 (1)/skeleton_teleop_v94_hw.py"

cp -a   "$BACKUP/l20_wuji_hw_bridge.py"   "/home/lzt/linkerhand-telop-ros2/tools/l20_wuji_hw_bridge.py"

echo "RESTORED:"
echo "/home/lzt/wuji_ftp1_collection_v8 (1)/skeleton_teleop_v94_hw.py"
echo "/home/lzt/linkerhand-telop-ros2/tools/l20_wuji_hw_bridge.py"

python3 -m py_compile "/home/lzt/wuji_ftp1_collection_v8 (1)/skeleton_teleop_v94_hw.py"
python3 -m py_compile "/home/lzt/linkerhand-telop-ros2/tools/l20_wuji_hw_bridge.py"

echo "RESTORE OK"
