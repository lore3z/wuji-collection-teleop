#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_DIR=/opt/apps/roboticsservice
SERVICE=$SERVICE_DIR/RoboticsServiceProcess

mkdir -p "$PROJECT/.runtime/logs"

echo "========================================"
echo "1. 启动 PICO PC-Service"
echo "========================================"

if pgrep -f "$SERVICE" >/dev/null 2>&1 || pgrep -x RoboticsServiceProcess >/dev/null 2>&1; then
    echo "[PASS] RoboticsService 已运行"
else
    (
        cd "$SERVICE_DIR"
        nohup env \
            -u LD_LIBRARY_PATH \
            LD_LIBRARY_PATH="$SERVICE_DIR" \
            "$SERVICE" \
            > "$PROJECT/.runtime/logs/roboticsservice.log" 2>&1 &
    )

    sleep 2
fi

if ! pgrep -f RoboticsServiceProcess >/dev/null 2>&1; then
    echo "[FAILED] RoboticsService 启动失败"
    cat "$PROJECT/.runtime/logs/roboticsservice.log" 2>/dev/null || true
    exit 1
fi

echo
echo "========================================"
echo "2. 检查 PC-Service 端口"
echo "========================================"

ss -lntp | grep -E '60061|63901' || true

if ! ss -lnt | grep -q ':63901'; then
    echo "[FAILED] PC-Service 没有监听 63901"
    exit 1
fi

echo "[PASS] PC-Service 63901 已监听"

echo
echo "========================================"
echo "3. 等待 PICO USB"
echo "========================================"

adb start-server >/dev/null
adb wait-for-device

adb devices -l

if ! adb get-state 2>/dev/null | grep -q device; then
    echo "[FAILED] PICO ADB 未连接"
    exit 1
fi

echo "[PASS] PICO ADB 已连接"

echo
echo "========================================"
echo "4. 建立 ADB reverse"
echo "========================================"

adb reverse tcp:63901 tcp:63901

adb reverse --list

if ! adb reverse --list | grep -q 'tcp:63901 tcp:63901'; then
    echo "[FAILED] adb reverse 63901 建立失败"
    exit 1
fi

echo
echo "========================================"
echo "PICO USB 链路准备完成"
echo "========================================"
echo
echo "接下来在 PICO 中："
echo "  1. 打开 XRoboToolkit"
echo "  2. Connect 127.0.0.1:63901"
echo "  3. Motion Tracker -> Full Body -> Send"
echo
echo "然后再启动数据采集。"
