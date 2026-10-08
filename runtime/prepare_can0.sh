#!/usr/bin/env bash
# Reinitialize CAN only when no LinkerHand driver, GUI, or teleop process owns it.
set -euo pipefail

sudo /usr/sbin/ip link set can0 down || true
sudo /usr/sbin/ip link set can0 up type can bitrate 1000000
ip -details link show can0
