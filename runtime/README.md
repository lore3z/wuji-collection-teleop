# Runtime 工具

完整操作说明见项目根目录 [README.md](../README.md)。

| 功能 | 工具 |
| --- | --- |
| FTP-1 编排 | `collect_all.sh`、`run_wuji_ftp1_collect.sh` |
| Wavletech 8 通道 EMG/IMU | `wavletech_serial_stream.py`、`run_wavletech_emg_live.sh` |
| 唯理 18 通道 EMG/IMU | `weili18_emg.py`、`run_weili18_emg_collect.sh`、`run_weili18_emg_live.sh` |
| EMG + Wuji 骨架录制 | `run_wuji_wavletech_skeleton_collect.sh` |
| Myo/BLED112 | `myo_200hz_stream.py`、`myo_transport.py`、`run_myo_emg_live.sh` |
| 触觉合同/诊断 | `wuji_tactile_contract_check.py`、`run_wuji_tactile_health.sh` |
| PICO 视频/Tracker | `pico_raw_camera_publisher.py`、`pico_vr_ego_publisher.py`、`start_pico_usb.sh` |
| L20/G20 | `run_l20_driver.sh`、`run_l20_teleop.sh`、`run_l20_sidecar_collect.sh` |
| EMG 模型解析 | `emg_model_runtime.py` |
| 冻结 A/B 控制链路 | `emg_teleop_baseline_v1/`、`project_paths.py`、`l20_wuji_hw_bridge.py` |

`test_*.py` 包含离线验证；`test_thumb_root_axis.py` 是手动硬件诊断入口，
不会作为 pytest 测试运行，禁止为验证仓库而直接执行它。

输出与进程日志在 `.runtime/`、`emg_model_hand_runs/`、`emg_teleop_sessions/`
等被 Git 排除的目录。`backups/` 仅保留两组仍受哈希审计的原始稳定控制器。
