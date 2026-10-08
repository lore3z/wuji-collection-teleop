# 项目内硬件驱动

本目录只保存本项目运行所需的**可复制源码**；不依赖开发机的外部工作空间。

| 设备 | 项目内源码 | 构建方式 | 运行入口 |
| --- | --- | --- | --- |
| Gemini2 第三人称相机 | `gemini2_mjpeg_publisher/` | `./setup.sh` | `runtime/run_gemini_main.sh` |
| LinkerHand L20/G20（CAN） | `linker_hand_ros2_sdk/src/linker_hand_ros2_sdk/` | `./setup.sh` | `./l20.sh driver` |
| RealSense D435/D415 第一人称相机 | ROS apt 包 `realsense2_camera` | `./setup.sh --install-system` | `runtime/run_realsense_ego.sh` |
| Wuji 手套 | Python `wuji-sdk` | `./setup.sh` | `./collect.sh` 或 `./l20.sh teleop` |
| Myo + BLED112 | Python `pyomyo` | `./setup.sh` | `./collect.sh` |

`linker_hand_ros2_sdk` 是 L20 遥操与 Sidecar 所需的厂商核心 ROS 包；厂商 GUI 不在
运行链路中，且运行遥操时不应同时启动 GUI（两个程序会竞争同一 CAN 设备）。

构建产物会在本目录下的 `install/` 与项目 `.runtime/` 中生成；这些文件均不进入发布
压缩包。迁移时只复制本项目目录，在目标机运行 `./setup.sh --install-system` 重建即可。
