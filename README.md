# Wuji 采集与遥操

用于 Wuji 手套、多路 EMG、第一/第三人称相机和 PICO Tracker 的数据采集，以及 Wuji/EMG 骨架到 LinkerHand L20/G20 的遥操。另保留 PICO 输入与 RealMan RM75 的 ROS 2 遥操模块。

## 选择运行入口

| 用途 | 入口 | 输出/行为 |
| --- | --- | --- |
| Wuji + Myo + Wavletech + RealSense + Gemini 多模态采集 | `./collect.sh` | FTP-1 Zarr；可选 MP4 |
| PICO 第一人称 + Gemini + 手套/EMG/Tracker 采集 | `./human_collect.sh` | 交互选择任务目录；默认快速保存、不生成 MP4 |
| Wavletech 8 通道 EMG + Wuji 21 点骨架采集 | `./runtime/run_wuji_wavletech_skeleton_collect.sh` | Zarr 和训练兼容的 `calibration.npz` |
| 唯理 WAVELETECH-18 腕带采集 | `./runtime/run_weili18_emg_collect.sh` | 18 通道 EMG + IMU Zarr |
| Wuji 手套控制 L20/G20、并记录机械手侧数据 | `./l20.sh` | CAN/ROS 控制；Sidecar Zarr |
| EMG 模型控制 L20/G20 | `./run_emg_model_hand.sh` | 骨架推理 → A/B 重定向 → ROS 驱动 |
| PICO/RealMan RM75 | `wuji_ros2_src/` | 独立 ROS 2 包，见下文 |

两种唯理/Wavletech 腕带协议不同：8 通道接收器为 921600 baud；18 通道腕带为 2000000 baud。不能互换接收程序。串口由单个采集/遥操进程独占。

## 环境与安装

采集与 ROS 驱动使用 Ubuntu 22.04、Python 3.10、ROS 2 Humble；相机需要 USB 3，机械手需要 SocketCAN。先安装 ROS 2 Humble 并配置其 apt 软件源。

```bash
git clone https://github.com/lore3z/wuji-collection-teleop.git
cd wuji-collection-teleop
cp config/local.env.example config/local.env
# 编辑 config/local.env，填写自己的设备信息。
./setup.sh --install-system
./check.sh --software
```

已有系统依赖时运行 `./setup.sh`。它创建 `.venv/`、安装 `requirements-collector.txt` 与 `pyomyo`，构建项目内 Gemini 和 LinkerHand 驱动。`--install-system` 会调用 sudo 安装系统软件。PICO 显示流模式还需 `adb`、`ffmpeg`：

```bash
sudo apt install adb ffmpeg
```

如果安装器提示缺少 `dialout`/`video` 组权限，按提示加入后注销并重新登录。Wuji SDK 的用户校准、匹配固件和硬件授权需在目标机器上单独准备。

常用本机配置：

```bash
# config/local.env
WUJI_GLOVE_SN=你的手套序列号
WUJI_WAVLETECH_TTY=/dev/serial/by-id/你的接收器
WUJI_MYO_TTY=/dev/serial/by-id/你的Bluegiga接收器
WUJI_MYO_MAC=auto
WUJI_DATA_DIR=/mnt/data/ftp1
# 未连接的 EMG 设备必须关闭，否则采集预检会失败。
WUJI_MYO_ENABLED=0
# WUJI_WAVLETECH_ENABLED=0
```

`config/collector.env` 是共享默认值，`config/local.env` 覆盖本机参数且不会提交。触觉健康 mask 是设备特定文件；诊断自己的手套后再设置 `WUJI_TACTILE_HEALTH_MASK`，不要沿用其他手套的坏点映射。

## 多模态 FTP-1 采集

接好已启用的手套、腕带和相机，执行：

```bash
./collect.sh
# PICO 第一人称模式：
./human_collect.sh
```

`collect.sh` 默认使用 RealSense 第一人称与 Gemini 第三人称。`human_collect.sh` 使用 PICO；当前默认是原生 CameraHandle H.264 双眼流，需要头显端具备相应 CameraHandle 服务的应用。原生视频与 Tracker 的 XRoboToolkit 链路分别准备；PICO 录屏模式仅用于合成显示诊断，可通过 `WUJI_VR_EGO_MODE=screenrecord` 配置，不能当作标定后的原始相机数据。

交互操作：

| 命令 | 含义 |
| --- | --- |
| `b` | 手套悬空，采 2 秒触觉基线 |
| `p` | 确认采集规范 |
| `s` | 按提示填写任务信息并开始 episode |
| `e` | 结束；质量门通过才保存 |
| `d` | 丢弃当前 episode |
| `status` | 查看各路数据状态 |
| `q` | 退出 |

启动器要求 raw tactile、区域统计、point cloud 均满足官方 **526 active taxels** 合同；时间同步、源中断、RGB 和 EMG 连续性也有拒录门。检查失败时修复设备或配置，不能靠插值伪造有效点。细节见 [FTP1_COLLECTION_QUALITY.md](FTP1_COLLECTION_QUALITY.md)。

常规输出在 `WUJI_DATA_DIR`，默认 `data/ftp1/`；`human_collect.sh` 按交互选择将任务数据写入项目 `data/` 下。每条 episode 保留训练对齐数据和 `streams/` 原始数据。

```bash
./.venv/bin/python read_wuji_glove_ftp1_zarr.py /路径/episode.zarr --frame 0
./.venv/bin/python read_wuji_glove_ftp1_zarr.py /路径/数据目录 --latest --play --camera ego
./.venv/bin/python visualize_glove_trajectory.py /路径/数据目录 --latest
```

## EMG 专项采集

Wavletech 8 通道与 Wuji 骨架同步采集，无需相机：

```bash
./runtime/run_wuji_wavletech_skeleton_collect.sh
# 定时采一条，避免交互：
./runtime/run_wuji_wavletech_skeleton_collect.sh --duration-seconds 10 --instruction pinch
```

默认输出 `data/wavletech_skeleton/episode_XXXXXX.zarr`。交互命令为 `s/e/d/status/q`。`data/` 为骨架时间轴上对齐的 EMG，`streams/` 保留原始 2000 Hz EMG、骨架、时间戳、序号及 IMU；episode 内的 `calibration.npz` 可交给现有 EMG 骨架训练流程。

唯理 WAVELETECH-18（18 通道，2000 Hz EMG、208 Hz IMU）：

```bash
./runtime/run_weili18_emg_live.sh --tty /dev/serial/by-id/你的18通道腕带
./runtime/run_weili18_emg_collect.sh --tty /dev/serial/by-id/你的18通道腕带 --duration-seconds 60
```

默认写入 `data/weili18_emg/episode_时间.zarr`，用 `--output /路径/文件.zarr` 改位置，Ctrl+C 保存已收到的数据。默认解析设备滤波开启的包；只有设备端已停止采集、关闭滤波、重新启动后才使用 `--packet-format raw`。Python 订阅接口位于 `runtime/weili18_emg.py`，波形显示与录制不能同时占用串口。

## Wuji → L20/G20 遥操

先准备手套 SDK 右手标定模型。在 `config/local.env` 中指定：

```bash
WUJI_HUMAN_URDF=/绝对路径/当前Wuji用户/models/right_hand.urdf
```

只找到一个 SDK 用户的右手模型时程序可自动选择；有多个用户时必须明确指定。不要把个人校准文件提交到仓库。

重定向额外依赖 MuJoCo、Pinocchio、NLopt 等，使用独立环境：

```bash
python3 -m venv .venv-teleop
./.venv-teleop/bin/pip install -r requirements-teleop.txt
```

先准备 CAN，并在终端 1 启动驱动：

```bash
./l20.sh can
./l20.sh driver
```

终端 2 启动手套遥操（默认使用 `.venv-teleop`；可用 `EMG_GEORT_PYTHON` 覆盖）：

```bash
./l20.sh teleop
```

终端 3 记录机械手侧数据：

```bash
./l20.sh record
```

默认文件为 `data/l20/l20_teleop_pressure_120hz.zarr`。120 Hz 是写入轴，源关节/压力的真实频率由时间戳与序号判断。遥操启动器默认控制参数为 30 Hz，可以显式传 `--control-hz` 覆盖。首次运行按提示完成 OPEN/FIST 标定及接管。

```bash
./.venv/bin/python read_l20_pressure_zarr.py data/l20/l20_teleop_pressure_120hz.zarr
```

CAN 重置前停止驱动及遥操。运行实体控制时关闭厂商 GUI，保持只有一份控制链路。Ctrl+C 停止当前入口拥有的进程。

## EMG 模型 → L20/G20 遥操

本仓库包含 EMG 数据接收、骨架适配、安全门、A/B 控制器和桥接模块。**训练工程、模型权重及其推理服务不在本仓库内**，需单独安装对应 EMG2Pose 工程，并把 `--model` 指向该工程里的绝对 checkpoint/模型目录；其服务与 Python 环境由模型解析器识别。分类器 checkpoint 不能代替 21×3 骨架模型。

安装上面的 `.venv-teleop`，保持终端 1 的 `./l20.sh driver` 运行。首次先验证实时骨架链路：

```bash
./run_wavletech_emg_model_hand.sh --model /绝对路径/模型工程/weights/finetune.pt --mode b --dry-run
```

确认是原生 Wavletech 2000 Hz 模型后启动实体控制：

```bash
./run_wavletech_emg_model_hand.sh \
  --model /绝对路径/模型工程/weights/finetune.pt \
  --mode b \
  --confirm-wavletech-model
```

Myo 输入使用对应的 Myo 模型：

```bash
./run_emg_model_hand.sh --model /绝对路径/Myo模型目录 --mode b --dry-run
./run_emg_model_hand.sh --model /绝对路径/Myo模型目录 --mode b
```

Myo 实体遥操在检查通过后按回车授权；Wavletech 的 `--confirm-wavletech-model` 直接授权控制。`--mode a` 是捏合辅助，`--mode b` 是自然动作。`--check-only` 检查文件、模型加载和端口；`--dry-run` 验证新骨架包，均不启动机械手控制器。控制器默认 Python 为 `.venv-teleop/bin/python`，可用 `EMG_GEORT_PYTHON` 覆盖。

Myo 模型输入为原始 8 通道约 200 Hz EMG；Wavletech 原生模型输入约 2000 Hz，两者不能互换。纯 EMG 遥操无需连接手套。日志保存在 `runtime/emg_model_hand_runs/` 和 `runtime/emg_teleop_sessions/`。骨架过期、会话变化或源中断会禁止输出，需要重新启动并授权。更多接口约定见 [EMG_MODEL_HAND.md](EMG_MODEL_HAND.md)。

## PICO Tracker 与 RM75 ROS 2 模块

`wuji_ros2_src/input_devices/pico_input/` 包含 Tracker 发布器、SDK 安装器、配套原生库及源码。主采集入口可使用本机 ROS overlay，也兼容已存在的 `wuji-hand-teleop` 容器。新机器优先构建本机包：

```bash
bash wuji_ros2_src/input_devices/pico_input/install_sdk.sh
source /opt/ros/humble/setup.bash
colcon --log-base .runtime/pico_ws/log build \
  --base-paths wuji_ros2_src/input_devices/pico_input \
  --build-base .runtime/pico_ws/build \
  --install-base .runtime/pico_ws/install
```

在 `config/local.env` 设置 `WUJI_PICO_SETUP_BASH=.runtime/pico_ws/install/setup.bash`、`WUJI_PICO_DOCKER_ENABLED=0` 和自己的两个 Tracker 序列号。PC-Service 需另行安装并监听 63901；安装后运行 `./runtime/start_pico_usb.sh` 建立 ADB reverse，再在头显连接 `127.0.0.1:63901`。原生相机应用仍需按其自身网络配置连接。

RM75 需要独立的 Pinocchio 3+ / `coal` 和官方 `Robotic_Arm` SDK 环境，不能直接复用上述 Pinocchio 2.7 的 L20 依赖清单。RM75 包和机器人描述位于 `wuji_ros2_src/output_devices/realman_rm75_output/` 与 `wuji_ros2_src/rm_description/`，按各模块文档配置网络、坐标映射及安全门，再用 colcon 构建。该链路属于独立硬件工作流，不由 `collect.sh` 自动启动。Tracker 直接发布器不依赖 Tianji；使用 `pico_input_node` 或 `pico_right_arm_input_node` 时还需另行安装上游 `tianji_world_output`，其中共享的配置/坐标转换库未随当前目录提供。PICO 的预编译 SDK 为 Linux x86_64/Python 3.10；其他平台按该包的 `install_sdk.sh --build` 从源码构建。

## 目录与开发验证

```text
config/                         共享默认值、本机配置模板
runtime/                        数据接收、启动编排、诊断、离线测试
runtime/emg_teleop_baseline_v1/  有 SHA256 校验的 A/B 控制器与标定参考
runtime/backups/                两个仍参与哈希审计的稳定原始版本
wuji_retargeting/                手套骨架重定向与机器人描述
example/                        重定向配置、输入设备、仿真工具
vendor/                         L20/G20 桥接实际使用的厂商映射代码与模型
wuji_ros2_src/                   PICO 与 RM75 ROS 2 源码
drivers/                        Gemini、LinkerHand ROS 驱动源码
data/                           本地采集结果，不提交
```

离线验证，不移动实体设备：

```bash
./.venv/bin/pip install pytest
./.venv/bin/python -m pytest runtime -q
./.venv/bin/python skeleton_teleop_MODE_A_emg.py --verify-only
./.venv/bin/python skeleton_teleop_MODE_B_emg.py --verify-only
```

本机配置、个人校准、录制数据、模型权重、日志、虚拟环境和 ROS 构建产物由 `.gitignore` 排除。迁移到新机器后重新安装/构建并录一条短 episode 验证硬件。项目中的第三方代码保留各自许可证，参见 [THIRD_PARTY.md](THIRD_PARTY.md)。
