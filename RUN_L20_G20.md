# L20/G20 运行入口

完整安装、手套标定、CAN 准备、遥操、Sidecar 录制和 EMG 遥操说明见
[README.md](README.md)。

入口为 `./l20.sh can|driver|teleop|record|all|check`。实体驱动来源于项目内
`drivers/linker_hand_ros2_sdk`，重定向依赖使用独立 `.venv-teleop` 环境。
手套遥操默认 30 Hz；Sidecar 写入轴为 120 Hz，不代表所有传感器的真实采样率。

错误排查：CAN 被占用时关闭旧遥操及厂商 GUI；缺少 ROS 库时使用入口脚本加载
ROS 2 Humble；缺少驱动构建产物时运行 `./setup.sh`。CAN 重置须在控制进程停止后执行。
