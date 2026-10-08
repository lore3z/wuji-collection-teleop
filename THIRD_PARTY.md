# Third-party sources

本仓库保留运行所需第三方源码、模型与许可证；各组件按原许可证使用。

| 组件 | 位置 | 许可证/说明 |
| --- | --- | --- |
| Wuji 机器人描述 | `wuji_retargeting/wuji-description/` | 该目录 `LICENSE` |
| Wuji MuJoCo 模型/工具 | `example/utils/mujoco-sim/` | 该目录及内嵌描述包的 `LICENSE` |
| LinkerHand 映射与模型 | `vendor/linkerhand_retarget/` | `vendor/LICENSE`、`vendor/NOTICE`、`vendor/THIRD_PARTY_LICENSES.md`；从本机已使用的上游工程提取运行依赖 |
| LinkerHand ROS 2 SDK | `drivers/linker_hand_ros2_sdk/` | 保留厂商包及原有声明 |
| XRoboToolkit PC-Service / Python binding | `wuji_ros2_src/input_devices/pico_input/vendor/` | 子项目 `LICENSE`、`THIRD_PARTY_NOTICE.txt` |
| PICO 预编译库/应用、RM75 描述 | `wuji_ros2_src/` | 保留原有来源和声明；须遵守厂商的使用条件 |

EMG 冻结控制器由 `runtime/emg_teleop_baseline_v1/manifest.json` 记录来源与哈希。
打包时只调整运行路径和模型路径，不更改捏合、抓握、拇指及硬件映射算法。

手套遥操的生成模板 `runtime/l20_bridge_template.py` 来自本机原有
`l20_wuji_hw_bridge.py.bak_before_v93_120hz`，保留 V6 启动器预期的输入模板；
EMG A/B 使用独立的冻结桥接版本。两者均在上述 manifest 中校验。
