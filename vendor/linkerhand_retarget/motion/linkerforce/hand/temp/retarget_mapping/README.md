# retarget_mapping

独立的手套源数据到目标 URDF 的对照映射运行时。

这个目录不启动 UDP、不连接 CAN/CANFD、不依赖 WebUI。它只做三件事：

1. 从 `calibration/glove_calibration.json` 读取原始手套样本。
2. 用 `profiles.py` 里的配置建立 `source_feature -> target_urdf_joint` 对照表。
3. 对新输入的手套源数据插值出目标 `urdf_joints`，并可选输出设备 `pose` 和电机值。

## 使用

```python
from pathlib import Path
from retarget_mapping import simulate_saved_pose, map_glove_values

result = simulate_saved_pose(
    project_root=Path(r"D:\project\python\linkerhand_telop_sdk\haocun"),
    profile="o20-right",
    pose_name="pinch_index_thumb",
)

print(result["source_features"])
print(result["correspondence"])
print(result["urdf_joints"])
print(result["device_pose"])
print(result["motor_values"])

live_result = map_glove_values(
    project_root=Path(r"D:\project\python\linkerhand_telop_sdk\haocun"),
    profile="o20-right",
    glove_values=result["glove_values"],
)
```

## 返回字段

- `source_features`: 从手套坐标提取出的源特征，例如 `index.root_flexion_yz`。
- `correspondence`: 每个目标关节的锚点对照表，包含源特征值和目标 URDF 标定值。
- `urdf_joints`: 根据对照表插值得到的目标 URDF 弧度值。
- `device_pose`: 目标设备协议 pose。O20 是 20 位 `0~255` pose。
- `motor_values`: 最终电机值。O20 是 CANFD 17 路电机值。

## 当前支持

- `o20-right`: 完整支持 `glove_raw.values -> urdf_joints -> O20 20 位 pose -> 17 路 CANFD motor_values`。

后续扩展其他手型时，在 `profiles.py` 增加 profile，并在 `simulator.py` 增加对应的 `device_backend` 转换即可。
