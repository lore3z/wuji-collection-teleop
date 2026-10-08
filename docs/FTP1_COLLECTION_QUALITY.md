# FTP-1 采集质量门槛与人工检查

采集器已经自动处理的项目：

- 写入 FTP-1 的 `data/` 与 `meta/episode_ends` contract；`audit/` 保存原始 source。
- 以第一人称相机发布的每张真实帧作为 canonical 时间轴（RealSense 约 60 Hz；PICO VR 从约 90 Hz 源流选择真实帧并稳定发布 60 Hz）；不补帧、不复用 RGB/Gemini 图像。
- 去掉按 `s` 前已经进入 callback 队列的边界帧。
- `matrix` tactile type、`zstd` 压缩、状态/触觉 256-frame chunk、RGB 单帧 chunk。
- 全手 24×31 matrix 使用 whole-hand area=5；另外导出 Wuji 官方 thumb/index/middle/ring/pinky/palm 六区 state group，避免把整手误标为拇指指尖。
- 每次 `b` 用无接触 2 秒基线扣除手套底噪；训练矩阵中的无效 taxel 为 0，原始 `-1` 与 mask 在 `audit/`。
- 任务描述不能为空；每一帧保存实际任务文本，不能再保存 `a` 这类占位符。
- FTP-1 canonical hand geometry 只处理角度跨 ±π 的表达跳变；异常单帧关节跳变直接拒绝。
- 每条成功 episode 自动把两路 RGB 保存为 `videos/episode_xxxxxx_ego.mp4` 与
  `videos/episode_xxxxxx_main.mp4`；MP4 导出失败会明确打印 `FAILED`，不会伪称视频已保存。
- 默认检测并模糊两路 RGB 中 Haar 检出的正面人脸；这不是对侧脸、屏幕和漏检的隐私保证，采集机位仍必须避免包含无关人员与敏感屏幕。
- 第一人称 RGB 在落盘前检查清晰度 5% 分位；低于阈值的 episode 拒绝保存。RealSense runtime 强制短曝光；PICO 源是按配置选择左右眼并应用 policy ROI 的 VR 合成画面，默认排除 XRoboToolkit UI 与圆形黑边，不声称具有原始相机内参。
- 采集前和 `b` 时强制检查 Wuji 24×31 完整矩阵和六区总数均为官方 **526 active taxels**；534/526 混用会直接拒绝，不会猜测那 8 个坐标或插值 row 10。
- 原有 Myo/BLED112 链路继续写入 `data/right_forearm_emg` 和 `streams/right_forearm_emg_raw`；新增 Wavletech 链路独立写入 `data/right_forearm_emg_wavletech`、`streams/right_forearm_emg_wavletech_raw` 与 `streams/right_forearm_imu_wavletech_raw`。两路都须在按 `s` 前连续稳定 3 秒；任一路在录制中断流、重连或 sequence 不连续都会终止并丢弃整条 episode。所有原生样本、sequence 和到达时间均保留，不补零、不伪造样本。
- PICO ego 偶发调度抖动允许单次 RGB gap 最大 50 ms，但整条时间轴缺帧比例仍必须不超过 3%。非图像 source 仍使用原 age 门槛；只有坏行不超过 3%且最长连续不超过 3 行时才删除该 canonical 行，并在 metadata 记录 drop 数与最长连续值；最终时间轴仍必须通过同一 RGB gap/missing 门槛。

每次采集前：

1. 执行 `python3 runtime/wuji_tactile_contract_check.py` 并确认 `[PASS]`；再启动第一人称源（human 模式为 PICO VR）、第三人称 Gemini、采集器；输入 `status` 确认全部刷新。
2. 穿好手套后输入 `b`，两秒内不要接触桌面、衣服或物体。
3. 输入 `p` 并完成确认：第二人操作录制；示范者双手均参与任务；无无关人脸/屏幕隐私；第三人称对准工作区。
4. 确认第一人称能看清手、工作区和被操纵物；第三人称必须能看到完整手套与任务对象。
5. 使用具体任务描述，例如 `pick up the white bottle and place it upright`，不要使用 `a`、`test` 或空文本。
5. 每条完成后运行：

   ```bash
   cd ~/wuji_datasets/wuji_glove_realsense_ftp1
   python3 src/read_wuji_glove_ftp1_zarr.py --latest --strict-quality --frame 1
   python3 src/read_wuji_glove_ftp1_zarr.py --latest --play --camera ego
   python3 src/read_wuji_glove_ftp1_zarr.py --latest --play --camera main
   ```

自动质量门槛拒绝时，不应修改 Zarr 强行保留；先处理相机/网络/光照后重录。

仍需人工或厂商处理，软件不能假装解决的项目：

- D435/Gemini USB、曝光、对焦、视野和物理固定；若有连续帧缺失，检查 USB 3 直连、带宽与供电，必要时降低其他 USB 设备负载。
- PICO 双 Tracker 已作为独立真实位姿源保存：tracker0 写入 `camera_tracker_pose`（z 加 0.10 m），tracker1 写入 `right_wrist_tracker_pose`，两者均为 `[x,y,z,roll,pitch,yaw]`、米/弧度；canonical 行只引用最近的真实位姿，不做数值插值。当 Tracker 频率略低时允许相邻行引用同一 source，原始流、source seq/时间戳与 age 全部保留供审计；不要与 Gemini IMU 的相对姿态混用。
- Wuji skeleton 由手套角度 FK 派生，不是独立光学骨架测量；解剖角在 `audit/wuji_hand_joints_anatomical` 中保留，不能声称是已校准的真实人体关节角。
- 厂商指定的无效 taxel 行、固定饱和 taxel、固件关节夹角、拇指串扰，必须通过空手/按压健康检查和厂商固件确认。不要把单个异常 taxel 当作可靠接触信号。
- 当前已实测 WG1K v0.11.1 的 24×31 / zones 是 534 点、点云是 526 点。这是固件与 SDK mapping 不一致，不能由采集代码修复。必须由 Wuji 提供并按其流程升级匹配固件；升级后对 row 6 / row 10 做实际按压健康检查。
- 当前手套压力是 SDK 标定 raw 值而不是牛顿；若需要力闭环或论文中的 N，需要额外砝码/力传感器标定。
- 采集含其他人时，必须取得同意；避免拍摄屏幕上的私人内容、姓名、聊天记录或人脸。需要公开数据时先离线模糊/裁剪。
