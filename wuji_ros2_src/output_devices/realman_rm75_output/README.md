# RealMan RM75 independent dry-run mapper

## Pure offline ROS/RViz Cartesian trajectory

The offline demo needs neither a RealMan controller nor PICO input. It loads
the RM75 URDF into Pinocchio, generates a smooth closed 3-D TCP path, solves
seven-axis IK continuously, and displays both the complete target path and the
actual FK trace in RViz:

```bash
cd /home/wuji/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select rm_description realman_rm75_output --symlink-install
source install/setup.bash
ros2 launch realman_rm75_output rm75_offline_rviz_trajectory.launch.py
```

The default path takes 12 seconds per loop after a two-second warmup. Its size
and speed can be changed without editing code:

```bash
ros2 launch realman_rm75_output rm75_offline_rviz_trajectory.launch.py \
  loop_period_sec:=8.0 radius_x_m:=0.06 radius_y_m:=0.05 radius_z_m:=0.03
```

Published topics:

```text
/rm75_sim/joint_states  sensor_msgs/msg/JointState
/rm75_sim/target_pose   geometry_msgs/msg/PoseStamped
/rm75_sim/fk_pose       geometry_msgs/msg/PoseStamped
/rm75_sim/target_path   nav_msgs/msg/Path
/rm75_sim/fk_path       nav_msgs/msg/Path
```

This launch contains no import of `Robotic_Arm`, no network connection, and no
robot motion backend.

## External operator pose input

For a generic ROS controller, launch the absolute-pose follower:

```bash
ros2 launch realman_rm75_output rm75_external_pose_rviz.launch.py
```

Publish `geometry_msgs/msg/PoseStamped` to
`/rm75_sim/target_pose_cmd`. Position is in metres in `base_link`; orientation
is the ROS quaternion order `[x, y, z, w]`. The quaternion is normalized by the
follower, but must not be all zero.

```bash
ros2 topic pub --rate 30 /rm75_sim/target_pose_cmd \
  geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: base_link}, pose: {position: {x: 0.38943, y: 0.02, z: 0.62150}, orientation: {x: 0.0, y: 0.43497, z: 0.0, w: 0.90045}}}"
```

For direct PICO input, use the single-wrist relative-pose launch. The first
valid wrist sample is automatically rebased onto the RM75 home pose, so only
subsequent wrist deltas move the simulated arm. If multiple Motion Trackers
are paired, pass the serial number printed by PC-Service:

```bash
ros2 launch realman_rm75_output rm75_pico_rviz_teleop.launch.py \
  tracker_serial:=PC2310XXXXXXXXXXXX
```

The minimal PICO node publishes `/pico/right_wrist/raw_pose` in
`pico_tracking`. This launch applies the OpenXR-to-RM-base axis rotation to
both translation and rotation. Rebase at the current simulated pose at any
time with:

```bash
ros2 service call /rm75_sim/rebase std_srvs/srv/Trigger '{}'
```

If input stops, contains a discontinuity, or IK cannot reach the target, the
RViz arm holds its last accepted pose. These launches remain pure offline
simulation and never connect to a RealMan controller.

This package subscribes to WUJI's incremental right-arm target:

```text
/right_arm_target_pose  geometry_msgs/msg/PoseStamped
frame_id: world_right
position: metres
orientation: ROS [x,y,z,w]
```

It records the first valid WUJI target and the RM75 pose read at startup, maps
only the target's relative change into the RM75 base frame, runs RealMan ordinary
inverse kinematics using the previous accepted seven-joint solution as its
reference, observes the resulting RM75 arm angle, and verifies the solution by
forward kinematics. Accepted targets are published only as diagnostics on
`/rm75/dry_run_target_pose`.

## Safety contract

- `dry_run` must be `true` and `enable_motion` must be `false`; otherwise the
  node refuses to start.
- The adapter exposes only connection, state query, conversion, ordinary IK,
  FK, arm-angle observation, algorithm joint-limit reads, and disconnect.
- IK or validation failure rejects only that input frame.
- No robot command path exists in this package.

## Coordinate calibration warning

`axis_mapping` means **WUJI right-chest frame to the installed RM75 base frame**.
Its identity default is an uncalibrated placeholder, not a claim about physical
axis alignment. All six translation directions and all three positive rotation
axes require physical verification before any motion-stage design.

The mapping is:

```text
p_target = p_R0 + scale * A * (p_P - p_P0)
R_target = A * (R_P * inverse(R_P0)) * inverse(A) * R_R0
```

## Build, test, and run

```bash
docker exec -it wuji-hand-teleop bash
cd /home/wuji/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select pico_input realman_rm75_output --symlink-install
source install/setup.bash
```

```bash
cd /home/wuji/ros2_ws/src/output_devices/realman_rm75_output
PYTHONPATH=$PWD python3 -m pytest -q
```

After both hardware data sources are deliberately made available:

```bash
ros2 launch pico_input pico_right_arm.launch.py
ros2 launch realman_rm75_output realman_rm75_output.launch.py
```
