"""Pure ROS/RViz RM75 Cartesian trajectory follower.

This module deliberately has no dependency on ``Robotic_Arm`` and never opens a
robot connection.  Pinocchio solves the seven-joint kinematics from the RM75
URDF, while ROS publishes the resulting joint states for RViz.
"""

import math
import time
from contextlib import suppress
from pathlib import Path

import numpy as np
import pinocchio as pin
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as PathMessage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


TAU = 2.0 * math.pi
DEFAULT_HOME_JOINTS_RAD = np.array(
    [0.0, 0.4, 0.0, 1.0, 0.0, -0.5, 0.0], dtype=float)


def singularity_adaptive_damping(minimum_singular_value,
                                 activation_sigma=0.03,
                                 base_damping=1e-5,
                                 maximum_lambda=0.08):
    """Return legacy damping normally and a smooth increase near singularity."""
    sigma = float(minimum_singular_value)
    threshold = float(activation_sigma)
    base = float(base_damping)
    maximum = float(maximum_lambda)
    if (not np.isfinite(sigma) or sigma < 0.0 or threshold <= 0.0 or
            base <= 0.0 or maximum <= math.sqrt(base)):
        raise ValueError("invalid adaptive damping parameters")
    if sigma >= threshold:
        return base
    fraction = float(np.clip(1.0 - sigma / threshold, 0.0, 1.0))
    smooth_fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    damping_lambda = (
        math.sqrt(base) + (maximum - math.sqrt(base)) * smooth_fraction)
    return damping_lambda * damping_lambda


def projected_elbow_direction(shoulder, wrist, elbow):
    """Return shoulder-wrist axis, projected elbow direction and radius."""
    shoulder = np.asarray(shoulder, dtype=float)
    wrist = np.asarray(wrist, dtype=float)
    elbow = np.asarray(elbow, dtype=float)
    if any(value.shape != (3,) for value in (shoulder, wrist, elbow)):
        raise ValueError("shoulder, wrist and elbow must be 3-vectors")
    if not all(np.all(np.isfinite(value)) for value in (
            shoulder, wrist, elbow)):
        raise ValueError("shoulder, wrist and elbow must be finite")
    axis_vector = wrist - shoulder
    axis_length = float(np.linalg.norm(axis_vector))
    if axis_length < 1e-6:
        raise ValueError("shoulder-wrist axis is singular")
    axis = axis_vector / axis_length
    projection = shoulder + float((elbow - shoulder) @ axis) * axis
    offset = elbow - projection
    radius = float(np.linalg.norm(offset))
    if radius < 1e-6:
        raise ValueError("elbow lies on the shoulder-wrist axis")
    return axis, offset / radius, radius, projection


def elbow_direction_error_rad(axis, current_direction, desired_direction):
    """Signed desired elbow swivel from current around shoulder-wrist axis."""
    axis = np.asarray(axis, dtype=float)
    current = np.asarray(current_direction, dtype=float)
    desired = np.asarray(desired_direction, dtype=float)
    if any(value.shape != (3,) for value in (axis, current, desired)):
        raise ValueError("axis and elbow directions must be 3-vectors")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-6:
        raise ValueError("elbow axis is singular")
    axis = axis / axis_norm
    current = current - float(current @ axis) * axis
    desired = desired - float(desired @ axis) * axis
    current_norm = float(np.linalg.norm(current))
    desired_norm = float(np.linalg.norm(desired))
    if current_norm < 1e-6 or desired_norm < 1e-6:
        raise ValueError("elbow direction is parallel to its axis")
    current /= current_norm
    desired /= desired_norm
    return math.atan2(
        float(axis @ np.cross(current, desired)),
        float(np.clip(current @ desired, -1.0, 1.0)),
    )


def elbow_nullspace_velocity(elbow_linear_jacobian, null_projector,
                             axis, current_direction, desired_direction,
                             radius_m, weight, max_error_rad):
    """Compute a low-gain elbow correction inside the TCP null space."""
    jacobian = np.asarray(elbow_linear_jacobian, dtype=float)
    projector = np.asarray(null_projector, dtype=float)
    if jacobian.shape != (3, 7) or projector.shape != (7, 7):
        raise ValueError("invalid elbow Jacobian or null projector shape")
    gain = float(weight)
    radius = float(radius_m)
    maximum = float(max_error_rad)
    if gain < 0.0 or radius <= 0.0 or maximum <= 0.0:
        raise ValueError("elbow weight, radius and maximum must be valid")
    error = elbow_direction_error_rad(
        axis, current_direction, desired_direction)
    error = float(np.clip(error, -maximum, maximum))
    tangent = np.cross(
        np.asarray(axis, dtype=float),
        np.asarray(current_direction, dtype=float))
    tangent /= np.linalg.norm(tangent)
    gradient = projector @ jacobian.T @ tangent
    response = float(tangent @ jacobian @ gradient)
    if response <= 1e-9 or gain == 0.0:
        return np.zeros(7), error
    velocity = gain * gradient * (radius * error) / (response + 1e-6)
    return velocity, error


def target_pose_at_phase(home_pose, phase, radius_x, radius_y, radius_z):
    """Return one pose on a smooth, closed 3-D Cartesian loop.

    Phase zero is exactly ``home_pose``, so starting the demo never produces a
    discontinuous Cartesian target.
    """
    phase = float(phase)
    offset = np.array([
        float(radius_x) * (math.cos(phase) - 1.0),
        float(radius_y) * math.sin(phase),
        float(radius_z) * math.sin(2.0 * phase),
    ])
    return pin.SE3(home_pose.rotation.copy(), home_pose.translation + offset)


class RM75Kinematics:
    """Small, state-free RM75 FK/IK wrapper suitable for offline simulation."""

    def __init__(self, urdf_path, frame_name="Link7", joint_margin_rad=0.02):
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        if self.model.nq != 7 or self.model.nv != 7:
            raise ValueError(
                f"RM75 model must have seven joints, got nq={self.model.nq}, "
                f"nv={self.model.nv}")
        if not self.model.existFrame(frame_name):
            raise ValueError(f"end-effector frame {frame_name!r} not in URDF")
        self.frame_id = self.model.getFrameId(frame_name)
        if not self.model.existFrame("Link4"):
            raise ValueError("elbow frame 'Link4' not in URDF")
        self.elbow_frame_id = self.model.getFrameId("Link4")
        self.shoulder_joint_id = self.model.getJointId("joint2")
        if self.shoulder_joint_id == 0:
            raise ValueError("shoulder joint 'joint2' not in URDF")
        self.joint_names = list(self.model.names[1:])
        self.last_minimum_singular_value = float("nan")
        self.last_damping = 1e-5
        self.adaptive_damping_was_active = False
        margin = float(joint_margin_rad)
        self.lower = np.asarray(self.model.lowerPositionLimit) + margin
        self.upper = np.asarray(self.model.upperPositionLimit) - margin
        if np.any(self.lower >= self.upper):
            raise ValueError("joint margin leaves an empty joint range")

    def forward(self, joints_rad):
        joints = np.asarray(joints_rad, dtype=float)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("joint vector must contain seven finite values")
        pin.framesForwardKinematics(self.model, self.data, joints)
        return self.data.oMf[self.frame_id].copy()

    def elbow_geometry(self, joints_rad):
        """Return FK shoulder, wrist, elbow and projected elbow geometry."""
        joints = np.asarray(joints_rad, dtype=float)
        self.forward(joints)
        shoulder = self.data.oMi[self.shoulder_joint_id].translation.copy()
        wrist = self.data.oMf[self.frame_id].translation.copy()
        elbow = self.data.oMf[self.elbow_frame_id].translation.copy()
        axis, direction, radius, projection = projected_elbow_direction(
            shoulder, wrist, elbow)
        return shoulder, wrist, elbow, axis, direction, radius, projection

    def solve(self, target, seed, nominal=None, max_iterations=60,
              position_tolerance_m=2e-5,
              orientation_tolerance_rad=2e-4,
              elbow_direction=None,
              elbow_weight=0.0,
              elbow_deadband_rad=math.radians(2.0),
              max_elbow_error_rad=math.radians(10.0),
              elbow_iterations=2,
              adaptive_singularity_damping=False):
        """Solve a pose while gently resolving redundancy toward ``nominal``."""
        joints = np.asarray(seed, dtype=float).copy()
        nominal = joints.copy() if nominal is None else np.asarray(
            nominal, dtype=float)
        if joints.shape != (7,) or nominal.shape != (7,):
            raise ValueError("seed and nominal must contain seven joints")
        if not np.all(np.isfinite(joints)) or not np.all(np.isfinite(nominal)):
            raise ValueError("seed and nominal must be finite")
        elbow_weight = float(elbow_weight)
        elbow_deadband = float(elbow_deadband_rad)
        max_elbow_error = float(max_elbow_error_rad)
        elbow_iteration_limit = int(elbow_iterations)
        desired_elbow = None
        if elbow_direction is not None:
            desired_elbow = np.asarray(elbow_direction, dtype=float)
            if desired_elbow.shape != (3,) or not np.all(
                    np.isfinite(desired_elbow)):
                raise ValueError("elbow_direction must be a finite 3-vector")
        if (elbow_weight < 0.0 or elbow_deadband < 0.0 or
                max_elbow_error <= 0.0 or elbow_iteration_limit < 0):
            raise ValueError("invalid elbow assistance parameters")
        joints = np.clip(joints, self.lower, self.upper)
        self.last_minimum_singular_value = float("nan")
        self.last_damping = 1e-5
        self.adaptive_damping_was_active = False
        final_error = np.full(6, np.inf)
        elbow_steps = 0
        for iteration in range(int(max_iterations)):
            current = self.forward(joints)
            current_to_target = current.actInv(target)
            final_error = pin.log6(current_to_target).vector
            primary_converged = bool(
                np.linalg.norm(final_error[:3]) <= position_tolerance_m and
                np.linalg.norm(final_error[3:]) <= orientation_tolerance_rad)
            assistance_active = bool(
                primary_converged and desired_elbow is not None and
                elbow_weight > 0.0 and elbow_steps < elbow_iteration_limit)
            if primary_converged and not assistance_active:
                return joints, True, iteration, final_error.copy()

            jacobian = pin.computeFrameJacobian(
                self.model, self.data, joints, self.frame_id,
                pin.ReferenceFrame.LOCAL)
            jacobian = (-pin.Jlog6(current_to_target.inverse()) @ jacobian)

            # Damped pseudo-inverse plus a null-space posture term keeps the
            # redundant RM75 configuration repeatable over successive loops.
            if adaptive_singularity_damping:
                # Preserve the legacy solver exactly outside the true
                # singularity region.  Below sigma=0.03, smoothly increase
                # Levenberg-Marquardt damping up to lambda=0.08.
                minimum_singular_value = float(np.min(
                    np.linalg.svd(jacobian, compute_uv=False)))
                damping = singularity_adaptive_damping(
                    minimum_singular_value)
                self.last_minimum_singular_value = min(
                    self.last_minimum_singular_value,
                    minimum_singular_value) if np.isfinite(
                        self.last_minimum_singular_value) else (
                            minimum_singular_value)
                self.last_damping = max(self.last_damping, damping)
                self.adaptive_damping_was_active |= damping > 1e-5 + 1e-12
            else:
                damping = 1e-5
            inverse = np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), np.eye(6))
            pseudo_inverse = jacobian.T @ inverse
            task_velocity = (
                np.zeros(7) if primary_converged
                else -pseudo_inverse @ final_error)
            null_projector = np.eye(7) - pseudo_inverse @ jacobian
            posture_velocity = null_projector @ (0.08 * (nominal - joints))
            elbow_velocity = np.zeros(7)
            if assistance_active:
                try:
                    _, _, _, axis, direction, radius, _ = (
                        self.elbow_geometry(joints))
                    elbow_error = elbow_direction_error_rad(
                        axis, direction, desired_elbow)
                    if abs(elbow_error) > elbow_deadband:
                        elbow_jacobian = pin.computeFrameJacobian(
                            self.model, self.data, joints,
                            self.elbow_frame_id,
                            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
                        elbow_velocity, _ = elbow_nullspace_velocity(
                            elbow_jacobian, null_projector, axis, direction,
                            desired_elbow, radius, elbow_weight,
                            max_elbow_error)
                        elbow_steps += 1
                    else:
                        return joints, True, iteration, final_error.copy()
                except ValueError:
                    return joints, True, iteration, final_error.copy()
            velocity = task_velocity + posture_velocity + elbow_velocity

            norm = float(np.linalg.norm(velocity))
            if norm > 1.0:
                velocity /= norm
            joints = pin.integrate(self.model, joints, velocity * 0.25)
            joints = np.clip(joints, self.lower, self.upper)

        current = self.forward(joints)
        final_error = pin.log6(current.actInv(target)).vector
        primary_converged = bool(
            np.linalg.norm(final_error[:3]) <= position_tolerance_m and
            np.linalg.norm(final_error[3:]) <= orientation_tolerance_rad)
        return joints, primary_converged, int(max_iterations), final_error.copy()


class RM75OfflineTrajectory(Node):
    """Generate a closed TCP path and publish its IK solution for RViz."""

    def __init__(self):
        super().__init__("rm75_offline_trajectory")
        defaults = {
            "rate_hz": 60.0,
            "loop_period_sec": 12.0,
            "warmup_sec": 2.0,
            "radius_x_m": 0.08,
            "radius_y_m": 0.06,
            "radius_z_m": 0.04,
            "end_effector_frame": "Link7",
            "home_joints_rad": DEFAULT_HOME_JOINTS_RAD.tolist(),
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.loop_period = float(
            self.get_parameter("loop_period_sec").value)
        self.warmup_sec = float(self.get_parameter("warmup_sec").value)
        self.radii = np.array([
            self.get_parameter("radius_x_m").value,
            self.get_parameter("radius_y_m").value,
            self.get_parameter("radius_z_m").value,
        ], dtype=float)
        if self.rate_hz <= 0.0 or self.loop_period <= 0.0:
            raise ValueError("rate_hz and loop_period_sec must be positive")
        if self.warmup_sec < 0.0 or np.any(self.radii < 0.0):
            raise ValueError("warmup and path radii must be non-negative")

        urdf = (Path(get_package_share_directory("rm_description")) /
                "urdf" / "rm_75.urdf")
        self.kinematics = RM75Kinematics(
            urdf, str(self.get_parameter("end_effector_frame").value))
        self.home_joints = np.asarray(
            self.get_parameter("home_joints_rad").value, dtype=float)
        if self.home_joints.shape != (7,):
            raise ValueError("home_joints_rad must contain seven values")
        if (np.any(self.home_joints <= self.kinematics.lower) or
                np.any(self.home_joints >= self.kinematics.upper)):
            raise ValueError("home_joints_rad is outside the RM75 joint limits")
        self.joints = self.home_joints.copy()
        self.home_pose = self.kinematics.forward(self.home_joints)

        self.joint_pub = self.create_publisher(
            JointState, "/rm75_sim/joint_states", 10)
        self.target_pose_pub = self.create_publisher(
            PoseStamped, "/rm75_sim/target_pose", 10)
        self.fk_pose_pub = self.create_publisher(
            PoseStamped, "/rm75_sim/fk_pose", 10)
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.target_path_pub = self.create_publisher(
            PathMessage, "/rm75_sim/target_path", latched_qos)
        self.fk_path_pub = self.create_publisher(
            PathMessage, "/rm75_sim/fk_path", 10)

        self.started_at = time.monotonic()
        self.previous_loop = -1
        self.fk_path = PathMessage()
        self.fk_path.header.frame_id = "base_link"
        self.target_path = self._make_target_path(360)
        self.target_path_pub.publish(self.target_path)
        self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            "OFFLINE ONLY: RM75 3-D closed TCP path -> Pinocchio IK -> RViz; "
            "no robot connection and no motion API")

    @staticmethod
    def _pose_message(stamp, pose):
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = "base_link"
        message.pose.position.x = float(pose.translation[0])
        message.pose.position.y = float(pose.translation[1])
        message.pose.position.z = float(pose.translation[2])
        quaternion = pin.Quaternion(pose.rotation)
        quaternion.normalize()
        coefficients = quaternion.coeffs()
        message.pose.orientation.x = float(coefficients[0])
        message.pose.orientation.y = float(coefficients[1])
        message.pose.orientation.z = float(coefficients[2])
        message.pose.orientation.w = float(coefficients[3])
        return message

    def _make_target_path(self, point_count):
        path = PathMessage()
        path.header.frame_id = "base_link"
        stamp = self.get_clock().now().to_msg()
        path.header.stamp = stamp
        for index in range(int(point_count) + 1):
            phase = TAU * index / int(point_count)
            target = target_pose_at_phase(
                self.home_pose, phase, *self.radii)
            path.poses.append(self._pose_message(stamp, target))
        return path

    def _tick(self):
        elapsed = time.monotonic() - self.started_at
        path_elapsed = max(0.0, elapsed - self.warmup_sec)
        loop_index = int(path_elapsed / self.loop_period)
        phase = (TAU * (path_elapsed % self.loop_period) /
                 self.loop_period if elapsed >= self.warmup_sec else 0.0)
        target = target_pose_at_phase(
            self.home_pose, phase, *self.radii)
        solved, accepted, iterations, error = self.kinematics.solve(
            target, self.joints, nominal=self.home_joints)
        if accepted:
            self.joints = solved
        else:
            self.get_logger().warning(
                "IK rejected; holding last joint state: "
                f"iterations={iterations}, error_norm={np.linalg.norm(error):.6g}",
                throttle_duration_sec=2.0)

        stamp = self.get_clock().now().to_msg()
        fk_pose = self.kinematics.forward(self.joints)
        joint_message = JointState()
        joint_message.header.stamp = stamp
        joint_message.name = self.kinematics.joint_names
        joint_message.position = self.joints.tolist()
        self.joint_pub.publish(joint_message)
        self.target_pose_pub.publish(self._pose_message(stamp, target))
        fk_message = self._pose_message(stamp, fk_pose)
        self.fk_pose_pub.publish(fk_message)

        if loop_index != self.previous_loop:
            self.previous_loop = loop_index
            self.fk_path = PathMessage()
            self.fk_path.header.frame_id = "base_link"
            self.get_logger().info(
                f"starting Cartesian loop {loop_index + 1}; "
                f"period={self.loop_period:.1f}s")
        self.fk_path.header.stamp = stamp
        self.fk_path.poses.append(fk_message)
        max_trace_points = max(2, int(self.rate_hz * self.loop_period) + 2)
        if len(self.fk_path.poses) > max_trace_points:
            self.fk_path.poses = self.fk_path.poses[-max_trace_points:]
        self.fk_path_pub.publish(self.fk_path)
        # Republish the latched target occasionally for RViz configurations
        # that are started after the node.
        if int(elapsed * self.rate_hz) % max(1, int(self.rate_hz * 2.0)) == 0:
            self.target_path.header.stamp = stamp
            self.target_path_pub.publish(self.target_path)


def main(args=None):
    rclpy.init(args=args)
    node = RM75OfflineTrajectory()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        with suppress(KeyboardInterrupt):
            node.destroy_node()
        if rclpy.ok():
            with suppress(KeyboardInterrupt):
                rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
