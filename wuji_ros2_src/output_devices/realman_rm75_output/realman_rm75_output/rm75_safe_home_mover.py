"""Model-validated, checkpointed low-speed move to the RM75 shadow home."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import coal
import numpy as np
import pinocchio as pin
from ament_index_python.packages import get_package_share_directory
from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

from .movej_backend import RealPlannedMoveJBackend


# J7=-350 is within 10 degrees of the shadow's equivalent zero orientation,
# avoids a longer +212 degree turn, and retains 10 degrees from the SDK limit.
SAFE_REAL_HOME_DEG = np.array([
    -90.0, 22.9183118, 0.0, 85.9436693, 0.0, -18.862933, -350.0])


def joint_vector(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError("joint vector must contain seven finite values")
    return result


def generate_linear_waypoints(start_deg, goal_deg, max_segment_deg=25.0):
    start = joint_vector(start_deg)
    goal = joint_vector(goal_deg)
    maximum_delta = float(np.max(np.abs(goal - start)))
    segments = max(1, int(math.ceil(maximum_delta / float(max_segment_deg))))
    return [
        start + (index / segments) * (goal - start)
        for index in range(1, segments + 1)
    ]


class CollisionValidator:
    """RM75 mesh self-collision and ground-clearance checker."""

    def __init__(self, urdf_path, min_self_clearance_m=0.04,
                 min_ground_clearance_m=0.10):
        self.urdf_path = Path(urdf_path)
        self.min_self = float(min_self_clearance_m)
        self.min_ground = float(min_ground_clearance_m)
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.geometry = pin.buildGeomFromUrdf(
            self.model, str(self.urdf_path), pin.GeometryType.COLLISION,
            [str(self.urdf_path.parents[2])])
        self.geometry.addAllCollisionPairs()
        kept = []
        for pair in self.geometry.collisionPairs:
            first = self.geometry.geometryObjects[pair.first]
            second = self.geometry.geometryObjects[pair.second]
            if abs(int(first.parentJoint) - int(second.parentJoint)) > 1:
                kept.append(pair)
        self.geometry.removeAllCollisionPairs()
        for pair in kept:
            self.geometry.addCollisionPair(pair)
        self.data = self.model.createData()
        self.geometry_data = pin.GeometryData(self.geometry)
        self.ground = coal.Plane(np.array([0.0, 0.0, 1.0]), 0.0)
        self.ground_transform = coal.Transform3s()

    def clearance(self, joints_deg):
        q = np.radians(joint_vector(joints_deg))
        pin.computeDistances(
            self.model, self.data, self.geometry, self.geometry_data, q)
        self_clearance = min(
            float(result.min_distance)
            for result in self.geometry_data.distanceResults)
        pin.updateGeometryPlacements(
            self.model, self.data, self.geometry, self.geometry_data, q)
        ground_clearance = float("inf")
        for index, obj in enumerate(self.geometry.geometryObjects[1:], 1):
            placement = self.geometry_data.oMg[index]
            transform = coal.Transform3s(
                placement.rotation, placement.translation)
            result = coal.DistanceResult()
            distance = coal.distance(
                obj.geometry, transform, self.ground, self.ground_transform,
                coal.DistanceRequest(), result)
            ground_clearance = min(ground_clearance, float(distance))
        return self_clearance, ground_clearance

    def validate_path(self, start_deg, goal_deg, samples=301):
        start = joint_vector(start_deg)
        goal = joint_vector(goal_deg)
        minimum_self = float("inf")
        minimum_ground = float("inf")
        minimum_self_alpha = 0.0
        minimum_ground_alpha = 0.0
        for alpha in np.linspace(0.0, 1.0, int(samples)):
            joints = (1.0 - alpha) * start + alpha * goal
            self_clearance, ground_clearance = self.clearance(joints)
            if self_clearance < minimum_self:
                minimum_self = self_clearance
                minimum_self_alpha = float(alpha)
            if ground_clearance < minimum_ground:
                minimum_ground = ground_clearance
                minimum_ground_alpha = float(alpha)
        report = {
            "minimum_self_clearance_m": minimum_self,
            "minimum_self_alpha": minimum_self_alpha,
            "minimum_ground_clearance_m": minimum_ground,
            "minimum_ground_alpha": minimum_ground_alpha,
            "samples": int(samples),
            "passed": (
                minimum_self >= self.min_self and
                minimum_ground >= self.min_ground),
        }
        return report

    def require_safe_state(self, joints_deg):
        self_clearance, ground_clearance = self.clearance(joints_deg)
        if self_clearance < self.min_self:
            raise RuntimeError(
                f"self clearance {self_clearance:.4f}m below "
                f"{self.min_self:.4f}m")
        if ground_clearance < self.min_ground:
            raise RuntimeError(
                f"ground clearance {ground_clearance:.4f}m below "
                f"{self.min_ground:.4f}m")


class SafeHomeMover:
    def __init__(self, robot, validator, speed_percent=1,
                 max_segment_deg=25.0, arrival_tolerance_deg=1.0,
                 segment_timeout_sec=90.0):
        self.robot = robot
        self.backend = RealPlannedMoveJBackend(robot)
        self.validator = validator
        self.speed = int(speed_percent)
        self.max_segment = float(max_segment_deg)
        self.arrival_tolerance = float(arrival_tolerance_deg)
        self.segment_timeout = float(segment_timeout_sec)
        self.motion_started = False

    def state(self):
        code, state = self.robot.rm_get_current_arm_state()
        if code != 0:
            raise RuntimeError(f"RM75 state query failed: {code}")
        joints = joint_vector(state.get("joint", []))
        return joints, state

    def preflight(self, goal_deg=SAFE_REAL_HOME_DEG):
        start, state = self.state()
        joint_min = joint_vector(self.robot.rm_algo_get_joint_min_limit())
        joint_max = joint_vector(self.robot.rm_algo_get_joint_max_limit())
        goal = joint_vector(goal_deg)
        margin = 5.0
        if np.any(goal < joint_min + margin) or np.any(goal > joint_max - margin):
            raise RuntimeError("safe-home target violates SDK joint-limit margin")
        if np.any(start < joint_min) or np.any(start > joint_max):
            raise RuntimeError("current joints violate SDK joint limits")
        trajectory = self.robot.rm_get_arm_current_trajectory()
        if int(trajectory.get("return_code", -999)) != 0:
            raise RuntimeError("failed to query current RM75 trajectory")
        path = self.validator.validate_path(start, goal)
        if not path["passed"]:
            raise RuntimeError("URDF collision/ground path validation failed")
        waypoints = generate_linear_waypoints(
            start, goal, self.max_segment)
        return {
            "start_joints_deg": start,
            "start_pose": state.get("pose", []),
            "goal_joints_deg": goal,
            "goal_pose": self.robot.rm_algo_forward_kinematics(goal.tolist()),
            "joint_delta_deg": goal - start,
            "waypoints": waypoints,
            "path": path,
            "trajectory_type_before": trajectory.get("trajectory_type"),
        }

    def _wait_for_waypoint(self, segment_start, target, segment_index):
        deadline = time.monotonic() + self.segment_timeout
        last_log = -float("inf")
        stable = 0
        previous = segment_start.copy()
        lower = np.minimum(segment_start, target) - 2.0
        upper = np.maximum(segment_start, target) + 2.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            joints, _state = self.state()
            if np.any(joints < lower) or np.any(joints > upper):
                raise RuntimeError(
                    f"segment {segment_index} escaped its joint corridor")
            if float(np.max(np.abs(joints - previous))) > 10.0:
                raise RuntimeError(
                    f"segment {segment_index} readback jump exceeded 10deg")
            self.validator.require_safe_state(joints)
            error = float(np.max(np.abs(joints - target)))
            if now - last_log >= 0.5:
                last_log = now
                print(
                    f"SEGMENT {segment_index} error_deg={error:.3f} "
                    f"q={np.round(joints, 3).tolist()}", flush=True)
            if error <= self.arrival_tolerance:
                stable += 1
                if stable >= 3:
                    return joints
            else:
                stable = 0
            previous = joints
            time.sleep(0.1)
        raise RuntimeError(f"segment {segment_index} timed out")

    def execute(self, preflight_report):
        planned_start = joint_vector(preflight_report["start_joints_deg"])
        actual_start, _state = self.state()
        if float(np.max(np.abs(actual_start - planned_start))) > 1.0:
            raise RuntimeError("RM75 moved after preflight; refusing stale path")
        previous = actual_start
        for index, waypoint in enumerate(preflight_report["waypoints"], 1):
            target = joint_vector(waypoint)
            print(
                f"COMMAND SEGMENT {index}/{len(preflight_report['waypoints'])} "
                f"target_deg={np.round(target, 3).tolist()}", flush=True)
            code = self.backend.send(target, self.speed)
            if code != 0:
                raise RuntimeError(
                    f"planned MoveJ segment {index} returned {code}")
            self.motion_started = True
            previous = self._wait_for_waypoint(previous, target, index)
        final_error = float(np.max(np.abs(
            previous - joint_vector(preflight_report["goal_joints_deg"]))))
        return {"final_joints_deg": previous, "final_error_deg": final_error}

    def stop(self, emergency=False):
        if not self.motion_started:
            return None
        return (
            self.backend.emergency_stop() if emergency
            else self.backend.slow_stop())


def _json(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.1.18")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--accept-workspace-clear", action="store_true")
    args = parser.parse_args(argv)
    execute = bool(args.execute)
    if execute and not (
            args.enable_motion and args.accept_workspace_clear):
        print(
            "REFUSED: execution requires --execute --enable-motion "
            "--accept-workspace-clear", file=sys.stderr)
        return 2

    urdf = (Path(get_package_share_directory("rm_description")) /
            "urdf" / "rm_75.urdf")
    validator = CollisionValidator(urdf)
    robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    connected = False
    mover = None
    exit_code = 1
    try:
        handle = robot.rm_create_robot_arm(args.robot_ip, args.robot_port)
        if handle.id < 0:
            raise RuntimeError(f"RM75 connection failed: {handle.id}")
        connected = True
        mover = SafeHomeMover(robot, validator)
        report = mover.preflight()
        printable = dict(report)
        printable["waypoints"] = [
            np.round(waypoint, 3).tolist()
            for waypoint in report["waypoints"]]
        print("SAFE_HOME_PREFLIGHT " + json.dumps(
            printable, default=_json, sort_keys=True), flush=True)
        if not execute:
            print("READ_ONLY: no motion API was called", flush=True)
            return 0
        print(
            f"EXECUTING: {len(report['waypoints'])} checkpointed MoveJ "
            "segments at 1% speed", flush=True)
        result = mover.execute(report)
        print("SAFE_HOME_RESULT " + json.dumps(
            result, default=_json, sort_keys=True), flush=True)
        exit_code = 0
        return 0
    except KeyboardInterrupt:
        print("INTERRUPTED: requesting RM75 slow stop", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"SAFE_HOME_FAULT: {error}", file=sys.stderr, flush=True)
        return exit_code
    finally:
        if mover is not None and mover.motion_started and exit_code != 0:
            try:
                code = mover.stop(False)
                print(f"SAFE_HOME_STOP_CODE {code}", file=sys.stderr)
            except Exception as stop_error:
                print(f"SAFE_HOME_STOP_FAILED: {stop_error}", file=sys.stderr)
        if connected:
            robot.rm_delete_robot_arm()


if __name__ == "__main__":
    raise SystemExit(main())
