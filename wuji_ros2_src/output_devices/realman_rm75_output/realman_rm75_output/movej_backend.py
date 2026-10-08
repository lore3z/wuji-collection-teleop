"""Narrow boundary containing the only real RM75 joint command call."""

import numpy as np


class FakeMoveJBackend:
    def __init__(self):
        self.sent = []

    def send(self, joints_deg):
        self.sent.append(list(joints_deg))
        return 0

    def slow_stop(self):
        return 0

    def emergency_stop(self):
        return 0


class RealMoveJBackend:
    def __init__(self, robot):
        self.robot = robot

    def send(self, joints_deg):
        # Low-follow is deliberate for the first 50 Hz, +/-1 degree trial.
        return self._send_canfd(joints_deg, False)

    def _send_canfd(self, joints_deg, follow):
        return self.robot.rm_movej_canfd(
            list(joints_deg), bool(follow), 0, 0, 0)

    def slow_stop(self):
        return self.robot.rm_set_arm_slow_stop()

    def emergency_stop(self):
        return self.robot.rm_set_arm_stop()


class RealFollowJointBackend(RealMoveJBackend):
    """125 Hz follow=True backend with a final joint-limit guard."""

    def __init__(self, robot, initial_joints, min_joints_deg, max_joints_deg):
        super().__init__(robot)
        self.initial = np.asarray(initial_joints, dtype=float)
        if self.initial.shape != (7,):
            raise ValueError("initial joints must contain seven values")
        self.min_joints = np.asarray(min_joints_deg, dtype=float)
        self.max_joints = np.asarray(max_joints_deg, dtype=float)
        if self.min_joints.shape != (7,) or self.max_joints.shape != (7,):
            raise ValueError("joint guards must contain seven values")
        self.max_observed_offset = 0.0

    def send(self, joints, _sent_at):
        command = np.asarray(joints, dtype=float)
        if command.shape != (7,) or not np.all(np.isfinite(command)):
            return -10000
        offset = float(np.max(np.abs(command - self.initial)))
        self.max_observed_offset = max(self.max_observed_offset, offset)
        if (np.any(command < self.min_joints - 1e-6) or
                np.any(command > self.max_joints + 1e-6)):
            return -10001
        return self._send_canfd(command, True)

    def hold_stop(self):
        return self.robot.rm_set_arm_slow_stop()

    def fault_stop(self):
        return self.robot.rm_set_arm_stop()


class RealPlannedMoveJBackend:
    """Controller-planned, non-blocking MoveJ used only by safe homing."""

    def __init__(self, robot):
        self.robot = robot

    def send(self, joints_deg, speed_percent):
        joints = np.asarray(joints_deg, dtype=float)
        speed = int(speed_percent)
        if (joints.shape != (7,) or not np.all(np.isfinite(joints)) or
                not 1 <= speed <= 100):
            return -10000
        return self.robot.rm_movej(
            joints.tolist(), speed, 0, 0, 0)

    def slow_stop(self):
        return self.robot.rm_set_arm_slow_stop()

    def emergency_stop(self):
        return self.robot.rm_set_arm_stop()
