"""Narrow pose-pass-through backend boundary."""

from .pose_mapper import normalize_quaternion_xyzw


class FakeMovePBackend:
    def __init__(self, publish_callback=None):
        self.publish_callback = publish_callback
        self.sent = []

    def send_pose(self, position, quaternion_xyzw, timestamp):
        record = (list(position), list(quaternion_xyzw), float(timestamp))
        self.sent.append(record)
        if self.publish_callback is not None:
            self.publish_callback(position, quaternion_xyzw)
        return 0


class RealMovePBackend:
    """The package's only real pose-transmission and stop call site."""

    def __init__(self, robot, follow=False):
        self.robot = robot
        self.follow = bool(follow)

    def send_pose(self, position, quaternion_xyzw, _timestamp):
        qx, qy, qz, qw = normalize_quaternion_xyzw(quaternion_xyzw)
        pose_wxyz = [float(position[0]), float(position[1]), float(position[2]),
                     float(qw), float(qx), float(qy), float(qz)]
        return self.robot.rm_movep_canfd(pose_wxyz, self.follow)

    def slow_stop(self):
        return self.robot.rm_set_arm_slow_stop()

    def emergency_stop(self):
        return self.robot.rm_set_arm_stop()
