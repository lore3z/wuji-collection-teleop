import numpy as np

from realman_rm75_output.movej_backend import RealPlannedMoveJBackend
from realman_rm75_output.rm75_safe_home_mover import (
    SAFE_REAL_HOME_DEG,
    generate_linear_waypoints,
)


class RobotStub:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name == "rm_" + "movej":
            def call(*args):
                self.calls.append(args)
                return 0
            return call
        raise AttributeError(name)

    def rm_set_arm_slow_stop(self):
        return 11

    def rm_set_arm_stop(self):
        return 12


def test_waypoints_bound_every_segment_and_finish_at_safe_home():
    start = np.array([
        0.0, 22.9183118, 0.0, 85.9436693, 0.0, -18.862933, -350.0])
    waypoints = generate_linear_waypoints(
        start, SAFE_REAL_HOME_DEG, max_segment_deg=25.0)
    assert len(waypoints) == 4
    previous = start
    for waypoint in waypoints:
        assert np.max(np.abs(waypoint - previous)) <= 25.0 + 1e-12
        previous = waypoint
    np.testing.assert_allclose(waypoints[-1], SAFE_REAL_HOME_DEG)


def test_planned_backend_uses_nonblocking_unblended_movej_at_one_percent():
    robot = RobotStub()
    backend = RealPlannedMoveJBackend(robot)
    assert backend.send(SAFE_REAL_HOME_DEG, 1) == 0
    assert robot.calls == [(
        SAFE_REAL_HOME_DEG.tolist(), 1, 0, 0, 0)]
    assert backend.slow_stop() == 11
    assert backend.emergency_stop() == 12


def test_planned_backend_rejects_invalid_command_without_sdk_call():
    robot = RobotStub()
    backend = RealPlannedMoveJBackend(robot)
    assert backend.send([0.0] * 6, 1) == -10000
    assert backend.send([0.0] * 7, 0) == -10000
    assert robot.calls == []
