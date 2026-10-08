from realman_rm75_output.movej_backend import RealFollowJointBackend


class RobotStub:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name == "rm_" + "movej_canfd":
            def call(*args):
                self.calls.append(args)
                return 0
            return call
        raise AttributeError(name)

    def rm_set_arm_slow_stop(self):
        return 11

    def rm_set_arm_stop(self):
        return 12


def test_real_backend_uses_follow_true_and_rejects_offset_escape():
    robot = RobotStub()
    backend = RealFollowJointBackend(robot, [10] * 7, [0] * 7, [20] * 7)
    assert backend.send([10.5] * 7, 0.0) == 0
    assert robot.calls == [([10.5] * 7, True, 0, 0, 0)]
    assert backend.send([20.01] * 7, 0.0) == -10001
    assert len(robot.calls) == 1
    assert backend.hold_stop() == 11
    assert backend.fault_stop() == 12
