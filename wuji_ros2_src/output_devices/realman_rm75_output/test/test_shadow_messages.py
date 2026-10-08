import math

import numpy as np
from builtin_interfaces.msg import Time

from realman_rm75_output.rm75_joint_sender_fake_test import (
    RM75JointSenderFakeTest,
)


def test_shadow_joint_state_converts_degrees_to_radians_in_rm75_order():
    msg = RM75JointSenderFakeTest._joint_state(
        Time(sec=1), [0, 90, -90, 180, -180, 45, -45])
    assert msg.name == [f"joint{index}" for index in range(1, 8)]
    np.testing.assert_allclose(msg.position, [
        0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi,
        math.pi / 4, -math.pi / 4,
    ])


def test_shadow_pose_uses_rm75_base_frame():
    msg = RM75JointSenderFakeTest._pose_stamped(
        Time(sec=2), [0.1, -0.2, 0.3], [0, 0, 0, 1])
    assert msg.header.frame_id == "base_link"
    assert msg.pose.position.x == 0.1
    assert msg.pose.position.y == -0.2
    assert msg.pose.position.z == 0.3
    assert msg.pose.orientation.w == 1.0


def test_shadow_mode_and_real_motion_are_mutually_exclusive_by_construction():
    # The diagnostic-only residual bypass is guarded by publish_shadow; the
    # public shadow entry point constructs real_motion=False.
    import inspect
    from realman_rm75_output import rm75_joint_sender_fake_test as module

    source = inspect.getsource(module.shadow_main)
    assert "publish_shadow=True" in source
    assert "real_motion=True" not in source


def test_shadow_fault_handler_cannot_enter_fault_phase():
    node = object.__new__(RM75JointSenderFakeTest)
    node.publish_shadow = True
    node.phase = "ACTIVE"

    class Logger:
        def warning(self, _message):
            pass

    node.get_logger = lambda: Logger()
    node._fault("diagnostic guard")
    assert node.phase == "ACTIVE"
