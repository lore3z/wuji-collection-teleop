#!/usr/bin/env python3
"""Right-arm-only PICO input node.

This keeps the upstream four-tracker/two-arm node unchanged while reusing its
tracking, filtering, incremental-control, TF, and topic-publishing logic.
Exactly two physical trackers are configured:

* ``pico_right_wrist``: tracker worn on the back of the right hand
* ``pico_right_arm``: tracker worn on the outside of the right upper arm

The wrist drives the right-arm target pose.  The upper-arm tracker contributes
position only to the right elbow-direction constraint.
"""

import rclpy

from pico_input import pico_input_node as dual_arm_node


_RIGHT_ARM_TRACKER_SLOTS = {
    "tracker_sn_right_wrist": "pico_right_wrist",
    "tracker_sn_right_arm": "pico_right_arm",
}


class PicoRightArmInputNode(dual_arm_node.PicoInputNode):
    """Two-tracker input node for right-arm teleoperation."""

    def __init__(self):
        # PicoInputNode deliberately uses this module-level schema for parameter
        # declaration, validation, role lookup, and missing-tracker checks.  Set
        # it before construction so every inherited path consistently expects
        # exactly the two right-side trackers.
        dual_arm_node._TRACKER_SLOTS = _RIGHT_ARM_TRACKER_SLOTS
        super().__init__()
        self.get_logger().info(
            "Right-arm-only mode: right wrist + right upper-arm trackers"
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PicoRightArmInputNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
