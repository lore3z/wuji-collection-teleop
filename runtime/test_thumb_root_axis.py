#!/usr/bin/env python3

import argparse
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class AxisTest(Node):
    def __init__(self):
        super().__init__("thumb_axis_test")

        self.state = None

        self.sub = self.create_subscription(
            JointState,
            "/cb_right_hand_state",
            self.on_state,
            10,
        )

        self.pub = self.create_publisher(
            JointState,
            "/cb_right_hand_control_cmd",
            10,
        )

    def on_state(self, msg):
        if len(msg.position) == 20:
            self.state = list(msg.position)

    def send(self, raw, duration=0.5):
        end = time.monotonic() + duration

        while time.monotonic() < end:
            msg = JointState()

            msg.header.stamp = (
                self.get_clock()
                .now()
                .to_msg()
            )

            msg.name = [
                f"joint{i+1}"
                for i in range(20)
            ]

            msg.position = [
                float(int(round(x)))
                for x in raw
            ]

            msg.velocity = [255.0] * 20

            self.pub.publish(msg)

            rclpy.spin_once(
                self,
                timeout_sec=0.001,
            )

            time.sleep(0.02)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--channel",
        type=int,
        required=True,
        choices=[0, 5, 10],
    )

    ap.add_argument(
        "--delta",
        type=int,
        default=8,
    )

    args = ap.parse_args()

    rclpy.init()

    node = AxisTest()

    print("waiting /cb_right_hand_state ...")

    t0 = time.monotonic()

    while (
        node.state is None
        and
        time.monotonic() - t0 < 3.0
    ):
        rclpy.spin_once(
            node,
            timeout_sec=0.1,
        )

    if node.state is None:
        raise SystemExit(
            "ERROR: 没收到 raw20 state"
        )

    base = node.state.copy()

    ch = args.channel
    d = args.delta

    plus = base.copy()
    minus = base.copy()

    plus[ch] = min(
        255,
        base[ch] + d,
    )

    minus[ch] = max(
        0,
        base[ch] - d,
    )

    print()
    print("channel =", ch)
    print("base    =", base[ch])
    print("+delta  =", plus[ch])
    print("-delta  =", minus[ch])
    print()

    print("[1] base")
    node.send(base, 0.6)

    print("[2] +delta")
    node.send(plus, 0.7)

    print("[3] back base")
    node.send(base, 0.7)

    print("[4] -delta")
    node.send(minus, 0.7)

    print("[5] back base")
    node.send(base, 0.8)

    print("DONE")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
