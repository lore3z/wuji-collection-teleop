#!/usr/bin/env python3

import json
import socket
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

import skeleton_teleop_v9_calibrated_sim as sim


ROOT = Path(__file__).resolve().parent

HOST = "127.0.0.1"
PORT = 15121


def drain_latest(sock):
    latest = None

    while True:
        try:
            data, _ = sock.recvfrom(
                65535
            )
            latest = data

        except BlockingIOError:
            break

    return latest


def main():

    print("=" * 72)
    print("V9.3 LIVE SKELETON + L20 VIEWER")
    print("=" * 72)
    print(
        "显示的是：实机控制程序实际产生的 V9.3 q21 target"
    )
    print(
        "Viewer 完全独立，不连接 Wuji SDK，不控制实体手"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Same L20 model/order as V9.3
    # --------------------------------------------------------

    cfg_path = (
        ROOT
        / "example/config/"
        "l20_feature_retarget_wuji_right.yaml"
    ).resolve()

    retargeter = sim.Retargeter.from_yaml(
        str(cfg_path),
        hand_side="right",
    )

    adapter = (
        retargeter.optimizer.adapter
    )

    with open(
        cfg_path,
        "r",
    ) as f:
        cfg = yaml.safe_load(f)

    mjcf_path = Path(
        cfg["optimizer"]["mjcf_path"]
    ).expanduser()

    if not mjcf_path.is_absolute():
        mjcf_path = (
            cfg_path.parent
            / mjcf_path
        ).resolve()

    print("MJCF:", mjcf_path)

    model = mujoco.MjModel.from_xml_path(
        str(mjcf_path)
    )

    data = mujoco.MjData(
        model
    )

    joint_map = sim.build_joint_map(
        model,
        adapter.q_names,
    )

    # --------------------------------------------------------
    # UDP input
    # --------------------------------------------------------

    sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )

    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_REUSEADDR,
        1,
    )

    sock.bind(
        (HOST, PORT)
    )

    sock.setblocking(
        False
    )

    print(
        f"Listening: udp://{HOST}:{PORT}"
    )

    kp = None
    q21 = None

    last_packet = 0.0

    # --------------------------------------------------------
    # Viewer
    # --------------------------------------------------------

    with mujoco.viewer.launch_passive(
        model,
        data,
    ) as viewer:

        while viewer.is_running():

            raw = drain_latest(
                sock
            )

            if raw is not None:

                try:
                    msg = json.loads(
                        raw.decode("utf-8")
                    )

                    new_kp = np.asarray(
                        msg["kp"],
                        dtype=np.float64,
                    )

                    new_q = np.asarray(
                        msg["q21"],
                        dtype=np.float64,
                    )

                    if (
                        new_kp.shape == (21, 3)
                        and
                        new_q.shape == (21,)
                        and
                        np.all(np.isfinite(new_kp))
                        and
                        np.all(np.isfinite(new_q))
                    ):
                        kp = new_kp
                        q21 = new_q
                        last_packet = (
                            time.monotonic()
                        )

                except Exception as e:
                    print(
                        "Bad viewer packet:",
                        repr(e),
                    )

            if q21 is not None:

                for (
                    qi,
                    qadr,
                    _name,
                ) in joint_map:

                    data.qpos[
                        qadr
                    ] = q21[
                        qi
                    ]

                mujoco.mj_forward(
                    model,
                    data,
                )

            if kp is not None:

                # EXACT same Skeleton drawing as previous sim.
                sim.draw_skeleton(
                    viewer.user_scn,
                    kp,
                )

            viewer.sync()

            # Viewer refresh is independent of hardware control.
            time.sleep(
                1.0 / 60.0
            )

    sock.close()


if __name__ == "__main__":
    main()
