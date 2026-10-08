from __future__ import annotations

from pathlib import Path

import numpy as np


# MediaPipe/Wuji:
# [MCP, PIP, DIP, TIP]
MP_FINGERS = [
    [1, 2, 3, 4],       # thumb
    [5, 6, 7, 8],       # index
    [9, 10, 11, 12],    # middle
    [13, 14, 15, 16],   # ring
    [17, 18, 19, 20],   # pinky
]


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)

    if n > 1e-9:
        return v / n

    if fallback is not None:
        f = np.asarray(fallback, dtype=np.float64)
        fn = np.linalg.norm(f)
        if fn > 1e-9:
            return f / fn

    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _rotation_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Shortest proper rotation mapping unit vector a onto b.
    """
    a = _unit(a)
    b = _unit(b)

    c = float(np.clip(np.dot(a, b), -1.0, 1.0))

    v = np.cross(a, b)
    ss = float(np.linalg.norm(v))

    if ss < 1e-10:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)

        # 180 deg: choose any stable orthogonal axis.
        basis = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        tmp = min(
            basis,
            key=lambda x: abs(float(np.dot(a, x))),
        )

        axis = _unit(np.cross(a, tmp))

        return (
            2.0 * np.outer(axis, axis)
            - np.eye(3, dtype=np.float64)
        )

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ], dtype=np.float64)

    return (
        np.eye(3, dtype=np.float64)
        + vx
        + (vx @ vx) * ((1.0 - c) / (ss * ss))
    )


def _relative_direction(
    current_human_dir: np.ndarray,
    robot_open_dir: np.ndarray,
    calib_R: np.ndarray,
) -> np.ndarray:
    """
    Preserve motion RELATIVE to the calibrated natural-open pose.

    Previous calibration satisfies approximately:

        calib_R @ human_open_dir = robot_open_dir

    Therefore:

        human_open_dir = calib_R.T @ robot_open_dir

    At runtime:
        delta_R: human_open -> human_current

    Then transfer that relative motion onto robot-open geometry:
        robot_current = delta_R @ robot_open
    """

    robot_open = _unit(robot_open_dir)

    human_open = _unit(
        calib_R.T @ robot_open
    )

    human_current = _unit(
        current_human_dir,
        human_open,
    )

    delta_R = _rotation_a_to_b(
        human_open,
        human_current,
    )

    return _unit(
        delta_R @ robot_open,
        robot_open,
    )


class RobotMorphology:
    """
    Convert a human/Wuji 21-point skeleton into a robot-shaped target skeleton.

    Important:
      - wrist stays at the input wrist
      - MCP positions come from the robot's actual canonical geometry
      - MCP->PIP / PIP->DIP / DIP->TIP lengths come from the robot
      - segment DIRECTIONS come from Wuji

    Therefore we preserve motion/flexion direction while removing the
    human-vs-robot bone-length and palm-width mismatch.
    """

    def __init__(self, optimizer, include_thumb: bool = True):
        self.optimizer = optimizer
        self.robot = optimizer.robot
        self.include_thumb = bool(include_thumb)

        # Optional per-segment index-finger direction calibration.
        #
        # Each 3x3 matrix maps the user's natural-open Wuji segment
        # direction onto the L20 canonical-open segment direction.
        self.index_dir_correction = None

        calib_path = (
            Path(__file__).resolve().parents[1]
            / "example"
            / "config"
            / "l20_right_index_dir_correction.npy"
        )

        if calib_path.exists():
            arr = np.load(calib_path)

            if arr.shape != (3, 3, 3):
                raise RuntimeError(
                    f"Bad index direction calibration shape: {arr.shape}"
                )

            self.index_dir_correction = arr.astype(np.float64)

            print(
                f"[RobotMorphology] loaded index direction calibration: "
                f"{calib_path}"
            )

        # Optional thumb natural-open direction calibration.
        self.thumb_dir_correction = None

        thumb_calib_path = (
            Path(__file__).resolve().parents[1]
            / "example"
            / "config"
            / "l20_right_thumb_dir_correction.npy"
        )

        if thumb_calib_path.exists():
            arr = np.load(thumb_calib_path)

            if arr.shape != (3, 3, 3):
                raise RuntimeError(
                    f"Bad thumb direction calibration shape: {arr.shape}"
                )

            self.thumb_dir_correction = arr.astype(np.float64)

            print(
                f"[RobotMorphology] loaded thumb direction calibration: "
                f"{thumb_calib_path}"
            )

        # Optional middle-finger natural-open direction calibration.
        self.middle_dir_correction = None

        middle_calib_path = (
            Path(__file__).resolve().parents[1]
            / "example"
            / "config"
            / "l20_right_middle_dir_correction.npy"
        )

        if middle_calib_path.exists():
            arr = np.load(middle_calib_path)

            if arr.shape != (3, 3, 3):
                raise RuntimeError(
                    f"Bad middle direction calibration shape: {arr.shape}"
                )

            self.middle_dir_correction = arr.astype(np.float64)

            print(
                f"[RobotMorphology] loaded middle direction calibration: "
                f"{middle_calib_path}"
            )

        # Canonical robot pose.
        q0 = np.zeros(self.robot.model.nq, dtype=np.float64)
        q0 = np.clip(
            q0,
            self.robot.joint_limits[:, 0],
            self.robot.joint_limits[:, 1],
        )

        self.robot.compute_forward_kinematics(q0)

        origin_id = self.robot.get_link_index(
            optimizer.origin_link_name
        )

        origin = self.robot.get_link_pose(origin_id)[:3, 3].copy()

        self.mcp_anchor = np.zeros((5, 3), dtype=np.float64)
        self.seg_len = np.zeros((5, 3), dtype=np.float64)
        self.canonical_dirs = np.zeros((5, 3, 3), dtype=np.float64)

        for fi in range(5):
            names = [
                optimizer.link1_names[fi],
                optimizer.link3_names[fi],
                optimizer.link4_names[fi],
                optimizer.task_link_names[fi],
            ]

            p = []

            for name in names:
                idx = self.robot.get_link_index(name)
                p.append(
                    self.robot.get_link_pose(idx)[:3, 3].copy()
                )

            p = np.asarray(p, dtype=np.float64)

            # MCP anchor relative to robot wrist/base.
            self.mcp_anchor[fi] = p[0] - origin

            # Robot physical segment lengths.
            self.seg_len[fi, 0] = np.linalg.norm(p[1] - p[0])
            self.seg_len[fi, 1] = np.linalg.norm(p[2] - p[1])
            self.seg_len[fi, 2] = np.linalg.norm(p[3] - p[2])

            # Canonical fallback directions.
            self.canonical_dirs[fi, 0] = _unit(p[1] - p[0])
            self.canonical_dirs[fi, 1] = _unit(p[2] - p[1])
            self.canonical_dirs[fi, 2] = _unit(p[3] - p[2])

        print("[RobotMorphology] enabled")
        print("[RobotMorphology] MCP anchors (cm):")
        print(np.round(self.mcp_anchor * 100.0, 3))

        print("[RobotMorphology] segment lengths MCP-PIP/PIP-DIP/DIP-TIP (cm):")
        print(np.round(self.seg_len * 100.0, 3))

    def apply(self, keypoints: np.ndarray) -> np.ndarray:
        kp = np.asarray(keypoints, dtype=np.float64)

        if kp.shape != (21, 3):
            raise ValueError(
                f"RobotMorphology expected (21,3), got {kp.shape}"
            )

        out = kp.copy()

        wrist = kp[0].copy()
        out[0] = wrist

        for fi, ids in enumerate(MP_FINGERS):
            if fi == 0 and not self.include_thumb:
                continue

            mcp, pip, dip, tip = ids

            # Use Wuji only for segment directions.
            d1 = _unit(
                kp[pip] - kp[mcp],
                self.canonical_dirs[fi, 0],
            )

            d2 = _unit(
                kp[dip] - kp[pip],
                self.canonical_dirs[fi, 1],
            )

            d3 = _unit(
                kp[tip] - kp[dip],
                self.canonical_dirs[fi, 2],
            )

            # -------------------------------------------------
            # Thumb:
            # map user's natural-open thumb segment directions
            # onto L20 canonical-open thumb directions.
            #
            # Only a fixed rotation is applied to each segment,
            # so runtime thumb motion is still preserved.
            # -------------------------------------------------
            if fi == 0 and self.thumb_dir_correction is not None:

                d1 = _relative_direction(
                    d1,
                    self.canonical_dirs[fi, 0],
                    self.thumb_dir_correction[0],
                )

                d2 = _relative_direction(
                    d2,
                    self.canonical_dirs[fi, 1],
                    self.thumb_dir_correction[1],
                )

                d3 = _relative_direction(
                    d3,
                    self.canonical_dirs[fi, 2],
                    self.thumb_dir_correction[2],
                )

            # -------------------------------------------------
            # Middle finger:
            # map user's natural-open middle segment directions
            # onto L20 canonical-open directions.
            #
            # This removes the inward natural-open bias that can
            # make middle collide with index, while preserving
            # dynamic bending.
            # -------------------------------------------------
            if fi == 2 and self.middle_dir_correction is not None:

                d1 = _relative_direction(
                    d1,
                    self.canonical_dirs[fi, 0],
                    self.middle_dir_correction[0],
                )

                d2 = _relative_direction(
                    d2,
                    self.canonical_dirs[fi, 1],
                    self.middle_dir_correction[1],
                )

                d3 = _relative_direction(
                    d3,
                    self.canonical_dirs[fi, 2],
                    self.middle_dir_correction[2],
                )

            # -------------------------------------------------
            # Index finger:
            # map user's natural-open segment directions onto
            # L20 canonical-open directions.
            #
            # This is a FIXED rotation per segment, so dynamic
            # finger motion is preserved after zero calibration.
            # -------------------------------------------------
            if fi == 1 and self.index_dir_correction is not None:

                d1 = _relative_direction(
                    d1,
                    self.canonical_dirs[fi, 0],
                    self.index_dir_correction[0],
                )

                d2 = _relative_direction(
                    d2,
                    self.canonical_dirs[fi, 1],
                    self.index_dir_correction[1],
                )

                d3 = _relative_direction(
                    d3,
                    self.canonical_dirs[fi, 2],
                    self.index_dir_correction[2],
                )

            # Robot MCP is fixed by actual palm morphology.
            out[mcp] = (
                wrist
                + self.mcp_anchor[fi]
            )

            # Reconstruct every segment using ROBOT length.
            out[pip] = (
                out[mcp]
                + d1 * self.seg_len[fi, 0]
            )

            out[dip] = (
                out[pip]
                + d2 * self.seg_len[fi, 1]
            )

            out[tip] = (
                out[dip]
                + d3 * self.seg_len[fi, 2]
            )

        return out
