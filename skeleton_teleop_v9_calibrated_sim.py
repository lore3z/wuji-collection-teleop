#!/usr/bin/env python3

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import mujoco
import mujoco.viewer


# ============================================================
# Project imports
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = PROJECT_DIR / "example"

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from input_devices.wuji_glove_device import WujiGloveDevice

from wuji_retargeting.retarget import Retargeter
from wuji_retargeting.mediapipe import (
    apply_mediapipe_transformations,
)


# ============================================================
# MediaPipe 21-point skeleton
# ============================================================

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (0, 9), (9, 10), (10, 11), (11, 12),

    (0, 13), (13, 14), (14, 15), (15, 16),

    (0, 17), (17, 18), (18, 19), (19, 20),

    (5, 9), (9, 13), (13, 17),
]


def draw_skeleton(
    scene,
    kp,
    offset=np.array([0.0, -0.18, 0.0]),
):
    kp = np.asarray(
        kp,
        dtype=np.float64,
    ).copy()

    if kp.shape != (21, 3):
        return

    if not np.all(np.isfinite(kp)):
        return

    # wrist centered
    kp -= kp[0]

    kp += np.asarray(
        offset,
        dtype=np.float64,
    )

    scene.ngeom = 0

    max_geom = len(scene.geoms)

    # bones
    for a, b in SKELETON_EDGES:

        if scene.ngeom >= max_geom:
            break

        g = scene.geoms[scene.ngeom]

        mujoco.mjv_connector(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.0025,
            kp[a],
            kp[b],
        )

        g.rgba[:] = [
            0.05,
            0.85,
            1.0,
            0.95,
        ]

        scene.ngeom += 1

    # joints
    for i in range(21):

        if scene.ngeom >= max_geom:
            break

        g = scene.geoms[scene.ngeom]

        radius = (
            0.006
            if i == 0
            else 0.004
        )

        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array(
                [radius, radius, radius],
                dtype=np.float64,
            ),
            kp[i],
            np.eye(
                3,
                dtype=np.float64,
            ).reshape(-1),
            np.array(
                [1.0, 0.25, 0.05, 1.0],
                dtype=np.float32,
            ),
        )

        scene.ngeom += 1


# ============================================================
# Utility
# ============================================================

def clamp01(x):
    return float(
        np.clip(
            x,
            0.0,
            1.0,
        )
    )


def norm01(
    x,
    open_value,
    closed_value,
):
    """
    Maps human calibrated range:
        OPEN -> 0
        CLOSED -> 1

    Works regardless of whether feature increases or decreases.
    """

    d = (
        closed_value
        - open_value
    )

    if abs(d) < 1e-6:
        return 0.0

    return clamp01(
        (
            x
            - open_value
        ) / d
    )


def lerp(a, b, t):
    return (
        np.asarray(a)
        + t
        * (
            np.asarray(b)
            - np.asarray(a)
        )
    )


def robot_flex_range(
    adapter,
    index,
):
    """
    Calibrated human progress 0..1 is mapped onto the COMPLETE
    legal L20 actuator interval.

        human OPEN -> robot lower
        human FIST -> robot upper

    Do NOT assume q=0 is the robot open posture.
    """
    return (
        float(adapter.lower[index]),
        float(adapter.upper[index]),
    )


# ============================================================
# MuJoCo mapping
# ============================================================

def build_joint_map(
    model,
    q_names,
):
    mapping = []
    missing = []

    for i, name in enumerate(q_names):

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if jid < 0:
            missing.append(name)
            continue

        qadr = int(
            model.jnt_qposadr[jid]
        )

        mapping.append(
            (
                i,
                qadr,
                name,
            )
        )

    if missing:

        print(
            "MuJoCo 缺少 joint:"
        )

        for x in missing:
            print("  ", x)

        raise RuntimeError(
            "MJCF / URDF joint mismatch"
        )

    return mapping


# ============================================================
# Human measurements
# ============================================================

def measure_pose(
    optimizer,
    kp,
):
    """
    Extract the same anatomical Skeleton features already
    implemented by L20FeatureRetargeter.
    """

    out = {
        "fingers": {},
    }

    for finger in optimizer.FINGERS:

        (
            spread,
            mcp,
            pip,
            dip,
        ) = optimizer._finger_features(
            kp,
            finger,
        )

        out["fingers"][finger] = {
            "spread": float(spread),
            "mcp": float(mcp),
            "pip": float(pip),
            "dip": float(dip),
        }

    (
        d1,
        d2,
        thumb_mcp,
        thumb_ip,
        observability,
    ) = optimizer._thumb_features(
        kp
    )

    out["thumb"] = {
        "d1": np.asarray(
            d1,
            dtype=np.float64,
        ),
        "d2": np.asarray(
            d2,
            dtype=np.float64,
        ),
        "mcp": float(thumb_mcp),
        "ip": float(thumb_ip),
        "observability": float(
            observability
        ),
    }

    # scale-normalized thumb-index distance
    palm_scale = float(
        np.linalg.norm(
            kp[9]
            - kp[0]
        )
    )

    palm_scale = max(
        palm_scale,
        1e-5,
    )

    out["thumb_index_distance"] = float(
        np.linalg.norm(
            kp[4]
            - kp[8]
        )
        / palm_scale
    )

    return out


def median_measurements(
    measurements,
):
    result = {
        "fingers": {},
    }

    fingers = [
        "index",
        "middle",
        "ring",
        "pinky",
    ]

    for finger in fingers:

        result["fingers"][finger] = {}

        for key in [
            "spread",
            "mcp",
            "pip",
            "dip",
        ]:

            result["fingers"][finger][key] = float(
                np.median(
                    [
                        m["fingers"][finger][key]
                        for m in measurements
                    ]
                )
            )

    result["thumb"] = {}

    for key in [
        "mcp",
        "ip",
        "observability",
    ]:

        result["thumb"][key] = float(
            np.median(
                [
                    m["thumb"][key]
                    for m in measurements
                ]
            )
        )

    result["thumb"]["d1"] = np.median(
        np.stack(
            [
                m["thumb"]["d1"]
                for m in measurements
            ]
        ),
        axis=0,
    )

    result["thumb"]["d2"] = np.median(
        np.stack(
            [
                m["thumb"]["d2"]
                for m in measurements
            ]
        ),
        axis=0,
    )

    result[
        "thumb_index_distance"
    ] = float(
        np.median(
            [
                m[
                    "thumb_index_distance"
                ]
                for m in measurements
            ]
        )
    )

    return result


# ============================================================
# Calibration
# ============================================================

def sample_calibration_pose(
    glove,
    key_name,
    hand,
    optimizer,
    label,
    instruction,
    seconds=1.5,
):
    print()
    print("=" * 70)
    print(label)
    print(instruction)
    print("=" * 70)

    input(
        "摆好姿态并保持不动，然后按 Enter 开始采样..."
    )

    measurements = []
    keypoints = []

    start = time.monotonic()

    while (
        time.monotonic()
        - start
        < seconds
    ):

        data = glove.get_fingers_data()

        raw = data.get(
            key_name
        )

        if raw is None:
            time.sleep(0.003)
            continue

        raw = np.asarray(
            raw,
            dtype=np.float64,
        )

        if raw.shape != (21, 3):
            continue

        if not np.all(
            np.isfinite(raw)
        ):
            continue

        kp = apply_mediapipe_transformations(
            raw,
            hand,
        )

        keypoints.append(
            kp.copy()
        )

        measurements.append(
            measure_pose(
                optimizer,
                kp,
            )
        )

        time.sleep(
            0.008
        )

    if len(measurements) < 20:
        raise RuntimeError(
            f"{label} 有效 Skeleton 太少: "
            f"{len(measurements)}"
        )

    med = median_measurements(
        measurements
    )

    med_kp = np.median(
        np.stack(keypoints),
        axis=0,
    )

    print(
        f"{label} COMPLETE: "
        f"{len(measurements)} samples"
    )

    return (
        med,
        med_kp,
    )


def print_calibration(
    calib,
):
    print()
    print("=" * 80)
    print("V9 HUMAN SKELETON CALIBRATION")
    print("=" * 80)

    for finger in [
        "index",
        "middle",
        "ring",
        "pinky",
    ]:

        o = calib[
            "OPEN"
        ]["fingers"][finger]

        f = calib[
            "FIST"
        ]["fingers"][finger]

        print(
            f"{finger:6s} | "
            f"MCP "
            f"{np.degrees(o['mcp']):6.1f}"
            f" -> "
            f"{np.degrees(f['mcp']):6.1f} deg | "
            f"PIP "
            f"{np.degrees(o['pip']):6.1f}"
            f" -> "
            f"{np.degrees(f['pip']):6.1f} | "
            f"DIP "
            f"{np.degrees(o['dip']):6.1f}"
            f" -> "
            f"{np.degrees(f['dip']):6.1f}"
        )

    print(
        "thumb-index distance: "
        f"OPEN={calib['OPEN']['thumb_index_distance']:.3f} "
        f"O={calib['O']['thumb_index_distance']:.3f}"
    )

    print("=" * 80)


# ============================================================
# Robot thumb anchors
# ============================================================

def build_thumb_anchors(
    optimizer,
    adapter,
    kp_open,
    kp_o,
    kp_fist,
):
    """
    OPEN uses robot-native nominal open thumb.

    O/FIST CMC orientations are estimated once from the old
    geometry solver while the thumb is bent and therefore much
    more observable.

    Runtime does NOT repeatedly solve the thumb numerical IK.
    """

    idx = adapter.u_index

    # Robot nominal OPEN thumb already used elsewhere in project.
    open_anchor = np.array(
        [
            np.deg2rad(-10.0),
            np.deg2rad(15.0),
            np.deg2rad(4.0),
        ],
        dtype=np.float64,
    )

    open_anchor = np.array(
        [
            np.clip(
                open_anchor[0],
                adapter.lower[
                    idx["thumb_cmc_roll"]
                ],
                adapter.upper[
                    idx["thumb_cmc_roll"]
                ],
            ),
            np.clip(
                open_anchor[1],
                adapter.lower[
                    idx["thumb_cmc_yaw"]
                ],
                adapter.upper[
                    idx["thumb_cmc_yaw"]
                ],
            ),
            np.clip(
                open_anchor[2],
                adapter.lower[
                    idx["thumb_cmc_pitch"]
                ],
                adapter.upper[
                    idx["thumb_cmc_pitch"]
                ],
            ),
        ]
    )

    def solve_anchor(kp):

        optimizer.last_u16 = None
        optimizer.last_qpos = None

        q = optimizer.solve(
            kp
        )

        u = adapter.compress(
            q
        )

        return np.array(
            [
                u[
                    idx["thumb_cmc_roll"]
                ],
                u[
                    idx["thumb_cmc_yaw"]
                ],
                u[
                    idx["thumb_cmc_pitch"]
                ],
            ],
            dtype=np.float64,
        )

    o_anchor = solve_anchor(
        kp_o
    )

    fist_anchor = solve_anchor(
        kp_fist
    )

    # reset temporal state before realtime loop
    optimizer.last_u16 = None
    optimizer.last_qpos = None

    print()
    print("===== THUMB CMC ANCHORS =====")

    print(
        "OPEN:",
        np.degrees(
            open_anchor
        ).round(1),
    )

    print(
        "O   :",
        np.degrees(
            o_anchor
        ).round(1),
    )

    print(
        "FIST:",
        np.degrees(
            fist_anchor
        ).round(1),
    )

    return {
        "OPEN": open_anchor,
        "O": o_anchor,
        "FIST": fist_anchor,
    }



# ============================================================
# Live Skeleton -> Thumb CMC 3DOF
# ============================================================

def solve_thumb_cmc_live(
    optimizer,
    adapter,
    u,
    measurement,
):
    """
    Full live thumb CMC tracking.

    Human Skeleton provides:
        thumb segment-1 direction
        thumb segment-2 direction

    Solve robot:
        thumb_cmc_roll
        thumb_cmc_yaw
        thumb_cmc_pitch

    every frame.

    No OPEN/O/FIST anchor interpolation is used here.
    """

    idx = optimizer._thumb_u_idx

    lo = np.asarray(
        adapter.lower[idx],
        dtype=np.float64,
    )

    hi = np.asarray(
        adapter.upper[idx],
        dtype=np.float64,
    )

    human_d1 = np.asarray(
        measurement["thumb"]["d1"],
        dtype=np.float64,
    )

    human_d2 = np.asarray(
        measurement["thumb"]["d2"],
        dtype=np.float64,
    )

    # --------------------------------------------
    # Previous frame = warm start
    # --------------------------------------------

    previous = getattr(
        optimizer,
        "_v9_last_thumb_cmc",
        None,
    )

    if previous is None:

        # Good physical initial branch for an open L20 thumb.
        previous = np.array(
            [
                np.deg2rad(-10.0),
                np.deg2rad(15.0),
                np.deg2rad(4.0),
            ],
            dtype=np.float64,
        )

    previous = np.clip(
        previous,
        lo,
        hi,
    )

    span = np.maximum(
        hi - lo,
        1e-6,
    )

    base_u = u.copy()

    # Much lighter temporal term than the old controller.
    # Skeleton direction should dominate.
    temporal_weight = 0.005

    def objective(x, grad):

        del grad

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        test_u = base_u.copy()

        test_u[idx] = x

        r1, r2 = (
            optimizer._robot_thumb_dirs(
                test_u
            )
        )

        d1_cost = (
            1.0
            - np.clip(
                np.dot(
                    r1,
                    human_d1,
                ),
                -1.0,
                1.0,
            )
        )

        d2_cost = (
            1.0
            - np.clip(
                np.dot(
                    r2,
                    human_d2,
                ),
                -1.0,
                1.0,
            )
        )

        dx = (
            x
            - previous
        ) / span

        temporal = float(
            dx @ dx
        )

        return float(
            optimizer.thumb_dir_w1
            * d1_cost
            +
            optimizer.thumb_dir_w2
            * d2_cost
            +
            temporal_weight
            * temporal
        )

    optimizer.thumb_opt.set_min_objective(
        objective
    )

    # --------------------------------------------
    # Multi-start only when near the yaw singularity.
    # Otherwise use previous frame for smooth tracking.
    # --------------------------------------------

    seeds = [
        previous.copy(),
    ]

    yaw_i = 1

    if (
        abs(
            previous[yaw_i]
            - lo[yaw_i]
        )
        < np.deg2rad(3.0)
    ):

        for yaw_deg in [
            15.0,
            30.0,
            50.0,
            70.0,
        ]:

            seed = previous.copy()

            seed[yaw_i] = np.deg2rad(
                yaw_deg
            )

            seeds.append(
                np.clip(
                    seed,
                    lo,
                    hi,
                )
            )

    best = previous.copy()
    best_cost = objective(
        best,
        None,
    )

    for seed in seeds:

        try:

            candidate = np.asarray(
                optimizer.thumb_opt.optimize(
                    seed
                ),
                dtype=np.float64,
            )

            candidate = np.clip(
                candidate,
                lo,
                hi,
            )

            cost = objective(
                candidate,
                None,
            )

            if cost < best_cost:

                best = candidate
                best_cost = cost

        except Exception:
            pass

    u[idx] = best

    optimizer._v9_last_thumb_cmc = (
        best.copy()
    )

    return best




# ============================================================
# V9.7 HUMAN-RELATIVE ROOT FLEX
# ============================================================

def build_root_calibration(
    semantic,
    calib_kp,
):
    """
    Build a HUMAN-ONLY OPEN -> FIST proximal-direction trajectory.

    Root flex is measured relative to the user's own OPEN posture,
    not by matching the absolute human direction to the robot.
    """

    out = {}

    for finger in semantic.FINGERS:

        open_dir, _ = (
            semantic._human_finger_features(
                calib_kp["OPEN"],
                finger,
            )
        )

        fist_dir, _ = (
            semantic._human_finger_features(
                calib_kp["FIST"],
                finger,
            )
        )

        o = np.asarray(
            open_dir,
            dtype=np.float64,
        )

        f = np.asarray(
            fist_dir,
            dtype=np.float64,
        )

        o /= max(
            np.linalg.norm(o),
            1e-8,
        )

        f /= max(
            np.linalg.norm(f),
            1e-8,
        )

        dot_of = float(
            np.clip(
                np.dot(o, f),
                -1.0,
                1.0,
            )
        )

        theta = float(
            np.arccos(dot_of)
        )

        # Tangent direction from OPEN toward FIST.
        tangent = (
            f
            -
            dot_of * o
        )

        tangent_norm = float(
            np.linalg.norm(tangent)
        )

        if (
            theta < np.deg2rad(5.0)
            or
            tangent_norm < 1e-6
        ):
            print(
                f"[ROOT CALIB WARN] {finger}: "
                f"OPEN/FIST direction span only "
                f"{np.degrees(theta):.1f} deg"
            )

            tangent = np.array(
                [0.0, 0.0, 0.0],
                dtype=np.float64,
            )

        else:

            tangent /= tangent_norm

        out[finger] = {
            "open": o,
            "tangent": tangent,
            "theta": theta,
        }

        print(
            f"[ROOT CALIB] {finger:6s} "
            f"span={np.degrees(theta):5.1f} deg"
        )

    return out


def root_flex_progress(
    human_dir,
    root_calib,
    finger,
):
    """
    Project the current proximal direction onto the OPEN->FIST
    great-circle plane.

    This deliberately rejects most finger spread / lateral motion.

    Returns:
        0 = user's natural OPEN
        1 = user's calibrated FIST
    """

    c = root_calib[
        finger
    ]

    o = c["open"]
    e = c["tangent"]
    theta = float(
        c["theta"]
    )

    if (
        theta < np.deg2rad(5.0)
        or
        np.linalg.norm(e) < 1e-6
    ):
        return 0.0

    h = np.asarray(
        human_dir,
        dtype=np.float64,
    )

    h /= max(
        np.linalg.norm(h),
        1e-8,
    )

    # Angular coordinate inside the OPEN->FIST flexion plane.
    #
    # Components orthogonal to this plane mostly correspond to
    # finger spread and therefore do not strongly affect root flex.
    a = float(
        np.dot(
            h,
            o,
        )
    )

    b = float(
        np.dot(
            h,
            e,
        )
    )

    phi = float(
        np.arctan2(
            b,
            a,
        )
    )

    t = float(
        np.clip(
            phi / theta,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Very small neutral tolerance.
    #
    # Only removes ~1.5 degrees of Skeleton jitter.
    # This is NOT the large dead-zone used in the failed V9.6.
    # --------------------------------------------------------

    neutral_angle = np.deg2rad(
        1.5
    )

    neutral_t = min(
        neutral_angle
        / max(theta, 1e-6),
        0.08,
    )

    if t <= neutral_t:

        t = 0.0

    else:

        t = (
            t - neutral_t
        ) / (
            1.0 - neutral_t
        )

    t = float(
        np.clip(
            t,
            0.0,
            1.0,
        )
    )

    # --------------------------------------------------------
    # Soft start ONLY for first 15%.
    #
    # y = 2x² - x³
    #
    # At the boundary:
    #   value matches the linear curve
    #   slope also becomes exactly 1
    #
    # Therefore after 15% the mapping is completely linear.
    # --------------------------------------------------------

    soft_zone = 0.15

    if t < soft_zone:

        x = (
            t
            / soft_zone
        )

        t = (
            soft_zone
            * (
                2.0 * x * x
                -
                x * x * x
            )
        )

    return float(
        np.clip(
            t,
            0.0,
            1.0,
        )
    )




# ============================================================
# THUMB PALM-GATE
#
# Skeleton controls:
#   CMC yaw / pitch + thumb MCP
#
# When thumb tip enters the palm region:
#   robot-side CMC roll gradually takes over so the thumb pad
#   faces the palm.
#
# IMPORTANT:
#   This is NOT a fixed O pose.
#   Only the Skeleton-unobservable axial/pad DOF is corrected.
# ============================================================

THUMB_PALMAR_ROLL_DEG = -37.5

# normalized by human palm width
THUMB_PALM_GATE_BEGIN = 1.10
THUMB_PALM_GATE_FULL  = 0.72


def _smoothstep01(x):
    x = float(
        np.clip(
            x,
            0.0,
            1.0,
        )
    )

    return (
        x * x
        * (3.0 - 2.0 * x)
    )


def thumb_palm_gate_alpha(
    optimizer,
    kp,
):
    """
    Detect whether the HUMAN thumb tip has entered the palm
    grasp region.

    Uses only relative hand geometry, therefore it does not
    depend on global hand translation / rotation.
    """

    kp = np.asarray(
        kp,
        dtype=np.float64,
    )

    # MediaPipe:
    # 0 wrist
    # 4 thumb tip
    # 5/9/13/17 four MCPs

    wrist = kp[0]

    mcp = np.stack(
        [
            kp[5],
            kp[9],
            kp[13],
            kp[17],
        ],
        axis=0,
    )

    mcp_center = np.mean(
        mcp,
        axis=0,
    )

    # Center of useful palm/grasp region.
    # Slightly toward MCPs rather than wrist.
    palm_center = (
        0.25 * wrist
        +
        0.75 * mcp_center
    )

    palm_width = float(
        np.linalg.norm(
            kp[5] - kp[17]
        )
    )

    if palm_width < 1e-6:
        return 0.0

    thumb_tip = kp[4]

    d = float(
        np.linalg.norm(
            thumb_tip
            - palm_center
        )
        / palm_width
    )

    # d >= BEGIN : no intervention
    # d <= FULL  : full palmar roll
    raw = (
        THUMB_PALM_GATE_BEGIN
        - d
    ) / (
        THUMB_PALM_GATE_BEGIN
        - THUMB_PALM_GATE_FULL
    )

    alpha = _smoothstep01(
        raw
    )

    # Very light temporal stabilization only for gate switching.
    # Does NOT filter the thumb motion itself.
    prev = float(
        getattr(
            optimizer,
            "_thumb_palm_alpha",
            alpha,
        )
    )

    alpha = (
        0.70 * prev
        +
        0.30 * alpha
    )

    optimizer._thumb_palm_alpha = (
        alpha
    )

    optimizer._thumb_palm_distance = (
        d
    )

    return alpha


def apply_thumb_palm_gate(
    optimizer,
    adapter,
    u,
    kp,
):
    """
    Preserve the existing live CMC solution.

    Only blend thumb_cmc_roll toward the robot-native palmar
    orientation when the human thumb enters the palm region.

    yaw / pitch / MCP are untouched.
    """

    alpha = thumb_palm_gate_alpha(
        optimizer,
        kp,
    )

    if alpha <= 1e-4:
        return alpha

    roll_i = adapter.u_index[
        "thumb_cmc_roll"
    ]

    current_roll = float(
        u[roll_i]
    )

    palmar_roll = float(
        np.clip(
            np.deg2rad(
                THUMB_PALMAR_ROLL_DEG
            ),
            adapter.lower[roll_i],
            adapter.upper[roll_i],
        )
    )

    u[roll_i] = (
        (1.0 - alpha)
        * current_roll
        +
        alpha
        * palmar_roll
    )

    u[roll_i] = np.clip(
        u[roll_i],
        adapter.lower[roll_i],
        adapter.upper[roll_i],
    )

    optimizer._thumb_palm_debug = {
        "alpha": float(alpha),
        "distance": float(
            optimizer._thumb_palm_distance
        ),
        "roll_before_deg": float(
            np.degrees(
                current_roll
            )
        ),
        "roll_after_deg": float(
            np.degrees(
                u[roll_i]
            )
        ),
    }

    return alpha



# ============================================================
# Calibrated Skeleton -> L20
# ============================================================

def calibrated_retarget(
    optimizer,
    adapter,
    measurement,
    calib,
    thumb_anchors,
    kp,
    semantic,
    root_calib,
):
    u = np.zeros(
        16,
        dtype=np.float64,
    )

    u = np.clip(
        u,
        adapter.lower,
        adapter.upper,
    )

    finger_progress = {}

    # ========================================================
    # Four fingers -- V9.3 Robot-Native Semantic LUT
    #
    # Human:
    #   proximal direction
    #   long-baseline PIP->TIP curl
    #
    # Robot:
    #   each finger has its OWN FK-derived reachable manifold
    #
    # Therefore:
    #   NO lower->upper linear actuator mapping
    #   NO MCP/PIP/DIP independent percentage mapping
    # ========================================================

    finger_progress = {}

    semantic_diag = {}

    for finger in semantic.FINGERS:

        # ----------------------------------------------------
        # Human Skeleton semantic geometry
        #
        # proximal:
        #     MCP -> PIP direction
        #
        # curl:
        #     angle(
        #         MCP->PIP,
        #         PIP->TIP
        #     )
        #
        # Long baseline intentionally suppresses noisy DIP.
        # ----------------------------------------------------

        (
            human_dir,
            human_curl,
        ) = semantic._human_finger_features(
            kp,
            finger,
        )

        # ====================================================
        # V9.4 CALIBRATED SEMANTIC CLOSURE
        #
        # Important:
        # Human absolute anatomical curl angle is NOT expected
        # to equal the robot absolute geometric curl angle.
        #
        # Use the already measured OPEN/FIST human calibration
        # to obtain closure progress first.
        # ====================================================

        cur_h = measurement[
            "fingers"
        ][finger]

        open_h = calib[
            "OPEN"
        ]["fingers"][finger]

        fist_h = calib[
            "FIST"
        ]["fingers"][finger]

        p_root = norm01(
            cur_h["mcp"],
            open_h["mcp"],
            fist_h["mcp"],
        )

        p_pip = norm01(
            cur_h["pip"],
            open_h["pip"],
            fist_h["pip"],
        )

        p_dip = norm01(
            cur_h["dip"],
            open_h["dip"],
            fist_h["dip"],
        )

        # Distal closure is robust to one noisy Skeleton joint.
        p_curl = float(
            np.median(
                [
                    p_pip,
                    p_dip,
                ]
            )
        )

        p_curl = clamp01(
            p_curl
        )

        # ----------------------------------------------------
        # Actual robot reachable proximal direction
        # -> MCP roll/pitch.
        #
        # Keep V9.3 geometry for natural intermediate motion.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Robot-native direction LUT is still excellent for
        # MCP roll / finger spread.
        #
        # But absolute human->robot pitch matching is NOT used.
        # ----------------------------------------------------

        (
            q_roll,
            q_pitch_absolute,
            dir_err,
        ) = semantic._inverse_direction(
            finger,
            human_dir,
        )

        # ====================================================
        # V9.7 root flex
        #
        # user's OPEN proximal direction = exactly 0
        # user's FIST proximal direction = exactly 1
        #
        # Spread motion is largely rejected geometrically.
        # ====================================================

        root_t = root_flex_progress(
            human_dir,
            root_calib,
            finger,
        )

        pitch_i_tmp = adapter.u_index[
            f"{finger}_mcp_pitch"
        ]

        # Robot natural OPEN.
        q_pitch_open = float(
            np.clip(
                0.0,
                adapter.lower[pitch_i_tmp],
                adapter.upper[pitch_i_tmp],
            )
        )

        # Robot maximum native MCP flex.
        #
        # Reuse the SAME direction LUT that gave V9.3/V9.4
        # its good robot morphology handling.
        _dirs_tmp, commands_tmp = (
            semantic._direction_lut[
                finger
            ]
        )

        pitches_tmp = np.asarray(
            commands_tmp[:, 1],
            dtype=np.float64,
        )

        # L20 configuration uses positive pitch for flexion.
        # Select the furthest positive reachable root command.
        q_pitch_closed = float(
            np.max(
                pitches_tmp
            )
        )

        q_pitch = (
            q_pitch_open
            +
            root_t
            * (
                q_pitch_closed
                - q_pitch_open
            )
        )

        # ----------------------------------------------------
        # HUMAN 0..1 closure
        #       ->
        # ROBOT OWN FK CURL RANGE
        #
        # This is the important V9.4 change.
        # ----------------------------------------------------

        robot_curls, robot_qs = (
            semantic._curl_lut[
                finger
            ]
        )

        # Robot OPEN = q closest to zero.
        q0_idx = int(
            np.argmin(
                np.abs(
                    robot_qs
                )
            )
        )

        robot_open_curl = float(
            robot_curls[
                q0_idx
            ]
        )

        # Robot CLOSED = maximum actual FK curl,
        # regardless of actuator sign convention.
        qc_idx = int(
            np.argmax(
                robot_curls
            )
        )

        robot_closed_curl = float(
            robot_curls[
                qc_idx
            ]
        )

        target_robot_curl = (
            robot_open_curl
            +
            p_curl
            * (
                robot_closed_curl
                -
                robot_open_curl
            )
        )

        (
            q_pip,
            curl_err,
        ) = semantic._inverse_curl(
            finger,
            target_robot_curl,
        )

        roll_i = adapter.u_index[
            f"{finger}_mcp_roll"
        ]

        pitch_i = adapter.u_index[
            f"{finger}_mcp_pitch"
        ]

        pip_i = adapter.u_index[
            f"{finger}_pip"
        ]

        u[roll_i] = np.clip(
            q_roll,
            adapter.lower[roll_i],
            adapter.upper[roll_i],
        )

        # ====================================================
        # V9.7 direct root output
        # ====================================================

        u[pitch_i] = np.clip(
            q_pitch,
            adapter.lower[pitch_i],
            adapter.upper[pitch_i],
        )

        u[pip_i] = np.clip(
            q_pip,
            adapter.lower[pip_i],
            adapter.upper[pip_i],
        )

        # ----------------------------------------------------
        # Only used for diagnostic/grip estimate.
        # Normalize against THIS robot finger's semantic curl
        # range, not actuator position.
        # ----------------------------------------------------

        # Human-calibrated closure progress.
        # OPEN=0, FIST=1 for every user's own hand.
        progress = clamp01(
            0.45 * root_t
            +
            0.55 * p_curl
        )

        finger_progress[
            finger
        ] = progress

        semantic_diag[
            finger
        ] = (
            np.degrees(
                human_curl
            ),
            dir_err,
            curl_err,
            np.degrees(
                q_pitch
            ),
            np.degrees(
                q_pip
            ),
        )

    optimizer._v93_semantic_diag = (
        semantic_diag
    )

    # ========================================================
    # Thumb flex
    # ========================================================

    thumb = measurement[
        "thumb"
    ]

    thumb_open = calib[
        "OPEN"
    ]["thumb"]

    thumb_fist = calib[
        "FIST"
    ]["thumb"]

    p_tmcp = norm01(
        thumb["mcp"],
        thumb_open["mcp"],
        thumb_fist["mcp"],
    )

    p_tip = norm01(
        thumb["ip"],
        thumb_open["ip"],
        thumb_fist["ip"],
    )

    p_thumb_flex = clamp01(
        0.65 * p_tmcp
        + 0.35 * p_tip
    )

    tmcp_i = adapter.u_index[
        "thumb_mcp"
    ]

    q0, q1 = robot_flex_range(
        adapter,
        tmcp_i,
    )

    u[tmcp_i] = (
        q0
        + p_thumb_flex
        * (
            q1
            - q0
        )
    )

    # ========================================================
    # Thumb opposition diagnostic
    # ========================================================

    d = measurement[
        "thumb_index_distance"
    ]

    d_open = calib[
        "OPEN"
    ]["thumb_index_distance"]

    d_o = calib[
        "O"
    ]["thumb_index_distance"]

    opposition = norm01(
        d,
        d_open,
        d_o,
    )

    grip = float(
        np.mean(
            list(
                finger_progress.values()
            )
        )
    )

    # ========================================================
    # REAL LIVE THUMB CMC 3DOF
    #
    # No anchor interpolation.
    # Skeleton segment directions directly drive:
    #   roll / yaw / pitch
    # ========================================================

    thumb_cmc = solve_thumb_cmc_live(
        optimizer,
        adapter,
        u,
        measurement,
    )

    # Thumb enters palm region:
    # only the missing pad-orientation DOF is corrected.
    apply_thumb_palm_gate(
        optimizer,
        adapter,
        u,
        kp,
    )

    u = np.clip(
        u,
        adapter.lower,
        adapter.upper,
    )

    # Keep current true L20 state for temporal warm-start.
    optimizer.last_u16 = u.copy()

    q21 = adapter.expand(
        u
    )

    debug = {
        "grip": grip,
        "opposition": opposition,

        "index": finger_progress[
            "index"
        ],

        "middle": finger_progress[
            "middle"
        ],

        "ring": finger_progress[
            "ring"
        ],

        "pinky": finger_progress[
            "pinky"
        ],

        "thumb_cmc": thumb_cmc.copy(),
    }

    return (
        q21,
        debug,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default=(
            "example/config/"
            "l20_feature_retarget_wuji_right.yaml"
        ),
    )

    parser.add_argument(
        "--hz",
        type=float,
        default=90.0,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--hand",
        choices=[
            "right",
            "left",
        ],
        default="right",
    )

    args = parser.parse_args()

    print()
    print("=" * 80)
    print("Skeleton Teleop V9 Calibrated")
    print("SIMULATION ONLY")
    print("=" * 80)
    print(
        "不会启动 ROS / CAN / L20 driver"
    )
    print("=" * 80)

    config_path = (
        PROJECT_DIR
        / args.config
    ).resolve()

    # ========================================================
    # Retarget infrastructure
    # ========================================================

    retargeter = Retargeter.from_yaml(
        str(config_path),
        hand_side=args.hand,
    )

    optimizer = retargeter.optimizer

    adapter = optimizer.adapter

    # ========================================================
    # V9.3 four-finger semantic retargeter
    # ========================================================

    semantic_config_path = (
        PROJECT_DIR
        / "example/config/l20_semantic_v4_wuji_right.yaml"
    ).resolve()

    semantic_retargeter = Retargeter.from_yaml(
        str(semantic_config_path),
        hand_side=args.hand,
    )

    semantic = (
        semantic_retargeter.optimizer
    )

    # Disable its periodic stdout diagnostics.
    semantic.debug_every = 0

    if list(
        semantic.adapter.u_names
    ) != list(
        adapter.u_names
    ):
        raise RuntimeError(
            "Feature / Semantic L20 control order mismatch"
        )

    print(
        "Four fingers:",
        semantic.__class__.__name__,
    )

    print(
        "Optimizer:",
        optimizer.__class__.__name__,
    )

    print(
        "L20 controls:",
        adapter.nu,
    )

    print(
        "L20 q:",
        adapter.nq,
    )

    # ========================================================
    # MJCF
    # ========================================================

    with open(
        config_path,
        "r",
    ) as f:

        cfg = yaml.safe_load(
            f
        )

    mjcf_path = Path(
        cfg["optimizer"][
            "mjcf_path"
        ]
    ).expanduser()

    if not mjcf_path.is_absolute():

        mjcf_path = (
            config_path.parent
            / mjcf_path
        ).resolve()

    print(
        "MJCF:",
        mjcf_path,
    )

    model = mujoco.MjModel.from_xml_path(
        str(mjcf_path)
    )

    sim_data = mujoco.MjData(
        model
    )

    joint_map = build_joint_map(
        model,
        adapter.q_names,
    )

    # ========================================================
    # Wuji Skeleton
    # ========================================================

    glove = WujiGloveDevice(
        hand_side=args.hand,
        device_name="glove",
    )

    key_name = (
        f"{args.hand}_fingers"
    )

    print()
    print(
        "等待 Wuji HandSkeleton..."
    )

    while True:

        d = glove.get_fingers_data()

        kp = d.get(
            key_name
        )

        if kp is not None:

            kp = np.asarray(
                kp
            )

            if kp.shape == (
                21,
                3,
            ):
                break

        time.sleep(
            0.01
        )

    print(
        "Skeleton OK"
    )

    # ========================================================
    # Human calibration
    # ========================================================

    calib = {}
    calib_kp = {}

    (
        calib["OPEN"],
        calib_kp["OPEN"],
    ) = sample_calibration_pose(
        glove,
        key_name,
        args.hand,
        optimizer,
        "1 / OPEN",
        (
            "五指完全自然张开并伸直，"
            "四指自然并拢，拇指自然展开。"
        ),
    )

    (
        calib["FIST"],
        calib_kp["FIST"],
    ) = sample_calibration_pose(
        glove,
        key_name,
        args.hand,
        optimizer,
        "2 / FIST",
        (
            "四指完全握拳，"
            "拇指自然贴近握拳姿态。"
        ),
    )

    (
        calib["O"],
        calib_kp["O"],
    ) = sample_calibration_pose(
        glove,
        key_name,
        args.hand,
        optimizer,
        "3 / O",
        (
            "拇指与食指指尖接触形成 O 型，"
            "其余手指自然伸展。"
        ),
    )

    print_calibration(
        calib
    )

    thumb_anchors = build_thumb_anchors(
        optimizer,
        adapter,
        calib_kp["OPEN"],
        calib_kp["O"],
        calib_kp["FIST"],
    )

    # V9.7 HUMAN-relative MCP root calibration.
    root_calib = build_root_calibration(
        semantic,
        calib_kp,
    )

    # ========================================================
    # Realtime
    # ========================================================

    alpha = float(
        np.clip(
            args.alpha,
            0.01,
            1.0,
        )
    )

    dt_target = (
        1.0
        / max(
            args.hz,
            1.0,
        )
    )

    q_filtered = None

    stats = []

    last_print = (
        time.monotonic()
    )

    print()
    print("=" * 80)
    print("NORMAL V9 SKELETON TELEOP START")
    print("=" * 80)
    print("测试：")
    print("1. OPEN")
    print("2. FIST")
    print("3. 半握拳")
    print("4. 单独弯食指")
    print("5. 食指左右 spread")
    print("6. 拇食指 O")
    print("=" * 80)

    with mujoco.viewer.launch_passive(
        model,
        sim_data,
    ) as viewer:

        while viewer.is_running():

            loop_start = (
                time.monotonic()
            )

            packet = (
                glove.get_fingers_data()
            )

            raw = packet.get(
                key_name
            )

            if raw is None:
                time.sleep(
                    0.001
                )
                continue

            raw = np.asarray(
                raw,
                dtype=np.float64,
            )

            if raw.shape != (
                21,
                3,
            ):
                continue

            if not np.all(
                np.isfinite(raw)
            ):
                continue

            kp = apply_mediapipe_transformations(
                raw,
                args.hand,
            )

            measurement = measure_pose(
                optimizer,
                kp,
            )

            t0 = time.perf_counter()

            (
                q21,
                debug,
            ) = calibrated_retarget(
                optimizer,
                adapter,
                measurement,
                calib,
                thumb_anchors,
                kp,
                semantic,
                root_calib,
            )

            solve_ms = (
                time.perf_counter()
                - t0
            ) * 1000.0

            stats.append(
                solve_ms
            )

            if len(stats) > 500:
                stats.pop(0)

            # optional light output filter
            if q_filtered is None:

                q_filtered = (
                    q21.copy()
                )

            else:

                q_filtered += (
                    alpha
                    * (
                        q21
                        - q_filtered
                    )
                )

            # q21 -> mujoco
            for (
                qi,
                qadr,
                _name,
            ) in joint_map:

                sim_data.qpos[
                    qadr
                ] = q_filtered[
                    qi
                ]

            mujoco.mj_forward(
                model,
                sim_data,
            )

            draw_skeleton(
                viewer.user_scn,
                kp,
            )

            viewer.sync()

            now = time.monotonic()

            if (
                now
                - last_print
                >= 1.0
            ):

                arr = np.asarray(
                    stats
                )

                sdiag = getattr(
                    optimizer,
                    "_v93_semantic_diag",
                    {},
                )

                if sdiag:
                    print(
                        "[SEMANTIC] "
                        + " ".join(
                            (
                                f"{name[0].upper()}:"
                                f"curl={vals[0]:.1f}"
                                f" p={vals[3]:.1f}"
                                f" pip={vals[4]:.1f}"
                                f" e={vals[2]:.1f}"
                            )
                            for name, vals
                            in sdiag.items()
                        ),
                        flush=True,
                    )

                diag = getattr(
                    optimizer,
                    "_v92_finger_diag",
                    {},
                )

                if diag:
                    print(
                        "[CURL RAW] "
                        + " ".join(
                            (
                                f"{name[0].upper()}:"
                                f"{vals[0]:.2f}/"
                                f"{vals[1]:.2f}/"
                                f"{vals[2]:.2f}"
                                f"=>{vals[3]:.2f}"
                            )
                            for name, vals
                            in diag.items()
                        ),
                        flush=True,
                    )

                print(
                    "[V9] "
                    f"grip={debug['grip']:.2f} "
                    f"I={debug['index']:.2f} "
                    f"M={debug['middle']:.2f} "
                    f"R={debug['ring']:.2f} "
                    f"P={debug['pinky']:.2f} "
                    f"opp={debug['opposition']:.2f} "
                    f"thumbCMC="
                    f"{np.degrees(debug['thumb_cmc']).round(1)} "
                    f"calc="
                    f"{np.percentile(arr,50):.3f}ms "
                    f"alpha={alpha:.2f}",
                    flush=True,
                )

                last_print = now

            elapsed = (
                time.monotonic()
                - loop_start
            )

            remain = (
                dt_target
                - elapsed
            )

            if remain > 0:
                time.sleep(
                    remain
                )

    glove.cleanup()


if __name__ == "__main__":
    main()
