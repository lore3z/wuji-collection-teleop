import os
#!/usr/bin/env python3

import argparse
import json
import pickle
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np

import skeleton_teleop_v9_calibrated_sim as sim
import wuji_l20_g20_teleop as packed


ROOT = Path(__file__).resolve().parent

UDP_HOST = "127.0.0.1"
UDP_PORT = 15120


def load_old_verified_bridge_support():
    """
    Reuse ONLY the already verified L20 hardware bridge from the
    existing one-file teleop.

    We do NOT reuse its old retarget algorithm.
    """

    mod = types.ModuleType(
        "v93_l20_bridge_support"
    )

    mod.__file__ = __file__
    mod.__package__ = ""

    source = packed._decode(
        packed._V1_B85
    )

    exec(
        compile(
            source,
            __file__,
            "exec",
        ),
        mod.__dict__,
    )

    # V9.3 already decides the complete hand pose.
    # Do NOT let the old GUI pinch anchor change it.
    mod.GUI_PINCH_ANCHOR_ENABLED = False

    # No extra hardware target viewer.
    # 显示经过 HandCore/raw20 后真正发送给实体手的目标
    mod.HARDWARE_TARGET_VIEWER_ENABLED = False

    return mod


def check_ros_driver():
    cp = subprocess.run(
        [
            "ros2",
            "topic",
            "info",
            "/cb_right_hand_control_cmd",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(cp.stdout)

    text = cp.stdout

    if "Subscription count: 1" not in text:
        raise SystemExit(
            "\nERROR: 没检测到唯一 L20 driver subscriber。\n"
            "先运行 ./l20.sh driver"
        )

    # Before our bridge starts there must be no command publisher.
    if "Publisher count: 0" not in text:
        raise SystemExit(
            "\nERROR: /cb_right_hand_control_cmd 已有 publisher。\n"
            "说明旧 teleop/bridge 还在运行，禁止双重控制。"
        )



# ============================================================
# L20 RETARGET SPACE -> G20 HARDWARE COMMAND SPACE
# ============================================================

L20_TO_G20_THUMB_ROLL_OFFSET = 0.77165


def l20_u16_to_g20_u16(
    adapter,
    u_l20,
):
    """
    Retarget core uses the L20 model coordinates.

    Physical L20 is driven through the official G20 driver
    command convention.

    The confirmed problematic axis is thumb_cmc_roll:

        L20 model : -0.8 .. +0.5 rad
        G20 driver:  0.0 .. +1.39 rad

    G20 URDF also contains a fixed -0.77165 rad frame rotation.

    Use an explicit hardware-boundary zero-coordinate adapter.
    No other retarget logic belongs here.
    """

    u_l20 = np.asarray(
        u_l20,
        dtype=np.float64,
    )

    if u_l20.shape != (16,):
        raise ValueError(
            f"Expected u16 shape (16,), got {u_l20.shape}"
        )

    u_hw = u_l20.copy()

    i = adapter.u_index[
        "thumb_cmc_roll"
    ]

    u_hw[i] = np.clip(
        u_l20[i]
        + L20_TO_G20_THUMB_ROLL_OFFSET,
        0.0,
        1.39,
    )

    return u_hw



def wait_first_skeleton(
    glove,
    key_name,
):
    print("等待 Wuji HandSkeleton...")

    while True:
        d = glove.get_fingers_data()

        kp = d.get(
            key_name
        )

        if kp is not None:
            kp = np.asarray(
                kp,
                dtype=np.float64,
            )

            if (
                kp.shape == (21, 3)
                and
                np.all(np.isfinite(kp))
            ):
                print("Skeleton OK")
                return

        time.sleep(0.01)




# =====================================================================
# V9.4 realtime gap-conditioned pad-to-pad pinch correction
#
# thumb tip  = MediaPipe 4
# index tip  = 8
# middle tip = 12
# ring tip   = 16
# pinky tip  = 20
#
# gap >= 45 mm : pure V9.3
# gap <= 25 mm : full verified pad target
# middle range : smoothstep blend
# =====================================================================


def _v10_apply_grasp_prior(raw_kp, u16):
    """
    Natural grasp/fist controller.

    Learned from:
        skeleton -> grasp latent
        skeleton -> thumb flex

    Important:
        - >=3 curled fingers preempt pinch
        - four finger flexion enhanced
        - thumb pitch/MCP enhanced
        - thumb lateral roll/yaw remain natural
    """

    import os as _os
    import time as _time
    from pathlib import Path as _Path

    global _V10_GRASP_PREEMPT_PINCH

    _V10_GRASP_PREEMPT_PINCH = False

    raw_kp = np.asarray(
        raw_kp,
        dtype=np.float64,
    )

    u16 = np.asarray(
        u16,
        dtype=np.float64,
    )

    if (
        raw_kp.shape != (21, 3)
        or
        u16.shape != (16,)
        or
        not np.all(np.isfinite(raw_kp))
        or
        not np.all(np.isfinite(u16))
    ):
        return u16

    # --------------------------------------------------------
    # Load learned runtime model
    # --------------------------------------------------------

    if not hasattr(
        _v10_apply_grasp_prior,
        "_model",
    ):

        model_path = (
            _Path.home()
            / "wuji_ftp1_collection_v8 (1)"
            / "runtime"
            / "action_prior"
            / "grasp_thumb_runtime_model.npz"
        )

        z = np.load(
            model_path,
            allow_pickle=False,
        )

        _v10_apply_grasp_prior._model = {
            "mu": np.asarray(
                z["feature_mean"],
                dtype=np.float64,
            ),

            "std": np.asarray(
                z["feature_std"],
                dtype=np.float64,
            ),

            "coef": np.asarray(
                z["coef"],
                dtype=np.float64,
            ),

            "bias": np.asarray(
                z["bias"],
                dtype=np.float64,
            ),
        }

        _v10_apply_grasp_prior._grasp_ema = 0.0
        _v10_apply_grasp_prior._thumb_ema = 0.0

        _v10_apply_grasp_prior._active = False
        _v10_apply_grasp_prior._candidate_since = 0.0
        _v10_apply_grasp_prior._release_since = 0.0

        _v10_apply_grasp_prior._assist = 0.0
        _v10_apply_grasp_prior._last_t = (
            _time.monotonic()
        )

        _v10_apply_grasp_prior._frame = 0

        print(
            "[V10 GRASP] learned model loaded",
            model_path,
            flush=True,
        )

    M = _v10_apply_grasp_prior._model


    def _angle(a, b):

        na = float(
            np.linalg.norm(a)
        )

        nb = float(
            np.linalg.norm(b)
        )

        if na < 1e-8 or nb < 1e-8:
            return 0.0

        return float(
            np.arccos(
                np.clip(
                    np.dot(a, b)
                    /
                    (na * nb),
                    -1.0,
                    1.0,
                )
            )
        )


    scale = float(
        np.median(
            [
                np.linalg.norm(
                    raw_kp[i]
                    -
                    raw_kp[0]
                )
                for i in (
                    5,
                    9,
                    13,
                    17,
                )
            ]
        )
    )

    scale = max(
        scale,
        1e-6,
    )

    feat = []


    # --------------------------------------------------------
    # Thumb features
    # --------------------------------------------------------

    tb0 = float(
        np.pi
        -
        _angle(
            raw_kp[0] - raw_kp[1],
            raw_kp[2] - raw_kp[1],
        )
    )

    tb1 = float(
        np.pi
        -
        _angle(
            raw_kp[1] - raw_kp[2],
            raw_kp[3] - raw_kp[2],
        )
    )

    tb2 = float(
        np.pi
        -
        _angle(
            raw_kp[2] - raw_kp[3],
            raw_kp[4] - raw_kp[3],
        )
    )

    feat.extend(
        [
            tb0,
            tb1,
            tb2,

            float(
                np.linalg.norm(
                    raw_kp[4]
                    -
                    raw_kp[0]
                )
                /
                scale
            ),

            float(
                np.linalg.norm(
                    raw_kp[4]
                    -
                    raw_kp[5]
                )
                /
                scale
            ),
        ]
    )


    # --------------------------------------------------------
    # Four-finger features
    # --------------------------------------------------------

    finger_ids = (
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    )

    finger_bend = []

    for mcp, pip, dip, tip in finger_ids:

        b0 = float(
            np.pi
            -
            _angle(
                raw_kp[0] - raw_kp[mcp],
                raw_kp[pip] - raw_kp[mcp],
            )
        )

        b1 = float(
            np.pi
            -
            _angle(
                raw_kp[mcp] - raw_kp[pip],
                raw_kp[dip] - raw_kp[pip],
            )
        )

        b2 = float(
            np.pi
            -
            _angle(
                raw_kp[pip] - raw_kp[dip],
                raw_kp[tip] - raw_kp[dip],
            )
        )

        finger_bend.append(
            float(
                np.mean(
                    [
                        b0,
                        b1,
                        b2,
                    ]
                )
            )
        )

        feat.extend(
            [
                b0,
                b1,
                b2,

                float(
                    np.linalg.norm(
                        raw_kp[tip]
                        -
                        raw_kp[0]
                    )
                    /
                    scale
                ),

                float(
                    np.linalg.norm(
                        raw_kp[tip]
                        -
                        raw_kp[mcp]
                    )
                    /
                    scale
                ),
            ]
        )


    feat = np.asarray(
        feat,
        dtype=np.float64,
    )

    x = (
        feat - M["mu"]
    ) / M["std"]

    pred = (
        x @ M["coef"]
        +
        M["bias"]
    )

    grasp_raw = float(
        pred[0]
    )

    thumb_raw = float(
        np.clip(
            pred[1],
            0.0,
            1.0,
        )
    )


    # --------------------------------------------------------
    # Temporal smoothing
    # --------------------------------------------------------

    now = _time.monotonic()

    last_t = float(
        getattr(
            _v10_apply_grasp_prior,
            "_last_t",
            now,
        )
    )

    dt = float(
        np.clip(
            now - last_t,
            0.0,
            0.1,
        )
    )

    _v10_apply_grasp_prior._last_t = now


    def _lp(old, value, tau):

        if tau <= 1e-6:
            return float(value)

        a = float(
            1.0
            -
            np.exp(
                -dt / tau
            )
        )

        return float(
            old
            +
            a
            *
            (
                value - old
            )
        )


    grasp = _lp(
        float(
            getattr(
                _v10_apply_grasp_prior,
                "_grasp_ema",
                grasp_raw,
            )
        ),
        grasp_raw,
        float(
            _os.environ.get(
                "V10_GRASP_TAU_S",
                "0.08",
            )
        ),
    )

    thumb = _lp(
        float(
            getattr(
                _v10_apply_grasp_prior,
                "_thumb_ema",
                thumb_raw,
            )
        ),
        thumb_raw,
        float(
            _os.environ.get(
                "V10_THUMB_TAU_S",
                "0.10",
            )
        ),
    )

    _v10_apply_grasp_prior._grasp_ema = (
        grasp
    )

    _v10_apply_grasp_prior._thumb_ema = (
        thumb
    )


    # --------------------------------------------------------
    # Fist / grasp gate
    # --------------------------------------------------------

    bend_gate = float(
        _os.environ.get(
            "V10_GRASP_FINGER_BEND_RAD",
            "0.36",
        )
    )

    curled = int(
        sum(
            b >= bend_gate
            for b in finger_bend
        )
    )

    preempt_threshold = float(
        _os.environ.get(
            "V10_GRASP_PREEMPT",
            "0.12",
        )
    )

    enter_threshold = float(
        _os.environ.get(
            "V10_GRASP_ENTER",
            "0.25",
        )
    )

    exit_threshold = float(
        _os.environ.get(
            "V10_GRASP_EXIT",
            "0.05",
        )
    )

    full_threshold = float(
        _os.environ.get(
            "V10_GRASP_FULL",
            "1.00",
        )
    )

    preempt = bool(
        curled >= 3
        and
        grasp >= preempt_threshold
    )

    active = bool(
        getattr(
            _v10_apply_grasp_prior,
            "_active",
            False,
        )
    )


    # --------------------------------------------------------
    # Crucial arbitration:
    # grasp/fist suppresses pinch BEFORE pinch can lock.
    # --------------------------------------------------------

    if preempt or active:

        _V10_GRASP_PREEMPT_PINCH = True

        if hasattr(
            _v94_apply_gap_pinch,
            "_locked_finger",
        ):

            if (
                getattr(
                    _v94_apply_gap_pinch,
                    "_locked_finger",
                    None,
                )
                is not None
            ):

                print(
                    "[V10 GRASP] "
                    "cancel false pinch",
                    flush=True,
                )

            _v94_apply_gap_pinch._locked_finger = None
            _v94_apply_gap_pinch._candidate_finger = None
            _v94_apply_gap_pinch._candidate_since = 0.0


    dwell = float(
        _os.environ.get(
            "V10_GRASP_DWELL_MS",
            "40",
        )
    ) / 1000.0

    release_dwell = float(
        _os.environ.get(
            "V10_GRASP_RELEASE_MS",
            "140",
        )
    ) / 1000.0


    if not active:

        if preempt:

            t0 = float(
                getattr(
                    _v10_apply_grasp_prior,
                    "_candidate_since",
                    0.0,
                )
            )

            if t0 <= 0.0:

                _v10_apply_grasp_prior._candidate_since = (
                    now
                )

            elif (
                now - t0
                >=
                dwell
            ):

                active = True

                _v10_apply_grasp_prior._active = True
                _v10_apply_grasp_prior._release_since = 0.0

                print(
                    "[V10 GRASP] ENTER "
                    f"latent={grasp:+.3f} "
                    f"thumb={thumb:.2f} "
                    f"curled={curled}/4",
                    flush=True,
                )

        else:

            _v10_apply_grasp_prior._candidate_since = (
                0.0
            )

    else:

        release_cond = bool(
            grasp <= exit_threshold
            or
            curled <= 1
        )

        if release_cond:

            t0 = float(
                getattr(
                    _v10_apply_grasp_prior,
                    "_release_since",
                    0.0,
                )
            )

            if t0 <= 0.0:

                _v10_apply_grasp_prior._release_since = (
                    now
                )

            elif (
                now - t0
                >=
                release_dwell
            ):

                active = False

                _v10_apply_grasp_prior._active = False
                _v10_apply_grasp_prior._candidate_since = 0.0
                _v10_apply_grasp_prior._release_since = 0.0

                print(
                    "[V10 GRASP] EXIT",
                    flush=True,
                )

        else:

            _v10_apply_grasp_prior._release_since = (
                0.0
            )


    if active:
        _V10_GRASP_PREEMPT_PINCH = True


    # --------------------------------------------------------
    # Continuous grasp assistance
    # --------------------------------------------------------

    if full_threshold <= enter_threshold:
        full_threshold = (
            enter_threshold
            +
            1e-3
        )

    strength = float(
        np.clip(
            (
                grasp
                -
                enter_threshold
            )
            /
            (
                full_threshold
                -
                enter_threshold
            ),
            0.0,
            1.0,
        )
    )

    strength = (
        strength
        *
        strength
        *
        (
            3.0
            -
            2.0 * strength
        )
    )

    requested = (
        strength
        if active
        else
        0.0
    )

    assist = _lp(
        float(
            getattr(
                _v10_apply_grasp_prior,
                "_assist",
                0.0,
            )
        ),
        requested,
        float(
            _os.environ.get(
                "V10_GRASP_ASSIST_TAU_S",
                "0.10",
            )
        ),
    )

    assist = float(
        np.clip(
            assist,
            0.0,
            1.0,
        )
    )

    _v10_apply_grasp_prior._assist = (
        assist
    )

    if assist <= 1e-4:
        return u16


    out = u16.copy()


    # --------------------------------------------------------
    # Four-finger flexion
    # --------------------------------------------------------

    roots = (
        1,
        4,
        7,
        10,
    )

    pips = (
        2,
        5,
        8,
        11,
    )

    root_max = float(
        _os.environ.get(
            "V10_GRASP_ROOT_MAX",
            "1.33",
        )
    )

    pip_max = float(
        _os.environ.get(
            "V10_GRASP_PIP_MAX",
            "1.75",
        )
    )

    finger_gain = float(
        _os.environ.get(
            "V10_GRASP_FINGER_ASSIST",
            "0.95",
        )
    )


    for k in range(4):

        local = float(
            np.clip(
                (
                    finger_bend[k]
                    -
                    0.16
                )
                /
                0.78,
                0.0,
                1.0,
            )
        )

        local = (
            local
            *
            local
            *
            (
                3.0
                -
                2.0 * local
            )
        )

        g = float(
            assist
            *
            local
            *
            finger_gain
        )

        ri = roots[k]
        pi = pips[k]

        desired_root = max(
            float(out[ri]),
            root_max
            *
            local,
        )

        desired_pip = max(
            float(out[pi]),
            pip_max
            *
            local,
        )

        out[ri] = (
            out[ri]
            +
            g
            *
            (
                desired_root
                -
                out[ri]
            )
        )

        out[pi] = (
            out[pi]
            +
            g
            *
            (
                desired_pip
                -
                out[pi]
            )
        )


    # --------------------------------------------------------
    # Thumb FLEXION only.
    #
    # q12 roll / q13 yaw are deliberately left untouched.
    # This prevents learned grasp from causing lateral sweep.
    # --------------------------------------------------------

    thumb_gain = float(
        _os.environ.get(
            "V10_GRASP_THUMB_ASSIST",
            "0.95",
        )
    )

    tg = float(
        assist
        *
        thumb_gain
    )

    pitch_max = float(
        _os.environ.get(
            "V10_THUMB_PITCH_MAX",
            "0.83",
        )
    )

    mcp_max = float(
        _os.environ.get(
            "V10_THUMB_MCP_MAX",
            "1.25",
        )
    )

    # ========================================================
    # V10.1 thumb tip direct geometric control.
    #
    # Learned thumb value is useful for global grasp posture,
    # but thumb-tip flex needs faster local response.
    #
    # tb1 = human thumb MCP bend
    # tb2 = human thumb IP/distal bend
    #
    # L20 has no independent thumb-IP actuator, therefore q15
    # drives the physical thumb MCP + mimicked distal joint.
    # ========================================================

    thumb_tip_direct = float(
        np.clip(
            (
                0.35 * tb1
                +
                0.65 * tb2
                -
                float(
                    _os.environ.get(
                        "V101_THUMB_TIP_DEADZONE_RAD",
                        "0.06",
                    )
                )
            )
            /
            float(
                _os.environ.get(
                    "V101_THUMB_TIP_FULL_RAD",
                    "0.70",
                )
            ),
            0.0,
            1.0,
        )
    )

    # Smoothstep, but still much more responsive than the
    # previous learned scalar alone.
    thumb_tip_direct = (
        thumb_tip_direct
        *
        thumb_tip_direct
        *
        (
            3.0
            -
            2.0 * thumb_tip_direct
        )
    )

    tip_mix = float(
        np.clip(
            float(
                _os.environ.get(
                    "V101_THUMB_TIP_DIRECT_MIX",
                    "0.80",
                )
            ),
            0.0,
            1.0,
        )
    )

    thumb_tip = float(
        np.clip(
            (
                1.0 - tip_mix
            )
            *
            thumb
            +
            tip_mix
            *
            thumb_tip_direct,
            0.0,
            1.0,
        )
    )

    desired_pitch = max(
        float(out[14]),
        pitch_max
        *
        thumb,
    )

    desired_mcp = max(
        float(out[15]),
        mcp_max
        *
        thumb_tip
        *
        float(
            _os.environ.get(
                "V101_THUMB_TIP_GAIN",
                "1.10",
            )
        ),
    )

    # V11.8:
    # DO NOT use human MCP prediction to drive robot CMC pitch.
    #
    # q14 is thumb_cmc_pitch/root.
    # Human MCP/IP model has no valid supervision for this DOF.
    #
    # Therefore q14 remains exactly the natural retarget result.
    out[14] = float(out[14])


    out[15] = (
        out[15]
        +
        tg
        *
        (
            desired_mcp
            -
            out[15]
        )
    )


    _v10_apply_grasp_prior._frame += 1

    if (
        _v10_apply_grasp_prior._frame
        %
        20
        ==
        0
    ):

        print(
            "[V10 GRASP] "
            f"latent={grasp:+.3f} "
            f"thumb={thumb:.2f} "
            f"tip={thumb_tip:.2f} "
            f"assist={assist:.2f} "
            f"curled={curled}/4 "
            f"q14={out[14]:.2f} "
            f"q15={out[15]:.2f}",
            flush=True,
        )

    return out



def _v116_apply_direct_thumb(raw_kp, u16):
    """
    V11.6 direct thumb MCP/IP runtime model.

    Learned independently:
        skeleton -> human thumb MCP
        skeleton -> human thumb IP

    Robot mapping:
        predicted MCP -> L20 q14 thumb_pitch
        predicted IP  -> L20 q15 thumb_mcp
                         -> physical thumb distal mimic

    q12/q13 remain untouched.
    """

    import os as _os
    import time as _time
    import numpy as _np
    from pathlib import Path as _Path

    raw_kp = _np.asarray(
        raw_kp,
        dtype=_np.float64,
    )

    u16 = _np.asarray(
        u16,
        dtype=_np.float64,
    )

    if (
        raw_kp.shape != (21, 3)
        or
        u16.shape != (16,)
        or
        not _np.all(_np.isfinite(raw_kp))
        or
        not _np.all(_np.isfinite(u16))
    ):
        return u16

    if not hasattr(
        _v116_apply_direct_thumb,
        "_model",
    ):

        path = (
            _Path.home()
            / "wuji_ftp1_collection_v8 (1)"
            / "runtime"
            / "action_prior"
            / "thumb_mcp_ip_runtime_model.npz"
        )

        z = _np.load(
            path,
            allow_pickle=False,
        )

        _v116_apply_direct_thumb._model = {
            "mu": _np.asarray(
                z["feature_mean"],
                dtype=_np.float64,
            ),

            "std": _np.asarray(
                z["feature_std"],
                dtype=_np.float64,
            ),

            "coef": _np.asarray(
                z["coef"],
                dtype=_np.float64,
            ),

            "bias": _np.asarray(
                z["bias"],
                dtype=_np.float64,
            ),
        }

        _v116_apply_direct_thumb._mcp = 0.0
        _v116_apply_direct_thumb._ip = 0.0

        _v116_apply_direct_thumb._last_t = (
            _time.monotonic()
        )

        _v116_apply_direct_thumb._n = 0

        print(
            "[V11.6 THUMB2] "
            "direct MCP/IP model loaded:",
            path,
            flush=True,
        )


    M = _v116_apply_direct_thumb._model


    def angle(a, b):

        na = float(
            _np.linalg.norm(a)
        )

        nb = float(
            _np.linalg.norm(b)
        )

        if na < 1e-8 or nb < 1e-8:
            return 0.0

        return float(
            _np.arccos(
                _np.clip(
                    _np.dot(a, b)
                    /
                    (na * nb),
                    -1.0,
                    1.0,
                )
            )
        )


    scale = float(
        _np.median(
            [
                _np.linalg.norm(
                    raw_kp[i] - raw_kp[0]
                )
                for i in (
                    5,
                    9,
                    13,
                    17,
                )
            ]
        )
    )

    scale = max(
        scale,
        1e-6,
    )

    feat = []

    # ========================================================
    # thumb 5 features
    # ========================================================

    feat += [
        _np.pi - angle(
            raw_kp[0] - raw_kp[1],
            raw_kp[2] - raw_kp[1],
        ),

        _np.pi - angle(
            raw_kp[1] - raw_kp[2],
            raw_kp[3] - raw_kp[2],
        ),

        _np.pi - angle(
            raw_kp[2] - raw_kp[3],
            raw_kp[4] - raw_kp[3],
        ),

        _np.linalg.norm(
            raw_kp[4] - raw_kp[0]
        ) / scale,

        _np.linalg.norm(
            raw_kp[4] - raw_kp[5]
        ) / scale,
    ]


    # ========================================================
    # four fingers 20 features
    # ========================================================

    for mcp, pip, dip, tip in (
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ):

        feat += [
            _np.pi - angle(
                raw_kp[0] - raw_kp[mcp],
                raw_kp[pip] - raw_kp[mcp],
            ),

            _np.pi - angle(
                raw_kp[mcp] - raw_kp[pip],
                raw_kp[dip] - raw_kp[pip],
            ),

            _np.pi - angle(
                raw_kp[pip] - raw_kp[dip],
                raw_kp[tip] - raw_kp[dip],
            ),

            _np.linalg.norm(
                raw_kp[tip] - raw_kp[0]
            ) / scale,

            _np.linalg.norm(
                raw_kp[tip] - raw_kp[mcp]
            ) / scale,
        ]


    feat = _np.asarray(
        feat,
        dtype=_np.float64,
    )


    x = (
        feat - M["mu"]
    ) / M["std"]


    pred = (
        x @ M["coef"]
        +
        M["bias"]
    )


    pred_mcp = float(
        _np.clip(
            pred[0],
            0.0,
            1.0,
        )
    )

    pred_ip = float(
        _np.clip(
            pred[1],
            0.0,
            1.0,
        )
    )


    # ========================================================
    # IP range expansion
    #
    # Validation showed mild regression-to-mean:
    #
    # real 0.08 -> pred about 0.18
    # real 1.00 -> pred about 0.89
    #
    # Expand around 0.5.
    # ========================================================

    ip_expand = float(
        _os.environ.get(
            "V116_IP_EXPAND",
            "1.20",
        )
    )

    pred_ip = float(
        _np.clip(
            0.5
            +
            (
                pred_ip - 0.5
            )
            *
            ip_expand,
            0.0,
            1.0,
        )
    )


    mcp_expand = float(
        _os.environ.get(
            "V116_MCP_EXPAND",
            "1.00",
        )
    )

    pred_mcp = float(
        _np.clip(
            0.5
            +
            (
                pred_mcp - 0.5
            )
            *
            mcp_expand,
            0.0,
            1.0,
        )
    )


    # ========================================================
    # Fast low-pass
    # ========================================================

    now = _time.monotonic()

    last_t = float(
        getattr(
            _v116_apply_direct_thumb,
            "_last_t",
            now,
        )
    )

    dt = float(
        _np.clip(
            now - last_t,
            0.0,
            0.1,
        )
    )

    _v116_apply_direct_thumb._last_t = now


    tau = float(
        _os.environ.get(
            "V116_THUMB_TAU_S",
            "0.045",
        )
    )


    if tau <= 1e-6:

        a = 1.0

    else:

        a = float(
            1.0
            -
            _np.exp(
                -dt / tau
            )
        )


    mcp_f = float(
        getattr(
            _v116_apply_direct_thumb,
            "_mcp",
            pred_mcp,
        )
    )

    ip_f = float(
        getattr(
            _v116_apply_direct_thumb,
            "_ip",
            pred_ip,
        )
    )


    mcp_f += (
        a
        *
        (
            pred_mcp - mcp_f
        )
    )

    ip_f += (
        a
        *
        (
            pred_ip - ip_f
        )
    )


    _v116_apply_direct_thumb._mcp = mcp_f
    _v116_apply_direct_thumb._ip = ip_f


    # ========================================================
    # Human -> L20
    #
    # q14 = thumb pitch/root flex
    # q15 = thumb MCP, with distal mimic
    #
    # q12/q13 untouched.
    # ========================================================

    pitch_max = float(
        _os.environ.get(
            "V116_THUMB_PITCH_MAX",
            "0.83",
        )
    )

    tip_max = float(
        _os.environ.get(
            "V116_THUMB_TIP_MAX",
            "1.25",
        )
    )


    # ========================================================
    # V11.7 calibrated bidirectional thumb-root mapping
    #
    # The learned MCP predictor is accurate, but its runtime
    # output does not necessarily span exactly 0..1.
    #
    # Explicitly remap:
    #
    #   MCP <= OPEN   -> q14 = ROOT_MIN
    #   MCP >= CLOSE  -> q14 = ROOT_MAX
    #
    # This fixes both:
    #   root cannot come up
    #   root cannot go down
    # ========================================================

    root_open = float(
        _os.environ.get(
            "V117_ROOT_MCP_OPEN",
            "0.18",
        )
    )

    root_close = float(
        _os.environ.get(
            "V117_ROOT_MCP_CLOSE",
            "0.82",
        )
    )

    root_min = float(
        _os.environ.get(
            "V117_ROOT_Q14_MIN",
            "0.00",
        )
    )

    root_max = float(
        _os.environ.get(
            "V117_ROOT_Q14_MAX",
            "0.83",
        )
    )

    if root_close <= root_open:
        root_close = root_open + 1e-3

    root_phase = float(
        _np.clip(
            (
                mcp_f - root_open
            )
            /
            (
                root_close - root_open
            ),
            0.0,
            1.0,
        )
    )

    # smoothstep
    root_phase = (
        root_phase
        *
        root_phase
        *
        (
            3.0
            -
            2.0 * root_phase
        )
    )

    target_q14 = (
        root_min
        +
        root_phase
        *
        (
            root_max - root_min
        )
    )

    target_q15 = (
        ip_f
        *
        tip_max
    )


    mcp_blend = float(
        _np.clip(
            float(
                _os.environ.get(
                    "V116_MCP_BLEND",
                    "0.75",
                )
            ),
            0.0,
            1.0,
        )
    )

    ip_blend = float(
        _np.clip(
            float(
                _os.environ.get(
                    "V116_IP_BLEND",
                    "1.00",
                )
            ),
            0.0,
            1.0,
        )
    )


    out = u16.copy()


    out[14] = (
        (
            1.0 - mcp_blend
        )
        *
        out[14]
        +
        mcp_blend
        *
        target_q14
    )


    out[15] = (
        (
            1.0 - ip_blend
        )
        *
        out[15]
        +
        ip_blend
        *
        target_q15
    )


    _v116_apply_direct_thumb._n += 1


    if (
        _v116_apply_direct_thumb._n
        %
        20
        ==
        0
    ):

        print(
            "[V11.6 THUMB2] "
            f"MCP={mcp_f:.3f} "
            f"IP={ip_f:.3f} "
            f"q14={out[14]:.3f} "
            f"q15={out[15]:.3f}",
            flush=True,
        )


    return out



def _v118_apply_thumb_root(raw_kp, u16):
    """
    V11.8 independent thumb-root controller.

    skeleton
        -> learned human thumb_adduction
        -> L20 q14 thumb_cmc_pitch / physical raw[0]

    q12/q13 untouched.
    q15 handled independently by thumb-IP model.
    """

    import os as _os
    import time as _time
    import numpy as _np
    from pathlib import Path as _Path

    raw_kp = _np.asarray(raw_kp, dtype=_np.float64)
    u16 = _np.asarray(u16, dtype=_np.float64)

    if (
        raw_kp.shape != (21, 3)
        or u16.shape != (16,)
        or not _np.all(_np.isfinite(raw_kp))
        or not _np.all(_np.isfinite(u16))
    ):
        return u16

    if not hasattr(_v118_apply_thumb_root, "_model"):

        model_path = (
            _Path.home()
            / "wuji_ftp1_collection_v8 (1)"
            / "runtime"
            / "action_prior"
            / "thumb_root_runtime_model.npz"
        )

        z = _np.load(
            model_path,
            allow_pickle=False,
        )

        _v118_apply_thumb_root._model = {
            "mu": _np.asarray(
                z["feature_mean"],
                dtype=_np.float64,
            ),

            "std": _np.asarray(
                z["feature_std"],
                dtype=_np.float64,
            ),

            "coef": _np.asarray(
                z["coef"],
                dtype=_np.float64,
            ),

            "bias": float(z["bias"]),
        }

        _v118_apply_thumb_root._root = 0.5
        _v118_apply_thumb_root._last_t = time.monotonic()
        _v118_apply_thumb_root._n = 0

        print(
            "[V11.8 ROOT] model loaded:",
            model_path,
            flush=True,
        )

    M = _v118_apply_thumb_root._model


    def angle(a, b):

        na = float(_np.linalg.norm(a))
        nb = float(_np.linalg.norm(b))

        if na < 1e-8 or nb < 1e-8:
            return 0.0

        return float(
            _np.arccos(
                _np.clip(
                    _np.dot(a, b) / (na * nb),
                    -1.0,
                    1.0,
                )
            )
        )


    scale = float(
        _np.median(
            [
                _np.linalg.norm(
                    raw_kp[i] - raw_kp[0]
                )
                for i in (5, 9, 13, 17)
            ]
        )
    )

    scale = max(scale, 1e-6)

    feat = [
        _np.pi - angle(
            raw_kp[0] - raw_kp[1],
            raw_kp[2] - raw_kp[1],
        ),

        _np.pi - angle(
            raw_kp[1] - raw_kp[2],
            raw_kp[3] - raw_kp[2],
        ),

        _np.pi - angle(
            raw_kp[2] - raw_kp[3],
            raw_kp[4] - raw_kp[3],
        ),

        _np.linalg.norm(
            raw_kp[4] - raw_kp[0]
        ) / scale,

        _np.linalg.norm(
            raw_kp[4] - raw_kp[5]
        ) / scale,
    ]


    for mcp, pip, dip, tip in (
        (5, 6, 7, 8),
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ):

        feat += [
            _np.pi - angle(
                raw_kp[0] - raw_kp[mcp],
                raw_kp[pip] - raw_kp[mcp],
            ),

            _np.pi - angle(
                raw_kp[mcp] - raw_kp[pip],
                raw_kp[dip] - raw_kp[pip],
            ),

            _np.pi - angle(
                raw_kp[pip] - raw_kp[dip],
                raw_kp[tip] - raw_kp[dip],
            ),

            _np.linalg.norm(
                raw_kp[tip] - raw_kp[0]
            ) / scale,

            _np.linalg.norm(
                raw_kp[tip] - raw_kp[mcp]
            ) / scale,
        ]


    feat = _np.asarray(
        feat,
        dtype=_np.float64,
    )

    x = (
        feat - M["mu"]
    ) / M["std"]

    pred = float(
        _np.clip(
            x @ M["coef"] + M["bias"],
            0.0,
            1.0,
        )
    )


    # Mild range expansion
    expand = float(
        _os.environ.get(
            "V118_ROOT_EXPAND",
            "1.10",
        )
    )

    pred = float(
        _np.clip(
            0.5
            +
            (pred - 0.5)
            *
            expand,
            0.0,
            1.0,
        )
    )


    # Optional physical direction reversal.
    reverse = int(
        _os.environ.get(
            "V118_ROOT_REVERSE",
            "0",
        )
    )

    if reverse:
        pred = 1.0 - pred


    # Fast low-pass
    now = _time.monotonic()

    last_t = float(
        getattr(
            _v118_apply_thumb_root,
            "_last_t",
            now,
        )
    )

    dt = float(
        _np.clip(
            now - last_t,
            0.0,
            0.1,
        )
    )

    _v118_apply_thumb_root._last_t = now

    tau = float(
        _os.environ.get(
            "V118_ROOT_TAU_S",
            "0.055",
        )
    )

    if tau <= 1e-6:
        a = 1.0
    else:
        a = float(
            1.0
            -
            _np.exp(
                -dt / tau
            )
        )

    root = float(
        getattr(
            _v118_apply_thumb_root,
            "_root",
            pred,
        )
    )

    root += (
        a
        *
        (
            pred - root
        )
    )

    _v118_apply_thumb_root._root = root


    q14_min = float(
        _os.environ.get(
            "V118_ROOT_Q14_MIN",
            "0.00",
        )
    )

    q14_max = float(
        _os.environ.get(
            "V118_ROOT_Q14_MAX",
            "0.83",
        )
    )

    target_q14 = (
        q14_min
        +
        root
        *
        (
            q14_max - q14_min
        )
    )


    blend = float(
        _np.clip(
            float(
                _os.environ.get(
                    "V118_ROOT_BLEND",
                    "1.00",
                )
            ),
            0.0,
            1.0,
        )
    )


    out = u16.copy()

    # ONLY q14.
    out[14] = (
        (
            1.0 - blend
        )
        *
        out[14]
        +
        blend
        *
        target_q14
    )


    _v118_apply_thumb_root._n += 1

    if (
        _v118_apply_thumb_root._n
        %
        20
        ==
        0
    ):

        print(
            "[V11.8 ROOT] "
            f"pred={pred:.3f} "
            f"root={root:.3f} "
            f"q14={out[14]:.3f}",
            flush=True,
        )


    return out


def _v94_apply_gap_pinch(raw_kp, u16):
    """
    V11 PURE NATURAL PINCH.

    Absolutely NO attraction.

    - no target pose blending
    - no PREP
    - no COMMIT
    - no ROOT_ONLY
    - no HOLD
    - no terminal correction
    - no fixed MuJoCo pinch endpoint
    - no pinch sim2real residual request

    Pinch is now entirely produced by the normal Wuji retargeter.

    This function only clears all historical pinch controller state.
    """

    u16 = np.asarray(
        u16,
        dtype=np.float64,
    )

    # ---------------------------------------------------------
    # Completely disable old pinch-controller state.
    # ---------------------------------------------------------

    _v94_apply_gap_pinch._locked_finger = None
    _v94_apply_gap_pinch._candidate_finger = None
    _v94_apply_gap_pinch._candidate_since = 0.0

    _v94_apply_gap_pinch._hw_active_pinch = None

    _v94_apply_gap_pinch._hw_thumb_alpha = 0.0
    _v94_apply_gap_pinch._hw_finger_alpha = 0.0
    _v94_apply_gap_pinch._hw_root_alpha = 0.0

    # Legacy fields
    _v94_apply_gap_pinch._hw_prep_alpha = 0.0
    _v94_apply_gap_pinch._hw_close_alpha = 0.0

    # ---------------------------------------------------------
    # Critical:
    #
    # Return the current natural retarget result unchanged.
    # ---------------------------------------------------------

    return u16


def _v94_expand_u16(u):
    """
    TRUE native L20 u16 -> MuJoCo q21.
    q21 order:
      index4, middle4, pinky4, ring4, thumb5
    """
    u = np.asarray(u, dtype=np.float64)

    if u.shape != (16,):
        raise ValueError(
            f"V9.4 u16 shape错误: {u.shape}"
        )

    r = 0.8079096

    return np.asarray([
        # index
        u[0], u[1], u[2], u[2] * r,

        # middle
        u[3], u[4], u[5], u[5] * r,

        # pinky
        u[9], u[10], u[11], u[11] * r,

        # ring
        u[6], u[7], u[8], u[8] * r,

        # thumb
        u[12], u[13], u[14], u[15], u[15] * r,
    ], dtype=np.float64)


# =====================================================================
# V9.4 REAL-HARDWARE thumb live trim
# ONLY affects physical hand, NOT MuJoCo viewer.
# =====================================================================

_V94_THUMB_TRIM = {
    "thumb_cmc_roll": 0.0,
    "thumb_cmc_yaw": 0.0,
    "thumb_cmc_pitch": 0.0,
    "thumb_mcp": 0.0,
}

_V94_THUMB_TRIM_STEP = 0.02


def _v94_thumb_trim_print():
    vals = [
        _V94_THUMB_TRIM["thumb_cmc_roll"],
        _V94_THUMB_TRIM["thumb_cmc_yaw"],
        _V94_THUMB_TRIM["thumb_cmc_pitch"],
        _V94_THUMB_TRIM["thumb_mcp"],
    ]

    deg = np.degrees(vals)

    print(
        "[THUMB TRIM] "
        f"roll={vals[0]:+.4f} ({deg[0]:+.1f}deg)  "
        f"yaw={vals[1]:+.4f} ({deg[1]:+.1f}deg)  "
        f"pitch={vals[2]:+.4f} ({deg[2]:+.1f}deg)  "
        f"mcp={vals[3]:+.4f} ({deg[3]:+.1f}deg)  "
        f"step={_V94_THUMB_TRIM_STEP:.4f}",
        flush=True,
    )


def _v94_save_thumb_trim(label):
    import json
    from pathlib import Path

    out = (
        Path(__file__).resolve().parent
        / "runtime"
        / "v94_thumb_sim2real_samples.json"
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if out.exists():
        try:
            data = json.loads(out.read_text())
        except Exception:
            data = {}
    else:
        data = {}

    data[label] = {
        k: float(v)
        for k, v in _V94_THUMB_TRIM.items()
    }

    out.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(f"[THUMB TRIM] SAVED {label}")
    print(f"[THUMB TRIM] {out}")
    _v94_thumb_trim_print()


def _v94_start_thumb_trim_keyboard():
    import sys
    import tty
    import termios
    import select
    import threading

    def worker():
        global _V94_THUMB_TRIM_STEP

        fd = sys.stdin.fileno()

        try:
            old = termios.tcgetattr(fd)
        except Exception as e:
            print(
                "[THUMB TRIM] keyboard unavailable:",
                e,
                flush=True,
            )
            return

        try:
            tty.setcbreak(fd)

            print()
            print("=" * 72)
            print("REAL THUMB LIVE TRIM")
            print("=" * 72)
            print("a / d : roll   - / +")
            print("j / l : yaw    - / +")
            print("k / i : pitch  - / +")
            print("u / o : MCP    - / +")
            print("[ / ] : step   /2 / x2")
            print("r     : reset")
            print("p     : print")
            print("1     : save INDEX")
            print("2     : save MIDDLE")
            print("3     : save RING")
            print("4     : save PINKY")
            print("=" * 72)

            _v94_thumb_trim_print()

            while True:

                ready, _, _ = select.select(
                    [sys.stdin],
                    [],
                    [],
                    0.1,
                )

                if not ready:
                    continue

                ch = sys.stdin.read(1)
                step = float(_V94_THUMB_TRIM_STEP)
                changed = False

                if ch == "a":
                    _V94_THUMB_TRIM["thumb_cmc_roll"] -= step
                    changed = True

                elif ch == "d":
                    _V94_THUMB_TRIM["thumb_cmc_roll"] += step
                    changed = True

                elif ch == "j":
                    _V94_THUMB_TRIM["thumb_cmc_yaw"] -= step
                    changed = True

                elif ch == "l":
                    _V94_THUMB_TRIM["thumb_cmc_yaw"] += step
                    changed = True

                elif ch == "k":
                    _V94_THUMB_TRIM["thumb_cmc_pitch"] -= step
                    changed = True

                elif ch == "i":
                    _V94_THUMB_TRIM["thumb_cmc_pitch"] += step
                    changed = True

                elif ch == "u":
                    _V94_THUMB_TRIM["thumb_mcp"] -= step
                    changed = True

                elif ch == "o":
                    _V94_THUMB_TRIM["thumb_mcp"] += step
                    changed = True

                elif ch == "[":
                    _V94_THUMB_TRIM_STEP = max(
                        0.0025,
                        _V94_THUMB_TRIM_STEP / 2.0,
                    )
                    changed = True

                elif ch == "]":
                    _V94_THUMB_TRIM_STEP = min(
                        0.10,
                        _V94_THUMB_TRIM_STEP * 2.0,
                    )
                    changed = True

                elif ch == "r":
                    for key in _V94_THUMB_TRIM:
                        _V94_THUMB_TRIM[key] = 0.0
                    changed = True

                elif ch == "p":
                    _v94_thumb_trim_print()

                elif ch == "1":
                    _v94_save_thumb_trim("index")

                elif ch == "2":
                    _v94_save_thumb_trim("middle")

                elif ch == "3":
                    _v94_save_thumb_trim("ring")

                elif ch == "4":
                    _v94_save_thumb_trim("pinky")

                if changed:
                    for key in _V94_THUMB_TRIM:
                        _V94_THUMB_TRIM[key] = float(
                            np.clip(
                                _V94_THUMB_TRIM[key],
                                -0.60,
                                +0.60,
                            )
                        )

                    _v94_thumb_trim_print()

        finally:
            try:
                termios.tcsetattr(
                    fd,
                    termios.TCSADRAIN,
                    old,
                )
            except Exception:
                pass

    threading.Thread(
        target=worker,
        daemon=True,
    ).start()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hz",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--arm",
        action="store_true",
        help="Actually enable L20 hardware output",
    )

    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Ignore saved OPEN/FIST/O calibration and collect again",
    )


    parser.add_argument(
        "--dual-object-anchor",
        action="store_true",
        help="Enable dual-object grasp anchors",
    )

    parser.add_argument(
        "--anchor-small-enter",
        type=float,
        default=40.0,
    )

    parser.add_argument(
        "--anchor-small-full",
        type=float,
        default=18.0,
    )

    parser.add_argument(
        "--anchor-box-enter",
        type=float,
        default=40.0,
    )

    parser.add_argument(
        "--anchor-box-full",
        type=float,
        default=18.0,
    )

    parser.add_argument(
        "--anchor-reset-far",
        type=float,
        default=85.0,
    )

    parser.add_argument(
        "--anchor-reset-ms",
        type=float,
        default=300.0,
    )

    parser.add_argument(
        "--anchor-debug",
        action="store_true",
    )

    args = parser.parse_args()


    # Dual-object anchor settings inherited by child bridge
    os.environ["L20_DUAL_ANCHOR_ENABLE"] = (
        "1" if args.dual_object_anchor else "0"
    )
    os.environ["L20_DUAL_SMALL_ENTER"] = str(
        args.anchor_small_enter
    )
    os.environ["L20_DUAL_SMALL_FULL"] = str(
        args.anchor_small_full
    )
    os.environ["L20_DUAL_BOX_ENTER"] = str(
        args.anchor_box_enter
    )
    os.environ["L20_DUAL_BOX_FULL"] = str(
        args.anchor_box_full
    )
    os.environ["L20_DUAL_RESET_FAR"] = str(
        args.anchor_reset_far
    )
    os.environ["L20_DUAL_RESET_FRAMES"] = str(
        max(
            1,
            int(args.anchor_reset_ms * 0.001 * 120.0),
        )
    )
    os.environ["L20_DUAL_ANCHOR_DEBUG"] = (
        "1" if args.anchor_debug else "0"
    )

    if args.arm:
        print("[V9.4] MODE = HARDWARE")
    else:
        print("[V9.4] MODE = PREVIEW ONLY")
        print("[V9.4] L20 hardware output = DISABLED")

    hz = float(
        np.clip(
            args.hz,
            10.0,
            120.0,
        )
    )

    print("=" * 76)
    print("V9.3 SKELETON -> L20 REAL HARDWARE")
    print("=" * 76)
    print(f"hardware rate = {hz:.1f} Hz")
    print("old V6/V8 retarget = DISABLED")
    print("GUI pinch anchor   = DISABLED")
    print("V9.3 semantic      = ENABLED")
    print("=" * 76)

    # ========================================================
    # Feature retarget infrastructure
    # Mainly used for:
    #   adapter
    #   live thumb CMC FK solver
    # ========================================================

    feature_cfg = (
        ROOT
        / "example/config/"
        "l20_feature_retarget_wuji_right.yaml"
    ).resolve()

    feature_rt = sim.Retargeter.from_yaml(
        str(feature_cfg),
        hand_side="right",
    )

    optimizer = (
        feature_rt.optimizer
    )

    adapter = (
        optimizer.adapter
    )

    # ========================================================
    # V9.3 Semantic four-finger retarget
    # ========================================================

    semantic_cfg = (
        ROOT
        / "example/config/"
        "l20_semantic_v4_wuji_right.yaml"
    ).resolve()

    semantic_rt = sim.Retargeter.from_yaml(
        str(semantic_cfg),
        hand_side="right",
    )

    semantic = (
        semantic_rt.optimizer
    )

    semantic.debug_every = 0

    if (
        list(semantic.adapter.u_names)
        !=
        list(adapter.u_names)
    ):
        raise RuntimeError(
            "Semantic / Feature u16 order mismatch"
        )

    print()
    print(
        "u16 order:"
    )

    for i, name in enumerate(
        adapter.u_names
    ):
        print(
            f"  {i:2d}: {name}"
        )

    # ========================================================
    # Wuji Skeleton
    # ========================================================

    glove = sim.WujiGloveDevice(
        hand_side="right",
        device_name="glove",
    )

    key_name = "right_fingers"

    wait_first_skeleton(
        glove,
        key_name,
    )

    # ========================================================
    # Persistent OPEN / FIST / O calibration
    # ========================================================

    calib_cache = (
        ROOT
        / "runtime"
        / "v93_skeleton_calib_right.pkl"
    )

    calib_cache.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    calib = None
    calib_kp = None

    # --------------------------------------------------------
    # Normal startup: reuse previous calibration.
    # --------------------------------------------------------

    if (
        calib_cache.exists()
        and
        not args.recalibrate
    ):

        try:

            with open(
                calib_cache,
                "rb",
            ) as f:

                saved = pickle.load(f)

            calib = saved[
                "calib"
            ]

            calib_kp = saved[
                "calib_kp"
            ]

            print()
            print("=" * 76)
            print("加载已有 Skeleton 标定")
            print("cache:", calib_cache)
            print("跳过 OPEN / FIST / O 三姿态采集")
            print("=" * 76)

            sim.print_calibration(
                calib
            )

        except Exception as e:

            print(
                "[CALIB CACHE] 读取失败:",
                repr(e),
            )

            calib = None
            calib_kp = None

    # --------------------------------------------------------
    # No cache / explicit --recalibrate:
    # collect once and save.
    # --------------------------------------------------------

    if (
        calib is None
        or
        calib_kp is None
    ):

        calib = {}
        calib_kp = {}

        (
            calib["OPEN"],
            calib_kp["OPEN"],
        ) = sim.sample_calibration_pose(
            glove,
            key_name,
            "right",
            optimizer,
            "1 / OPEN",
            (
                "五指自然完全张开；"
                "按 Enter 后保持 1.5 秒不动。"
            ),
        )

        (
            calib["FIST"],
            calib_kp["FIST"],
        ) = sim.sample_calibration_pose(
            glove,
            key_name,
            "right",
            optimizer,
            "2 / FIST",
            (
                "四指完整握拳；"
                "按 Enter 后保持 1.5 秒不动。"
            ),
        )

        (
            calib["O"],
            calib_kp["O"],
        ) = sim.sample_calibration_pose(
            glove,
            key_name,
            "right",
            optimizer,
            "3 / O",
            (
                "拇指与食指形成 O；"
                "按 Enter 后保持 1.5 秒不动。"
            ),
        )

        sim.print_calibration(
            calib
        )

        with open(
            calib_cache,
            "wb",
        ) as f:

            pickle.dump(
                {
                    "version": "v93_skeleton_calib_1",
                    "hand": "right",
                    "calib": calib,
                    "calib_kp": calib_kp,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        print()
        print(
            "[CALIB CACHE] 已保存:",
            calib_cache,
        )

    # Kept for compatibility with current V9.3 function signature.
    thumb_anchors = sim.build_thumb_anchors(
        optimizer,
        adapter,
        calib_kp["OPEN"],
        calib_kp["O"],
        calib_kp["FIST"],
    )

    root_calib = sim.build_root_calibration(
        semantic,
        calib_kp,
    )



    print()
    print("=" * 76)
    print("标定完成")
    print("=" * 76)
    print("现在把人手重新完全张开。")
    if args.arm:
        print("确认实体 L20 周围没有物体、手指没有夹住东西。")
        print()

        input(
            "保持 OPEN，按 Enter 开始实机接管..."
        )

        # ====================================================
        # Driver preflight
        # ====================================================

        check_ros_driver()
    else:
        print()
        print("[V9.4] PREVIEW ONLY")
        print("[V9.4] 不检查 ROS hand driver")
        print("[V9.4] 不启动 L20 raw20 bridge")
        print("[V9.4] 不发送 15120 hardware command")
        print("[V9.4] 仅发送 15121 MuJoCo viewer")
        print()

    # ========================================================
    # Start the already verified raw20 hardware bridge
    # ========================================================

    bridge_support = (
        load_old_verified_bridge_support()
    )

    bridge_support.stop_old_bridge()

    # ========================================================
    # V9.4 hardware bridge:
    # use the real static bridge that contains the normalized
    # L20 -> G20 hardware mapping.
    #
    # DO NOT generate the old /tmp/v93_l20_hw_* bridge anymore.
    # ========================================================
    bridge_py_static = (
        Path.home()
        / "linkerhand-telop-ros2"
        / "tools"
        / "l20_wuji_hw_bridge.py"
    )

    if not bridge_py_static.exists():
        raise RuntimeError(
            "Normalized hardware bridge not found: "
            + str(bridge_py_static)
        )

    bridge_proc = None
    sock = None

    try:

        if args.arm:

            bridge_py = bridge_py_static

            print()
            print(
                "启动 NORMALIZED L20 raw20 bridge..."
            )
            print(
                "[V9.4] bridge =",
                bridge_py,
            )

            bridge_proc = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(bridge_py),
                ],
                stdin=subprocess.DEVNULL,
                cwd=str(
                    Path.home()
                    / "linkerhand-telop-ros2"
                ),
                env=dict(
                    __import__("os").environ
                ),
            )

            bridge_support.wait_udp_listener(
                bridge_proc,
                timeout=10.0,
            )

            print(
                "Bridge READY"
            )

            # Start REAL-HARDWARE thumb live trim keyboard
            # Disabled during normalized L20->G20 mapping validation.
            # _v94_start_thumb_trim_keyboard()

        else:
            print("[V9.4] bridge startup skipped")

        sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        addr = (
            UDP_HOST,
            UDP_PORT,
        )

        # Independent MuJoCo viewer mirror.
        # No viewer connected -> UDP packet is simply dropped.
        viz_addr = (
            "127.0.0.1",
            15121,
        )

        viz_counter = 0

        period = (
            1.0
            / hz
        )

        next_tick = (
            time.monotonic()
        )

        count = 0
        stat_t0 = (
            time.monotonic()
        )

        print()
        print("=" * 76)
        print("L20 HARDWARE CONTROL ACTIVE")
        print("Ctrl+C = 停止 V9.3 / bridge")
        print("=" * 76)

        while True:

            packet = (
                glove.get_fingers_data()
            )

            raw = packet.get(
                key_name
            )

            if raw is None:
                time.sleep(0.001)
                continue

            raw = np.asarray(
                raw,
                dtype=np.float64,
            )

            if (
                raw.shape != (21, 3)
                or
                not np.all(np.isfinite(raw))
            ):
                continue

            # Same coordinate transformation as successful sim.
            kp = (
                sim.apply_mediapipe_transformations(
                    raw,
                    "right",
                )
            )

            measurement = (
                sim.measure_pose(
                    optimizer,
                    kp,
                )
            )

            (
                q21,
                debug,
            ) = sim.calibrated_retarget(
                optimizer,
                adapter,
                measurement,
                calib,
                thumb_anchors,
                kp,
                semantic,
                root_calib,
            )

            # =========================================================
            # V9.4 realtime pad-to-pad correction
            # =========================================================
            _v94_u16 = adapter.compress(
                np.asarray(q21, dtype=np.float64)
            )

            # V9.4 DEBUG: print four finger MCP spread/roll
            if count % 30 == 0:
                print(
                    "[ROLL]",
                    f"I={_v94_u16[0]:+.4f}",
                    f"M={_v94_u16[3]:+.4f}",
                    f"R={_v94_u16[6]:+.4f}",
                    f"P={_v94_u16[9]:+.4f}",
                    flush=True,
                )
            # V9.4 finger-spread regularization
            # index / middle / ring / pinky MCP roll
            _v94_u16[[0, 3, 6, 9]] *= 0.35

            _v94_u16 = _v10_apply_grasp_prior(raw, _v94_u16)
            _v94_u16 = _v116_apply_direct_thumb(raw, _v94_u16)
            _v94_u16 = _v118_apply_thumb_root(raw, _v94_u16)
            _v94_u16 = _v94_apply_gap_pinch(
                raw,
                _v94_u16,
            )
            q21 = _v94_expand_u16(_v94_u16)


            # =================================================
            # Independent visualization mirror
            #
            # Send the EXACT V9.3 target used by this hardware
            # controller together with the transformed Wuji Skeleton.
            #
            # Hardware loop = 120 Hz
            # Viewer packets = about 30 Hz
            # =================================================

            viz_counter += 1

            if viz_counter >= 4:

                viz_counter = 0

                try:
                    viz_payload = {
                        "kp": np.asarray(
                            kp,
                            dtype=np.float32,
                        ).tolist(),

                        "q21": np.asarray(
                            q21,
                            dtype=np.float32,
                        ).tolist(),

                        "stamp": time.monotonic(),
                    }

                    sock.sendto(
                        json.dumps(
                            viz_payload,
                            separators=(",", ":"),
                        ).encode("utf-8"),
                        viz_addr,
                    )

                except Exception:
                    pass

            # =================================================
            # q21 -> TRUE native L20 u16
            #
            # This is exactly what the verified hardware bridge
            # expects. No second retargeting.
            # =================================================

            u16 = adapter.compress(
                np.asarray(
                    q21,
                    dtype=np.float64,
                )
            )

            if (
                u16.shape != (16,)
                or
                not np.all(np.isfinite(u16))
            ):
                continue

            # =================================================
            # SAME hardware-coordinate conversion as the
            # previously verified one-file controller.
            #
            # V9.3 / MuJoCo L20:
            #     thumb_cmc_roll may be negative.
            #
            # Physical G20 / HandCore:
            #     thumb roll uses positive hardware coordinates.
            # =================================================

            # IMPORTANT:
            #
            # u16 already contains L20 mechanical joint angles.
            # l20_wuji_hw_bridge.py itself performs:
            #
            #   radians
            #      -> URDF limits
            #      -> trans_to_motor_right()
            #      -> raw20
            #
            # Therefore DO NOT add another thumb-roll offset here.
            # Final L20 retarget target
            # -> official G20 hardware command coordinates.
            # =================================================
            # Keep PURE L20 mechanical coordinates here.
            #
            # L20 -> G20 normalized hardware mapping is now
            # performed centrally inside l20_wuji_hw_bridge.py.
            # Do NOT apply the old thumb-roll +0.77165 offset here.
            # =================================================
            u16_hw = np.asarray(
                u16,
                dtype=np.float64,
            ).copy()

            # =================================================
            # REAL-HARDWARE-ONLY thumb sim2real correction
            #
            # Viewer already received ideal q21 above, so these
            # biases affect ONLY the physical hand.
            # =================================================

            _thumb_real_bias = {
                "thumb_cmc_roll": float(
                    os.environ.get(
                        "V94_REAL_THUMB_ROLL_BIAS",
                        "0.0",
                    )
                ),
                "thumb_cmc_yaw": float(
                    os.environ.get(
                        "V94_REAL_THUMB_YAW_BIAS",
                        "0.0",
                    )
                ),
                "thumb_cmc_pitch": float(
                    os.environ.get(
                        "V94_REAL_THUMB_PITCH_BIAS",
                        "0.0",
                    )
                ),
                "thumb_mcp": float(
                    os.environ.get(
                        "V94_REAL_THUMB_MCP_BIAS",
                        "0.0",
                    )
                ),
            }

            for _name, _bias in _thumb_real_bias.items():
                _i = adapter.u_index[_name]
                u16_hw[_i] += _bias

            # =================================================
            # REAL-HARDWARE thumb live trim APPLY
            # Viewer q21 was already sent above, therefore
            # this modifies ONLY the physical hand.
            # =================================================
            for _trim_name, _trim_bias in _V94_THUMB_TRIM.items():
                _trim_i = adapter.u_index[_trim_name]
                u16_hw[_trim_i] += float(_trim_bias)

            # PURE L20 thumb-roll coordinate range.
            #
            # Hardware-space conversion happens inside
            # l20_wuji_hw_bridge.py.  Do NOT clip here using
            # the G20 range 0.0..1.39.
            _trim_roll_i = adapter.u_index["thumb_cmc_roll"]
            u16_hw[_trim_roll_i] = np.clip(
                u16_hw[_trim_roll_i],
                -0.8,
                0.5,
            )

            sim_thumb_roll = float(
                u16[12]
            )

            hw_thumb_roll = float(
                u16_hw[
                    adapter.u_index[
                        "thumb_cmc_roll"
                    ]
                ]
            )

            payload = {
                "u16":
                    [
                        float(x)
                        for x in u16_hw
                    ],

                "calib_done":
                    True,

                "stamp":
                    float(
                        time.time()
                    ),

                "names":
                    list(
                        adapter.u_names
                    ),

                # Soft-assist gains used by BOTH MuJoCo
                # and physical-hand sim2real.
                "active_pinch":
                    getattr(
                        _v94_apply_gap_pinch,
                        "_hw_active_pinch",
                        None,
                    ),

                "pinch_thumb_strength":
                    float(
                        getattr(
                            _v94_apply_gap_pinch,
                            "_hw_thumb_alpha",
                            0.0,
                        )
                    ),

                "pinch_finger_prep_strength":
                    float(
                        getattr(
                            _v94_apply_gap_pinch,
                            "_hw_finger_alpha",
                            0.0,
                        )
                    ),

                "pinch_root_strength":
                    float(
                        getattr(
                            _v94_apply_gap_pinch,
                            "_hw_root_alpha",
                            0.0,
                        )
                    ),

                # Backward compatibility.
                "pinch_prep_strength":
                    float(
                        getattr(
                            _v94_apply_gap_pinch,
                            "_hw_finger_alpha",
                            0.0,
                        )
                    ),

                "pinch_close_strength":
                    float(
                        getattr(
                            _v94_apply_gap_pinch,
                            "_hw_root_alpha",
                            0.0,
                        )
                    ),

                "pinch_strength":
                    float(
                        getattr(
                            _v94_apply_gap_pinch,
                            "_hw_root_alpha",
                            0.0,
                        )
                    ),
            }

            # =================================================
            # ROOT-ONLY CLOSE metadata.
            # Override legacy pinch fields from previous versions.
            # =================================================
            payload["active_pinch"] = getattr(
                _v94_apply_gap_pinch,
                "_hw_active_pinch",
                None,
            )

            payload["pinch_thumb_strength"] = float(
                getattr(
                    _v94_apply_gap_pinch,
                    "_hw_thumb_alpha",
                    0.0,
                )
            )

            payload["pinch_finger_prep_strength"] = float(
                getattr(
                    _v94_apply_gap_pinch,
                    "_hw_finger_alpha",
                    0.0,
                )
            )

            payload["pinch_root_strength"] = float(
                getattr(
                    _v94_apply_gap_pinch,
                    "_hw_root_alpha",
                    0.0,
                )
            )

            # Legacy fields retained only for compatibility.
            payload["pinch_prep_strength"] = (
                payload["pinch_finger_prep_strength"]
            )

            payload["pinch_close_strength"] = (
                payload["pinch_root_strength"]
            )

            payload["pinch_strength"] = (
                payload["pinch_root_strength"]
            )

            if args.arm:
                sock.sendto(
                    json.dumps(
                        payload,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                    addr,
                )

            count += 1

            now = time.monotonic()

            if (
                now - stat_t0
                >= 1.0
            ):

                actual_hz = (
                    count
                    / (
                        now - stat_t0
                    )
                )

                thumb = np.degrees(
                    debug["thumb_cmc"]
                )

                print(
                    "[V9.3 HW] "
                    f"{actual_hz:5.1f}Hz "
                    f"I={debug['index']:.2f} "
                    f"M={debug['middle']:.2f} "
                    f"R={debug['ring']:.2f} "
                    f"P={debug['pinky']:.2f} "
                    f"thumbSIM="
                    f"{np.round(thumb,1)} "
                    f"rollHW={np.degrees(hw_thumb_roll):.1f}",
                    flush=True,
                )

                count = 0
                stat_t0 = now

            # Fixed-rate deadline scheduler.
            next_tick += period

            sleep_time = (
                next_tick
                - time.monotonic()
            )

            if sleep_time > 0:
                time.sleep(
                    sleep_time
                )
            else:
                # If one iteration overruns, discard accumulated lag.
                next_tick = (
                    time.monotonic()
                )

    except KeyboardInterrupt:

        print()
        print(
            "Ctrl+C: 停止 V9.3 实机控制"
        )

    finally:

        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

        try:
            glove.cleanup()
        except Exception:
            pass

        try:
            bridge_support.stop_proc(
                bridge_proc
            )
        except Exception:
            pass

        try:
            shutil.rmtree(
                tmpdir,
                ignore_errors=True,
            )
        except Exception:
            pass

        print(
            "V9.3 hardware bridge stopped."
        )


if __name__ == "__main__":
    main()
