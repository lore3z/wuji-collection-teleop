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

def _v94_apply_gap_pinch(raw_kp, u16):
    """
    Smooth assistive pinch state machine.

    FREE:
        pure Wuji control

    ANTICIPATE 90->70 mm:
        only thumb slowly learns toward target
        maximum assistance is small

    PREP 70->55 mm:
        thumb + target finger roll/PIP prepare
        operator still has substantial control

    COMMIT 55->45 mm:
        smoothly converge thumb/finger orientation to
        exact pinch preparation pose

    CLOSE 45->25 mm:
        thumb orientation LOCKED
        target roll/PIP LOCKED
        ONLY target MCP/root continues closing

    HOLD <=25 mm:
        exact verified MuJoCo target

    Reverse opening from 25->45 mm:
        ONLY root opens.
        Thumb does NOT sweep sideways.
    """

    import os as _os
    import time as _time
    from pathlib import Path as _Path

    raw_kp = np.asarray(raw_kp, dtype=np.float64)
    u16 = np.asarray(u16, dtype=np.float64)

    # Hardware metadata
    _v94_apply_gap_pinch._hw_active_pinch = None
    _v94_apply_gap_pinch._hw_thumb_alpha = 0.0
    _v94_apply_gap_pinch._hw_finger_alpha = 0.0
    _v94_apply_gap_pinch._hw_root_alpha = 0.0

    if raw_kp.shape != (21, 3):
        return u16

    if u16.shape != (16,):
        return u16

    if not np.all(np.isfinite(raw_kp)):
        return u16

    if not np.all(np.isfinite(u16)):
        return u16

    # =========================================================
    # Load verified MuJoCo contact targets
    # =========================================================
    if not hasattr(_v94_apply_gap_pinch, "_targets"):

        target_path = _Path(
            _os.environ.get(
                "V94_PAD_TARGETS",
                str(
                    _Path.home()
                    / "GeoRT"
                    / "data"
                    / "thumb_all_fingers_manual_pad_IK.npz"
                ),
            )
        ).expanduser()

        z = np.load(
            target_path,
            allow_pickle=False,
        )

        targets = np.stack(
            [
                z["q_index"],
                z["q_middle"],
                z["q_ring"],
                z["q_pinky"],
            ],
            axis=0,
        ).astype(np.float64)

        if targets.shape != (4, 16):
            raise RuntimeError(
                f"bad V9.4 target shape: {targets.shape}"
            )

        _v94_apply_gap_pinch._targets = targets

        _v94_apply_gap_pinch._locked_finger = None
        _v94_apply_gap_pinch._candidate_finger = None
        _v94_apply_gap_pinch._candidate_since = 0.0

        _v94_apply_gap_pinch._assist_thumb = 0.0
        _v94_apply_gap_pinch._assist_finger = 0.0
        _v94_apply_gap_pinch._assist_root = 0.0
        _v94_apply_gap_pinch._assist_last_t = _time.monotonic()

        _v94_apply_gap_pinch._last_phase = None

        print(
            "[V9.4 ROOT-CLOSE] loaded:",
            target_path,
            flush=True,
        )

    targets = _v94_apply_gap_pinch._targets

    finger_names = (
        "index",
        "middle",
        "ring",
        "pinky",
    )

    # =========================================================
    # Human fingertip gaps
    # =========================================================
    thumb = raw_kp[4]

    tip_ids = (
        8,
        12,
        16,
        20,
    )

    gaps = np.asarray(
        [
            np.linalg.norm(
                thumb - raw_kp[k]
            )
            for k in tip_ids
        ],
        dtype=np.float64,
    )

    now = _time.monotonic()

    # =========================================================
    # Parameters
    # =========================================================
    DETECT_ON = float(
        _os.environ.get(
            "V94_PINCH_DETECT_ON_MM",
            "90",
        )
    ) / 1000.0

    PREP_ON = float(
        _os.environ.get(
            "V94_PINCH_PREP_ON_MM",
            "70",
        )
    ) / 1000.0

    COMMIT_ON = float(
        _os.environ.get(
            "V94_PINCH_COMMIT_ON_MM",
            "55",
        )
    ) / 1000.0

    ROOT_ONLY_ON = float(
        _os.environ.get(
            "V94_PINCH_ROOT_ONLY_ON_MM",
            "45",
        )
    ) / 1000.0

    CLOSE_DONE = float(
        _os.environ.get(
            "V94_PINCH_CLOSE_DONE_MM",
            "25",
        )
    ) / 1000.0

    RELEASE = float(
        _os.environ.get(
            "V94_PINCH_RELEASE_MM",
            "100",
        )
    ) / 1000.0

    DWELL_S = float(
        _os.environ.get(
            "V94_PINCH_LOCK_DWELL_MS",
            "120",
        )
    ) / 1000.0

    DOMINANCE = float(
        _os.environ.get(
            "V94_PINCH_DOMINANCE_MM",
            "4",
        )
    ) / 1000.0

    ANTICIPATE_MAX = float(
        _os.environ.get(
            "V94_PINCH_ANTICIPATE_GAIN",
            "0.15",
        )
    )

    PREP_MAX = float(
        _os.environ.get(
            "V94_PINCH_PREP_GAIN",
            "0.65",
        )
    )

    ROOT_PREBEND = float(
        _os.environ.get(
            "V94_PINCH_ROOT_PREBEND",
            "0.20",
        )
    )

    SLEW_TAU = float(
        _os.environ.get(
            "V94_PINCH_ASSIST_SLEW_S",
            "0.18",
        )
    )

    ANTICIPATE_MAX = float(
        np.clip(ANTICIPATE_MAX, 0.0, 0.4)
    )

    PREP_MAX = float(
        np.clip(PREP_MAX, ANTICIPATE_MAX, 0.9)
    )

    ROOT_PREBEND = float(
        np.clip(ROOT_PREBEND, 0.0, 0.5)
    )

    if not (
        RELEASE
        >
        DETECT_ON
        >
        PREP_ON
        >
        COMMIT_ON
        >
        ROOT_ONLY_ON
        >
        CLOSE_DONE
    ):
        raise RuntimeError(
            "invalid pinch thresholds"
        )

    def smooth01(x):
        x = float(
            np.clip(x, 0.0, 1.0)
        )
        return float(
            x * x * (3.0 - 2.0 * x)
        )

    def desc(g, far_g, near_g):

        if g >= far_g:
            return 0.0

        if g <= near_g:
            return 1.0

        return smooth01(
            (far_g - g)
            /
            (far_g - near_g)
        )

    # =========================================================
    # Stable target identification
    # =========================================================
    locked = getattr(
        _v94_apply_gap_pinch,
        "_locked_finger",
        None,
    )

    nearest = int(
        np.argmin(gaps)
    )

    order = np.argsort(gaps)

    nearest_gap = float(
        gaps[order[0]]
    )

    second_gap = float(
        gaps[order[1]]
    )

    dominance_ok = (
        second_gap
        -
        nearest_gap
        >=
        DOMINANCE
    )

    if locked is None:

        eligible = (
            nearest_gap <= DETECT_ON
            and
            (
                dominance_ok
                or
                nearest_gap <= PREP_ON
            )
        )

        if not eligible:

            _v94_apply_gap_pinch._candidate_finger = None
            _v94_apply_gap_pinch._candidate_since = 0.0

            return u16

        candidate = getattr(
            _v94_apply_gap_pinch,
            "_candidate_finger",
            None,
        )

        if candidate != nearest:

            _v94_apply_gap_pinch._candidate_finger = (
                nearest
            )

            _v94_apply_gap_pinch._candidate_since = (
                now
            )

            return u16

        candidate_since = float(
            getattr(
                _v94_apply_gap_pinch,
                "_candidate_since",
                now,
            )
        )

        stable_s = (
            now
            -
            candidate_since
        )

        if stable_s < DWELL_S:
            return u16

        locked = nearest

        _v94_apply_gap_pinch._locked_finger = (
            locked
        )

        print(
            "[V9.4 ROOT-CLOSE] "
            f"LOCK {finger_names[locked]} "
            f"gap={gaps[locked]*1000:.1f}mm",
            flush=True,
        )

    g = float(
        gaps[locked]
    )

    # =========================================================
    # Requested assist gains
    # =========================================================
    if g >= RELEASE:

        _v94_apply_gap_pinch._locked_finger = None
        _v94_apply_gap_pinch._candidate_finger = None
        _v94_apply_gap_pinch._candidate_since = 0.0

        _v94_apply_gap_pinch._assist_thumb = 0.0
        _v94_apply_gap_pinch._assist_finger = 0.0
        _v94_apply_gap_pinch._assist_root = 0.0

        print(
            "[V9.4 ROOT-CLOSE] RELEASE",
            flush=True,
        )

        return u16

    if g > PREP_ON:
        # 90 -> 70
        phase = "ANTICIPATE"

        x = desc(
            g,
            DETECT_ON,
            PREP_ON,
        )

        thumb_req = (
            ANTICIPATE_MAX
            *
            x
        )

        finger_req = 0.0
        root_req = 0.0

    elif g > COMMIT_ON:
        # 70 -> 55
        phase = "PREP"

        x = desc(
            g,
            PREP_ON,
            COMMIT_ON,
        )

        thumb_req = (
            ANTICIPATE_MAX
            +
            (
                PREP_MAX
                -
                ANTICIPATE_MAX
            )
            *
            x
        )

        finger_req = (
            PREP_MAX
            *
            x
        )

        root_req = (
            ROOT_PREBEND
            *
            x
        )

    elif g > ROOT_ONLY_ON:
        # 55 -> 45
        #
        # Smoothly commit thumb + finger orientation to 1.0.
        # Root remains only lightly pre-bent.
        phase = "COMMIT"

        x = desc(
            g,
            COMMIT_ON,
            ROOT_ONLY_ON,
        )

        thumb_req = (
            PREP_MAX
            +
            (1.0 - PREP_MAX)
            *
            x
        )

        finger_req = (
            PREP_MAX
            +
            (1.0 - PREP_MAX)
            *
            x
        )

        root_req = (
            ROOT_PREBEND
        )

    elif g > CLOSE_DONE:
        # =====================================================
        # CRITICAL:
        #
        # 45 -> 25 mm:
        # thumb = completely locked
        # finger roll/PIP = completely locked
        # ONLY MCP/root moves
        #
        # Same rule also holds while reopening 25 -> 45 mm.
        # =====================================================
        phase = "ROOT_ONLY"

        thumb_req = 1.0
        finger_req = 1.0

        x = desc(
            g,
            ROOT_ONLY_ON,
            CLOSE_DONE,
        )

        root_req = (
            ROOT_PREBEND
            +
            (
                1.0
                -
                ROOT_PREBEND
            )
            *
            x
        )

    else:
        phase = "HOLD"

        thumb_req = 1.0
        finger_req = 1.0
        root_req = 1.0

    # =========================================================
    # Smooth gain transitions.
    #
    # Prevents a newly recognized finger from snapping toward
    # its target in a single frame.
    # =========================================================
    last_t = float(
        getattr(
            _v94_apply_gap_pinch,
            "_assist_last_t",
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

    _v94_apply_gap_pinch._assist_last_t = now

    if SLEW_TAU <= 1e-6:
        beta = 1.0
    else:
        beta = float(
            1.0
            -
            np.exp(
                -dt / SLEW_TAU
            )
        )

    def slew(attr, target):

        old = float(
            getattr(
                _v94_apply_gap_pinch,
                attr,
                0.0,
            )
        )

        new = (
            old
            +
            beta
            *
            (
                float(target)
                -
                old
            )
        )

        new = float(
            np.clip(
                new,
                0.0,
                1.0,
            )
        )

        setattr(
            _v94_apply_gap_pinch,
            attr,
            new,
        )

        return new

    thumb_gain = slew(
        "_assist_thumb",
        thumb_req,
    )

    finger_gain = slew(
        "_assist_finger",
        finger_req,
    )

    root_gain = slew(
        "_assist_root",
        root_req,
    )

    target = np.asarray(
        targets[locked],
        dtype=np.float64,
    )

    base = locked * 3

    roll_i = base + 0
    root_i = base + 1
    pip_i = base + 2

    out = u16.copy()

    # =========================================================
    # Always blend from CURRENT Wuji command.
    #
    # Until gains reach 1, operator control remains active.
    # =========================================================
    out[12:16] = (
        (1.0 - thumb_gain)
        *
        u16[12:16]
        +
        thumb_gain
        *
        target[12:16]
    )

    out[roll_i] = (
        (1.0 - finger_gain)
        *
        u16[roll_i]
        +
        finger_gain
        *
        target[roll_i]
    )

    out[pip_i] = (
        (1.0 - finger_gain)
        *
        u16[pip_i]
        +
        finger_gain
        *
        target[pip_i]
    )

    out[root_i] = (
        (1.0 - root_gain)
        *
        u16[root_i]
        +
        root_gain
        *
        target[root_i]
    )

    # Hardware uses exactly the same trajectory phases.
    _v94_apply_gap_pinch._hw_active_pinch = (
        finger_names[locked]
    )

    _v94_apply_gap_pinch._hw_thumb_alpha = (
        thumb_gain
    )

    _v94_apply_gap_pinch._hw_finger_alpha = (
        finger_gain
    )

    _v94_apply_gap_pinch._hw_root_alpha = (
        root_gain
    )

    last_phase = getattr(
        _v94_apply_gap_pinch,
        "_last_phase",
        None,
    )

    if phase != last_phase:

        print(
            "[V9.4 ROOT-CLOSE] "
            f"{finger_names[locked]} "
            f"phase={phase} "
            f"gap={g*1000:.1f}mm "
            f"thumb={thumb_gain:.3f} "
            f"finger={finger_gain:.3f} "
            f"root={root_gain:.3f}",
            flush=True,
        )

        _v94_apply_gap_pinch._last_phase = (
            phase
        )

    return out


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
