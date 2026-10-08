#!/usr/bin/env python3

import os
import time
import threading
from datetime import datetime

import cv2
import numpy as np
from wuji_sdk import SdkManager


SN = "WG1KA06260622532"

ROWS = 24
COLS = 31
N = ROWS * COLS

DISPLAY_FPS = 30.0
PRESS_THRESHOLD = 0.05
SCALE = 24

# 重点观察
WATCH_ROWS = [6, 10]

state = {
    "latest": None,
    "latest_host_ts": None,
    "latest_device_ts": np.nan,
    "rx_count": 0,
}

lock = threading.Lock()

record_frames = []
record_host_ts = []
record_device_ts = []


def on_tactile(frame):
    try:
        arr = np.asarray(frame.data, dtype=np.float32).reshape(-1)

        if arr.size != N:
            print(f"[WARN] tactile size={arr.size}, expected={N}")
            return

        arr = arr.reshape(ROWS, COLS)

        host_ts = time.time()

        device_ts = np.nan
        try:
            header = getattr(frame, "header", None)
            if header is not None:
                device_ts = float(getattr(header, "timestamp_us", np.nan))
        except Exception:
            pass

        with lock:
            state["latest"] = arr.copy()
            state["latest_host_ts"] = host_ts
            state["latest_device_ts"] = device_ts
            state["rx_count"] += 1

        # 保存完整 120 Hz RAW 数据
        record_frames.append(arr.copy())
        record_host_ts.append(host_ts)
        record_device_ts.append(device_ts)

    except Exception as e:
        print("[callback error]", e)


def make_heatmap(frame):
    valid = frame >= 0.0

    # 压力 0~1 -> 0~255
    gray = np.zeros((ROWS, COLS), dtype=np.uint8)

    if np.any(valid):
        gray[valid] = np.clip(
            frame[valid] * 255.0,
            0,
            255
        ).astype(np.uint8)

    # Turbo colormap
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

    # 无效 taxel -> 黑色
    color[~valid] = (0, 0, 0)

    # 放大，保持每个 taxel 为一个清晰方格
    color = cv2.resize(
        color,
        (COLS * SCALE, ROWS * SCALE),
        interpolation=cv2.INTER_NEAREST,
    )

    # 画 taxel 网格
    for r in range(ROWS + 1):
        y = min(r * SCALE, color.shape[0] - 1)
        cv2.line(
            color,
            (0, y),
            (color.shape[1] - 1, y),
            (50, 50, 50),
            1
        )

    for c in range(COLS + 1):
        x = min(c * SCALE, color.shape[1] - 1)
        cv2.line(
            color,
            (x, 0),
            (x, color.shape[0] - 1),
            (50, 50, 50),
            1
        )

    # 标记 row 6 / row 10
    for row in WATCH_ROWS:
        y1 = row * SCALE
        y2 = (row + 1) * SCALE - 1

        cv2.rectangle(
            color,
            (0, y1),
            (color.shape[1] - 1, y2),
            (255, 255, 255),
            2
        )

        cv2.putText(
            color,
            f"ROW {row}",
            (5, y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return color, valid


def main():
    print("========================================")
    print("Wuji tactile real-time heatmap")
    print(f"SN: {SN}")
    print("Turbo: blue -> cyan -> yellow -> red")
    print("Black: invalid taxel")
    print("White boxes: row 6 / row 10")
    print("Press Q or ESC to stop and save")
    print("========================================")

    manager = SdkManager.instance()

    print("\n[1] scanning...")
    devices = manager.scan()

    for d in devices:
        try:
            print(
                f"  SN={d.sn} "
                f"type={d.device_type} "
                f"address={d.address}"
            )
        except Exception:
            print(" ", d)

    print("\n[2] connecting...")
    glove = manager.connect(
        sn=SN,
        device_name="tactile_live"
    )

    print(f"[OK] connected: {glove.serial_number}")

    print("\n[3] subscribing tactile...")
    sub = glove.tactile().subscribe_with_callback(
        callback=on_tactile
    )

    # 请求 120 Hz；固件会返回实际生效频率
    try:
        actual_rate = sub.set_rate(120.0)
        print(f"[OK] tactile requested=120 Hz, actual={actual_rate}")
    except Exception as e:
        print(f"[INFO] set_rate skipped: {e}")

    cv2.namedWindow(
        "Wuji Tactile Live",
        cv2.WINDOW_NORMAL
    )

    last_count = 0
    last_rate_time = time.monotonic()
    measured_hz = 0.0

    try:
        while True:
            now = time.monotonic()

            with lock:
                frame = (
                    None
                    if state["latest"] is None
                    else state["latest"].copy()
                )
                total_rx = state["rx_count"]

            if now - last_rate_time >= 1.0:
                dt = now - last_rate_time
                measured_hz = (total_rx - last_count) / dt
                last_count = total_rx
                last_rate_time = now

            if frame is None:
                canvas = np.zeros((300, 700, 3), dtype=np.uint8)

                cv2.putText(
                    canvas,
                    "Waiting for tactile frames...",
                    (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Wuji Tactile Live", canvas)

            else:
                heatmap, valid = make_heatmap(frame)

                values = frame[valid]

                if values.size:
                    max_pressure = float(np.max(values))
                    mean_pressure = float(np.mean(values))
                    active = int(
                        np.sum(values > PRESS_THRESHOLD)
                    )
                else:
                    max_pressure = 0.0
                    mean_pressure = 0.0
                    active = 0

                # row 6
                r6 = frame[6]
                r6_valid = r6 >= 0
                r6_max = (
                    float(np.max(r6[r6_valid]))
                    if np.any(r6_valid)
                    else 0.0
                )

                # row 10
                r10 = frame[10]
                r10_valid = r10 >= 0
                r10_max = (
                    float(np.max(r10[r10_valid]))
                    if np.any(r10_valid)
                    else 0.0
                )

                # 顶部状态栏
                header_h = 90

                canvas = np.zeros(
                    (
                        heatmap.shape[0] + header_h,
                        heatmap.shape[1],
                        3
                    ),
                    dtype=np.uint8
                )

                canvas[header_h:, :] = heatmap

                line1 = (
                    f"RX: {measured_hz:6.1f} Hz   "
                    f"MAX: {max_pressure:.4f}   "
                    f"MEAN: {mean_pressure:.4f}   "
                    f">0.05: {active}"
                )

                line2 = (
                    f"row6 max={r6_max:.4f}     "
                    f"row10 max={r10_max:.4f}"
                )

                cv2.putText(
                    canvas,
                    line1,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.putText(
                    canvas,
                    line2,
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                cv2.imshow(
                    "Wuji Tactile Live",
                    canvas
                )

            key = cv2.waitKey(
                max(1, int(1000 / DISPLAY_FPS))
            ) & 0xFF

            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        pass

    finally:
        print("\nStopping...")

        try:
            sub.close()
        except Exception:
            pass

        try:
            manager.disconnect(device_name="tactile_live")
        except Exception:
            pass

        cv2.destroyAllWindows()

        if record_frames:
            os.makedirs(".runtime", exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            out = (
                f".runtime/"
                f"wuji_tactile_live_{ts}.npz"
            )

            frames = np.asarray(
                record_frames,
                dtype=np.float32
            )

            host_ts = np.asarray(
                record_host_ts,
                dtype=np.float64
            )

            device_ts = np.asarray(
                record_device_ts,
                dtype=np.float64
            )

            valid_mask = frames[0] >= 0

            np.savez(
                out,
                tactile=frames,
                host_timestamp_s=host_ts,
                device_timestamp_us=device_ts,
                valid_mask=valid_mask,
                glove_sn=SN,
            )

            duration = (
                host_ts[-1] - host_ts[0]
                if len(host_ts) > 1
                else 0.0
            )

            hz = (
                (len(frames) - 1) / duration
                if duration > 0
                else 0.0
            )

            print("========================================")
            print("Saved")
            print("========================================")
            print(f"file     : {out}")
            print(f"frames   : {len(frames)}")
            print(f"shape    : {frames.shape}")
            print(f"duration : {duration:.3f} s")
            print(f"rate     : {hz:.2f} Hz")
            print(
                f"valid    : "
                f"{int(valid_mask.sum())}/744"
            )


if __name__ == "__main__":
    main()
