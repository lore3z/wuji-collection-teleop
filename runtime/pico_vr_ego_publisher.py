#!/usr/bin/env python3
"""Publish a PICO headset's live display as a policy-safe eye JPEG ROS stream.

The headset produces a raw H.264 stream through Android ``screenrecord`` over
USB ADB.  ffmpeg decodes each genuine frame, crops one eye and the configured
policy ROI, and JPEG-encodes it for efficient ROS transport. Frames are never
duplicated or interpolated.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from queue import Empty, Full, Queue


def _parse_roi(value: str) -> tuple[int, int, int, int] | None:
    """Parse an eye-local policy ROI as ``x,y,width,height``."""
    text = str(value).strip()
    if not text or text.lower() in {"none", "raw", "full"}:
        return None
    fields = [field.strip() for field in text.replace(":", ",").split(",")]
    if len(fields) != 4:
        raise ValueError("ROI must be x,y,width,height")
    try:
        roi = tuple(int(field) for field in fields)
    except ValueError as exc:
        raise ValueError("ROI must contain four integers") from exc
    if roi[0] < 0 or roi[1] < 0 or roi[2] <= 0 or roi[3] <= 0:
        raise ValueError("ROI x/y must be non-negative and width/height positive")
    return roi


def _eye_video_filter(eye: str, policy_roi: tuple[int, int, int, int] | None) -> str:
    """Build the ffmpeg filter for one eye or a side-by-side stereo pair."""
    if eye == "both":
        roi_filter = ""
        if policy_roi is not None:
            x, y, width, height = policy_roi
            roi_filter = f",crop={width}:{height}:{x}:{y}"
        # Crop each eye independently so the compositor UI/black boundary is
        # excluded from both halves before concatenating genuine pixels.
        return (
            "split=2[left_src][right_src];"
            f"[left_src]crop=iw/2:ih:0:0{roi_filter}[left_eye];"
            f"[right_src]crop=iw/2:ih:iw/2:0{roi_filter}[right_eye];"
            "[left_eye][right_eye]hstack=inputs=2:shortest=1[stereo]"
        )
    eye_x = "0" if eye == "left" else "iw/2"
    video_filter = f"crop=iw/2:ih:{eye_x}:0"
    if policy_roi is not None:
        x, y, width, height = policy_roi
        video_filter += f",crop={width}:{height}:{x}:{y}"
    return video_filter


def _parse_screen_size(value: str) -> tuple[int, int]:
    fields = str(value).strip().lower().split("x")
    if len(fields) != 2 or any(not field.isdigit() for field in fields):
        raise ValueError("screen size must be WIDTHxHEIGHT")
    width, height = (int(field) for field in fields)
    if width < 2 or height < 1 or width % 2:
        raise ValueError("screen width must be positive and even")
    return width, height


def _adb_devices(adb: str) -> list[str]:
    result = subprocess.run(
        [adb, "devices"], check=True, text=True, capture_output=True, timeout=10
    )
    return [
        fields[0]
        for line in result.stdout.splitlines()[1:]
        if len(fields := line.split()) >= 2 and fields[1] == "device"
    ]


def _select_serial(adb: str, requested: str) -> str:
    devices = _adb_devices(adb)
    if requested:
        if requested not in devices:
            raise RuntimeError(f"ADB device {requested!r} not connected; available={devices}")
        return requested
    if len(devices) != 1:
        raise RuntimeError(f"expected exactly one authorized ADB device; found={devices}")
    return devices[0]


def _drain_stderr(pipe, label: str, tail: deque[str]) -> None:
    if pipe is None:
        return
    for raw in iter(pipe.readline, b""):
        line = raw.decode("utf-8", errors="replace").strip()
        if line:
            tail.append(f"{label}: {line}")


def _stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="", help="ADB serial; empty auto-selects the only device")
    parser.add_argument("--topic", default="/pico/ego/image_raw/compressed")
    parser.add_argument("--screen-size", default="1920x960", help="stereo screenrecord resolution")
    parser.add_argument("--bit-rate", default="40M", help="Android H.264 screenrecord bit rate")
    parser.add_argument("--eye", choices=("left", "right", "both"), default="left")
    parser.add_argument("--output-size", default="native", help="eye JPEG WxH, or native")
    parser.add_argument(
        "--policy-roi",
        default="520,180,300,600",
        help="eye-local policy crop x,y,width,height; use raw/none to retain the full eye",
    )
    parser.add_argument("--jpeg-quality", type=int, default=3, help="ffmpeg MJPEG q:v (2 best, 31 worst)")
    parser.add_argument("--fps", type=float, default=0.0, help="maximum published FPS; 0 keeps every decoded frame")
    parser.add_argument(
        "--queue-depth", type=int, default=12,
        help="decoded-frame buffer depth while pacing publication (default: 12)",
    )
    parser.add_argument("--frame-id", default="", help="empty derives pico_vr_<eye>_optical_frame")
    parser.add_argument("--restart-delay", type=float, default=1.0)
    args = parser.parse_args()
    if not 2 <= args.jpeg_quality <= 31:
        parser.error("--jpeg-quality must be in [2, 31]")
    if (0.0 < args.fps < 1.0) or args.fps > 120.0:
        parser.error("--fps must be 0 or in [1, 120]")
    if args.queue_depth < 2 or args.queue_depth > 180:
        parser.error("--queue-depth must be in [2, 180]")
    try:
        policy_roi = _parse_roi(args.policy_roi)
    except ValueError as exc:
        parser.error(f"--policy-roi: {exc}")
    try:
        screen_width, screen_height = _parse_screen_size(args.screen_size)
    except ValueError as exc:
        parser.error(f"--screen-size: {exc}")
    if policy_roi is not None:
        x, y, width, height = policy_roi
        if x + width > screen_width // 2 or y + height > screen_height:
            parser.error("--policy-roi must fit inside one eye of --screen-size")
    for program in ("adb", "ffmpeg"):
        if shutil.which(program) is None:
            parser.error(f"required program not found: {program}")

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import CompressedImage

    serial = _select_serial("adb", args.serial.strip())
    frame_id = args.frame_id.strip() or f"pico_vr_{args.eye}_optical_frame"
    rclpy.init(args=[])
    node = rclpy.create_node("pico_vr_ego_publisher")
    # Camera data must not apply reliable-DDS backpressure to the ADB/ffmpeg
    # reader.  A bounded best-effort queue is the standard ROS sensor profile:
    # a transiently slow subscriber may lose a frame, but cannot stall the
    # source and create a long burst of subsequent timestamp gaps.
    image_qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=180,
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )
    publisher = node.create_publisher(CompressedImage, args.topic, image_qos)
    running = True
    adb_process: subprocess.Popen[bytes] | None = None
    ffmpeg_process: subprocess.Popen[bytes] | None = None

    def stop(_signum=None, _frame=None) -> None:
        nonlocal running
        running = False
        _stop_process(ffmpeg_process)
        _stop_process(adb_process)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        f"[PICO VR] device={serial}, stereo={args.screen_size}, eye={args.eye}, "
        f"output={args.output_size}, policy_roi={policy_roi or 'full-eye'}, "
        f"max_fps={args.fps or 'source'}, topic={args.topic}",
        flush=True,
    )

    frame_count = 0
    report_count = 0
    rate_dropped = 0
    publish_period = 1.0 / args.fps if args.fps > 0.0 else 0.0
    report_started = time.monotonic()
    # Keep enough genuine decoded frames to absorb short USB/ffmpeg scheduling
    # stalls.  The queue still drops the oldest frame when full, so this adds
    # resilience without replaying or synthesizing an image.
    frame_queue: Queue[bytes] = Queue(maxsize=args.queue_depth)

    def publish_jpeg(jpeg: bytes) -> None:
        nonlocal frame_count, report_count, report_started
        message = CompressedImage()
        message.header.stamp = node.get_clock().now().to_msg()
        message.header.frame_id = frame_id
        message.format = "jpeg"
        message.data = jpeg
        publisher.publish(message)
        frame_count += 1
        report_count += 1
        now = time.monotonic()
        if frame_count == 1:
            report_started = now
        elapsed = now - report_started
        if elapsed >= 5.0:
            print(
                f"[PICO VR] live={report_count / elapsed:.1f} FPS, "
                f"frames={frame_count}, rate_dropped={rate_dropped}, jpeg={len(jpeg) / 1024:.0f} KiB",
                flush=True,
            )
            report_count = 0
            report_started = now

    def paced_publish_loop() -> None:
        next_deadline = 0.0
        while running and rclpy.ok():
            try:
                jpeg = frame_queue.get(timeout=0.1)
            except Empty:
                next_deadline = 0.0
                continue
            now = time.monotonic()
            if not next_deadline or next_deadline < now - publish_period:
                next_deadline = now
            delay = next_deadline - now
            if delay > 0.0:
                time.sleep(delay)
            if not running or not rclpy.ok():
                break
            publish_jpeg(jpeg)
            next_deadline += publish_period

    publish_thread = None
    if publish_period:
        publish_thread = threading.Thread(target=paced_publish_loop, daemon=True)
        publish_thread.start()
    while running and rclpy.ok():
        tail: deque[str] = deque(maxlen=12)
        adb_command = [
            "adb", "-s", serial, "exec-out", "screenrecord",
            "--output-format=h264", "--size", args.screen_size,
            "--bit-rate", args.bit_rate, "--time-limit", "0", "-",
        ]
        video_filter = _eye_video_filter(args.eye, policy_roi)
        if args.output_size != "native":
            if "x" not in args.output_size:
                parser.error("--output-size must be native or WxH")
            width, height = args.output_size.lower().split("x", 1)
            if not width.isdigit() or not height.isdigit() or int(width) < 1 or int(height) < 1:
                parser.error("--output-size must be native or positive WxH")
            video_filter += f",scale={width}:{height}:flags=lanczos"
        filter_flag = "-filter_complex" if args.eye == "both" else "-vf"
        ffmpeg_command = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-fflags", "+nobuffer", "-flags", "low_delay",
            "-f", "h264", "-i", "pipe:0", "-an", filter_flag, video_filter,
        ]
        if args.eye == "both":
            ffmpeg_command += ["-map", "[stereo]"]
        ffmpeg_command += [
            # Ubuntu 22.04 ships ffmpeg 4.4, where the equivalent of modern
            # ``-fps_mode passthrough`` is ``-vsync 0``. It emits exactly one
            # JPEG per decoded H.264 frame and does not manufacture frames.
            "-vsync", "0", "-c:v", "mjpeg",
            "-q:v", str(args.jpeg_quality), "-f", "image2pipe", "pipe:1",
        ]
        try:
            adb_process = subprocess.Popen(adb_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if adb_process.stdout is None:
                raise RuntimeError("failed to open ADB stdout")
            ffmpeg_process = subprocess.Popen(
                ffmpeg_command, stdin=adb_process.stdout,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            adb_process.stdout.close()
            threading.Thread(target=_drain_stderr, args=(adb_process.stderr, "adb", tail), daemon=True).start()
            threading.Thread(target=_drain_stderr, args=(ffmpeg_process.stderr, "ffmpeg", tail), daemon=True).start()
            if ffmpeg_process.stdout is None:
                raise RuntimeError("failed to open ffmpeg stdout")

            buffer = bytearray()
            while running and rclpy.ok():
                chunk = os.read(ffmpeg_process.stdout.fileno(), 262144)
                if not chunk:
                    break
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        if len(buffer) > 2:
                            del buffer[:-2]
                        break
                    end = buffer.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        if start:
                            del buffer[:start]
                        if len(buffer) > 16 * 1024 * 1024:
                            raise RuntimeError("unterminated JPEG exceeded 16 MiB")
                        break
                    jpeg = bytes(buffer[start:end + 2])
                    del buffer[:end + 2]
                    if publish_period:
                        try:
                            frame_queue.put_nowait(jpeg)
                        except Full:
                            # Stay near live time: replace the oldest queued
                            # genuine frame. The publisher never repeats a
                            # frame when the decoder temporarily runs dry.
                            try:
                                frame_queue.get_nowait()
                            except Empty:
                                pass
                            rate_dropped += 1
                            frame_queue.put_nowait(jpeg)
                    else:
                        publish_jpeg(jpeg)
        except Exception as exc:
            tail.append(str(exc))
        finally:
            _stop_process(ffmpeg_process)
            _stop_process(adb_process)
            ffmpeg_process = adb_process = None
        if running and rclpy.ok():
            print(f"[PICO VR] stream ended; restarting in {args.restart_delay:.1f}s", file=sys.stderr)
            for line in tail:
                print(f"  {line}", file=sys.stderr)
            time.sleep(args.restart_delay)

    stop()
    if publish_thread is not None:
        publish_thread.join(timeout=1.0)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
