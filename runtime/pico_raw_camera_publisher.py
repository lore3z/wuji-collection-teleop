#!/usr/bin/env python3
"""Publish PICO's native stereo camera stream without rendering the headset UI.

XRoboToolkit exposes the PICO camera through its Android ``CameraHandle``.  The
headset receives an ``OPEN_CAMERA`` request on its length-framed control socket
and connects back with length-framed H.264 access units.  This bridge decodes
those genuine frames and publishes them as ROS ``CompressedImage`` messages.
"""

from __future__ import annotations

import argparse
import ipaddress
import socket
import struct
import signal
import subprocess
import threading
import time
from typing import Callable, Iterator, TypeVar
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue


MAX_PACKET_BYTES = 16 * 1024 * 1024
T = TypeVar("T")


def decoder_command(quality: int) -> list[str]:
    """Avoid FFmpeg probe and frame-thread queues on the live camera path."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-fflags", "nobuffer",
        "-flags", "low_delay", "-threads", "1", "-probesize", "32",
        "-analyzeduration", "0", "-f", "h264", "-i", "pipe:0",
        "-an", "-f", "mjpeg", "-q:v", str(quality),
        "-threads", "1", "-flush_packets", "1", "pipe:1",
    ]


def put_latest(queue: Queue[T], item: T) -> int:
    """Enqueue *item* without ever allowing old camera frames to add latency."""
    dropped = 0
    while True:
        try:
            queue.put_nowait(item)
            return dropped
        except Full:
            try:
                queue.get_nowait()
                dropped += 1
            except Empty:
                # Another consumer freed the slot between Full and get_nowait.
                continue


def _compact_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    if len(raw) > 255:
        raise ValueError("camera request strings are limited to 255 UTF-8 bytes")
    return bytes((len(raw),)) + raw


def serialize_camera_request(
    *, width: int, height: int, fps: int, bitrate: int, ip: str, port: int
) -> bytes:
    """Serialize XRoboToolkit's CAFE camera request payload."""
    if not 2 <= width <= 8192 or not 1 <= height <= 8192:
        raise ValueError("camera dimensions are outside the supported range")
    if not 1 <= fps <= 120 or not 1 <= bitrate <= 100_000_000:
        raise ValueError("invalid camera fps or bitrate")
    if not 1 <= port <= 65535:
        raise ValueError("invalid stream port")
    # CameraRequestSerializer writes all integer fields as little-endian.
    return (
        b"\xca\xfe\x01"
        + struct.pack("<7i", width, height, fps, bitrate, 0, 2, port)
        + _compact_string("VR")
        + _compact_string(ip)
    )


def serialize_open_camera(
    *, width: int, height: int, fps: int, bitrate: int, ip: str, port: int
) -> bytes:
    """Build the outer big-endian framed ``OPEN_CAMERA`` control packet."""
    camera_data = serialize_camera_request(
        width=width, height=height, fps=fps, bitrate=bitrate, ip=ip, port=port
    )
    command = b"OPEN_CAMERA"
    body = struct.pack("<i", len(command)) + command
    body += struct.pack("<i", len(camera_data)) + camera_data
    return struct.pack(">I", len(body)) + body


def serialize_close_camera() -> bytes:
    command = b"CLOSE_CAMERA"
    body = struct.pack("<i", len(command)) + command + struct.pack("<i", 0)
    return struct.pack(">I", len(body)) + body


def request_camera_reset(pico_ip: str, control_port: int, *, settle: float = 0.0) -> None:
    """Ask the headset to stop any encoder left by a previous operator."""
    try:
        with socket.create_connection((pico_ip, control_port), timeout=3.0) as control:
            control.sendall(serialize_close_camera())
            if settle > 0:
                time.sleep(settle)
    except OSError:
        # A reset is best effort. The following OPEN_CAMERA attempt provides
        # the useful connectivity error if the control service is unavailable.
        return


def _read_exact(stream: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("PICO camera stream closed")
        chunks.extend(chunk)
    return bytes(chunks)


def iter_h264_access_units(stream: socket.socket) -> Iterator[bytes]:
    """Yield one native H.264 packet at a time from the PICO TCP stream."""
    while True:
        length = struct.unpack(">I", _read_exact(stream, 4))[0]
        if length == 0:
            continue
        if length > MAX_PACKET_BYTES:
            raise ValueError(f"PICO H.264 packet is too large: {length} bytes")
        yield _read_exact(stream, length)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("0.0.0.0", 0))
        return int(probe.getsockname()[1])


def _pc_service_endpoint(service_port: int = 63901) -> tuple[str, str] | None:
    """Return (headset, host) from an established XRoboToolkit connection.

    This is more authoritative than subnet probing and remains valid when a
    VPN/TUN policy route makes arbitrary private addresses appear reachable.
    """
    try:
        output = subprocess.run(
            ["ss", "-Htn", "state", "established", "sport", "=", f":{service_port}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None

    def host(endpoint: str) -> str:
        if endpoint.startswith("[::ffff:"):
            return endpoint[len("[::ffff:") :].split("]:", 1)[0]
        return endpoint.rsplit(":", 1)[0]

    endpoints: list[tuple[str, str]] = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            local = host(fields[-2])
            peer = host(fields[-1])
            if ipaddress.ip_address(local).version == 4 and ipaddress.ip_address(peer).version == 4:
                endpoints.append((peer, local))
        except ValueError:
            continue
    if not endpoints:
        return None
    # PC-Service can retain an old routed/TUN session briefly after the PICO
    # rejoins Wi-Fi. Prefer the peer sharing the host's LAN /16 so ss output
    # ordering cannot send OPEN_CAMERA to the stale tunnel endpoint.
    endpoints.sort(
        key=lambda pair: pair[0].split(".")[:2] == pair[1].split(".")[:2],
        reverse=True,
    )
    return endpoints[0]


def _discover_pico_ip(control_port: int) -> tuple[str, str | None]:
    """Find a headset advertising the XRoboToolkit control socket."""
    import subprocess

    service_endpoint = _pc_service_endpoint()
    if service_endpoint is not None:
        return service_endpoint

    candidates: set[str] = set()
    neighbor_candidates: list[str] = []
    # DHCP reassigns the PICO after a reboot.  Probe recently observed LAN
    # neighbors first so discovery normally takes milliseconds instead of
    # scanning an entire /17 corporate Wi-Fi subnet.
    try:
        neighbor_output = subprocess.run(
            ["ip", "-4", "neigh", "show"], capture_output=True, text=True, check=True
        ).stdout
        for line in neighbor_output.splitlines():
            fields = line.split()
            if fields and fields[0] not in neighbor_candidates:
                neighbor_candidates.append(fields[0])
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        output = subprocess.run(
            ["ip", "-4", "addr", "show"], capture_output=True, text=True, check=True
        ).stdout
        for line in output.splitlines():
            if "inet " in line:
                cidr = line.split()[1]
                network = ipaddress.ip_network(cidr, strict=False)
                # Avoid probing loopback, Docker, and very large corporate
                # networks; the headset is on a normal private LAN.
                if network.is_private and network.num_addresses <= 65_536:
                    candidates.update(str(host) for host in network.hosts())
    except (OSError, subprocess.CalledProcessError, IndexError):
        pass
    if not candidates:
        raise ConnectionError("cannot discover local IPv4 subnets; set WUJI_VR_EGO_PICO_IP")

    def probe(address: str) -> str | None:
        try:
            with socket.create_connection((address, control_port), timeout=0.12):
                return address, None
        except OSError:
            return None

    with ThreadPoolExecutor(max_workers=32) as executor:
        for address in executor.map(probe, neighbor_candidates):
            if address:
                return address

    candidates.difference_update(neighbor_candidates)
    with ThreadPoolExecutor(max_workers=128) as executor:
        for address in executor.map(probe, sorted(candidates)):
            if address:
                return address, None
    raise ConnectionError(f"no PICO control socket found on port {control_port}")


def _parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(field) for field in value.lower().split("x"))
    except (TypeError, ValueError) as exc:
        raise ValueError("size must be WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise ValueError("size must be positive")
    return width, height


def publish_decoded_frames(
    frames: Queue[tuple[int, int, bytes]],
    stop: threading.Event,
    publish: Callable[[tuple[int, int, bytes]], None],
) -> None:
    """Forward decoded frames immediately, preserving their observed timestamps.

    The camera already paces acquisition. A second nominal-FPS pacer retains
    burst backlog indefinitely and adds latency when the device runs faster.
    Decode completion is an observation time, not an exposure-time estimate.
    """
    while not stop.is_set():
        try:
            frame = frames.get(timeout=0.25)
        except Empty:
            continue
        if not stop.is_set():
            publish(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pico-ip", required=True, help="PICO headset IPv4 address, or auto")
    parser.add_argument("--control-port", type=int, default=13579)
    parser.add_argument(
        "--stream-port",
        type=int,
        default=50081,
        help="callback TCP port; this PICO build reliably uses 50081",
    )
    parser.add_argument("--size", default="2160x810", help="total side-by-side stream size")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--bitrate", type=int, default=20_000_000)
    parser.add_argument("--topic", default="/pico/ego/image_raw/compressed")
    parser.add_argument("--callback-ip", default="auto", help="host IP advertised for PICO stream callback")
    parser.add_argument("--frame-id", default="pico_vst_optical_frame")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--queue-depth",
        type=int,
        default=6,
        help="small latest-frame JPEG queue; oldest frames are dropped on overflow",
    )
    parser.add_argument("--codec", choices=("auto", "h264", "hevc"), default="auto")
    parser.add_argument("--restart-delay", type=float, default=1.0)
    parser.add_argument(
        "--session-reset-delay",
        type=float,
        default=0.75,
        help="seconds to let CLOSE_CAMERA reset a stale headset encoder before OPEN_CAMERA",
    )
    args = parser.parse_args()
    try:
        width, height = _parse_size(args.size)
    except ValueError as exc:
        parser.error(str(exc))
    if width % 2 or not 1 <= args.fps <= 120:
        parser.error("stereo width must be even and fps must be in [1,120]")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("jpeg quality must be in [1,100]")
    if not 2 <= args.queue_depth <= 600:
        parser.error("queue depth must be in [2,600]")
    if args.stream_port < 0 or args.stream_port > 65535:
        parser.error("stream port must be in [0,65535]")

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import CompressedImage

    rclpy.init(args=[])
    node = rclpy.create_node("pico_raw_camera_publisher")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=min(max(args.queue_depth, 12), 240),
        reliability=ReliabilityPolicy.BEST_EFFORT,
    )
    publisher = node.create_publisher(CompressedImage, args.topic, qos)
    running = True
    control: socket.socket | None = None
    listener: socket.socket | None = None

    def _close_socket(sock: socket.socket, *, reset: bool = False) -> None:
        """Close a socket, optionally aborting a failed control session.

        The PICO app keeps its accepted control socket until it observes EOF.
        On a camera timeout it can otherwise leave the host in FIN-WAIT-2 and
        exhaust the app's small accept backlog after a few retries.
        """
        if reset:
            try:
                sock.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_LINGER,
                    struct.pack("ii", 1, 0),
                )
            except OSError:
                pass
        else:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            sock.close()
        except OSError:
            pass

    def stop(_signum=None, _frame=None) -> None:
        nonlocal running
        running = False
        if control is not None:
            _close_socket(control, reset=True)
        if listener is not None:
            _close_socket(listener)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(
        f"[PICO RAW] headset={args.pico_ip}:{args.control_port}, "
        f"stream={width}x{height}@{args.fps}, topic={args.topic}",
        flush=True,
    )

    try:
        while running:
            try:
                stream_port = args.stream_port or _find_free_port()
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("0.0.0.0", stream_port))
                listener.listen(1)
                listener.settimeout(15.0)
                pico_ip = args.pico_ip
                discovered_callback_ip = None
                if pico_ip.lower() == "auto":
                    pico_ip, discovered_callback_ip = _discover_pico_ip(args.control_port)
                    print(
                        f"[PICO RAW] discovered headset at {pico_ip} from active PC-Service session",
                        flush=True,
                    )
                request_camera_reset(
                    pico_ip,
                    args.control_port,
                    settle=max(args.session_reset_delay, 0.0),
                )
                try:
                    control = socket.create_connection((pico_ip, args.control_port), timeout=5.0)
                except OSError as exc:
                    raise ConnectionError(
                        f"PICO control socket {pico_ip}:{args.control_port} unavailable: {exc}"
                    ) from exc
                control.settimeout(15.0)
                if args.callback_ip.lower() == "auto":
                    local_ip = discovered_callback_ip or control.getsockname()[0]
                else:
                    local_ip = args.callback_ip
                control.sendall(
                    serialize_open_camera(
                        width=width,
                        height=height,
                        fps=args.fps,
                        bitrate=args.bitrate,
                        ip=local_ip,
                        port=stream_port,
                    )
                )
                print(
                    f"[PICO RAW] OPEN_CAMERA sent, callback={local_ip}:{stream_port}, "
                    f"waiting on 0.0.0.0:{stream_port}",
                    flush=True,
                )
                try:
                    stream, peer = listener.accept()
                except socket.timeout:
                    raise TimeoutError(
                        f"PICO did not connect the camera stream to {local_ip}:{stream_port} "
                        "within 15 seconds; XRoboToolkit CameraHandle may be stuck -- "
                        "restart the XRoboToolkit app/headset if repeated CLOSE/OPEN attempts fail"
                    )
                stream.settimeout(5.0)
                print(f"[PICO RAW] stream connected from {peer[0]}:{peer[1]}", flush=True)
                if args.codec == "hevc":
                    raise ValueError("PICO CameraHandle currently emits H.264; --codec=hevc is unsupported")
                # Keep only a short latest-frame window. Old camera pixels are
                # worse than dropped frames because publication-time ROS stamps
                # would make stale content appear synchronized.
                jpeg_q: Queue[tuple[int, int, bytes]] = Queue(maxsize=args.queue_depth)
                publish_stop = threading.Event()
                first_frame = threading.Event()
                frame_count = 0
                decoded_count = 0
                publish_started = 0.0
                dropped_frames = 0
                quality = max(2, min(31, round(31 - args.jpeg_quality * 29 / 100)))
                decoder = subprocess.Popen(
                    decoder_command(quality),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )

                def read_jpegs() -> None:
                    nonlocal dropped_frames, decoded_count
                    pending = bytearray()
                    while decoder.stdout is not None:
                        # ``read(64 KiB)`` waits for a large block and returns
                        # multiple JPEGs in a burst. The publisher previously
                        # mistook the second frame in that normal burst for
                        # stale backlog, effectively reducing 60 Hz to 30 Hz.
                        # ``read1`` returns currently available pipe data and
                        # preserves the decoder's frame cadence.
                        chunk = decoder.stdout.read1(64 * 1024)
                        if not chunk:
                            break
                        pending.extend(chunk)
                        while True:
                            start = pending.find(b"\xff\xd8")
                            if start < 0:
                                if len(pending) > 1:
                                    del pending[:-1]
                                break
                            end = pending.find(b"\xff\xd9", start + 2)
                            if end < 0:
                                if start:
                                    del pending[:start]
                                break
                            jpeg = bytes(pending[start : end + 2])
                            del pending[: end + 2]
                            if running and not publish_stop.is_set():
                                # Camera data is useful only while it is fresh.
                                # Never backpressure FFmpeg/TCP with a full
                                # queue: that retains stale pixels and later
                                # assigns them a new ROS publication stamp.
                                decoded_count += 1
                                dropped_frames += put_latest(
                                    jpeg_q, (decoded_count, time.time_ns(), jpeg)
                                )

                decoder_reader = threading.Thread(target=read_jpegs, daemon=True)
                decoder_reader.start()

                def publish_jpeg(frame: tuple[int, int, bytes]) -> None:
                    nonlocal frame_count, publish_started
                    _decoded_seq, decoded_wall_ns, jpeg = frame
                    message = CompressedImage()
                    message.header.stamp.sec = decoded_wall_ns // 1_000_000_000
                    message.header.stamp.nanosec = decoded_wall_ns % 1_000_000_000
                    message.header.frame_id = args.frame_id
                    message.format = "jpeg"
                    message.data = jpeg
                    publisher.publish(message)
                    first_frame.set()
                    frame_count += 1
                    if frame_count == 1:
                        publish_started = time.monotonic()
                    if frame_count == 1 or frame_count % 60 == 0:
                        elapsed = max(time.monotonic() - publish_started, 1e-6)
                        published_rate = (frame_count - 1) / elapsed
                        print(
                            f"[PICO RAW] frames={frame_count} rate={published_rate:.1f}Hz "
                            f"jpeg_bytes={len(jpeg)} queue={jpeg_q.qsize()} "
                            f"decode_age_ms={(time.time_ns() - decoded_wall_ns) / 1e6:.1f} "
                            f"dropped_stale={dropped_frames}",
                            flush=True,
                        )

                publisher_thread = threading.Thread(
                    target=publish_decoded_frames,
                    args=(jpeg_q, publish_stop, publish_jpeg),
                    daemon=True,
                )
                publisher_thread.start()
                try:
                    first_frame_deadline = time.monotonic() + 8.0
                    for access_unit in iter_h264_access_units(stream):
                        if not running:
                            break
                        if not first_frame.is_set() and time.monotonic() >= first_frame_deadline:
                            raise TimeoutError(
                                "PICO stream connected but no decodable frame arrived within 8 seconds"
                            )
                        if decoder.stdin is None:
                            raise ConnectionError("FFmpeg decoder stdin is closed")
                        decoder.stdin.write(access_unit)
                        decoder.stdin.flush()
                finally:
                    publish_stop.set()
                    if decoder.stdin is not None:
                        decoder.stdin.close()
                    decoder.wait(timeout=3)
                    decoder_reader.join(timeout=1)
                    publisher_thread.join(timeout=3)
                    stream.close()
                for sock in (control, listener):
                    if sock is not None:
                        _close_socket(sock)
                control = None
                listener = None
            except (ConnectionError, OSError, TimeoutError, ValueError) as exc:
                if running:
                    print(f"[PICO RAW] session failed: {exc}; retrying", flush=True)
                    time.sleep(max(args.restart_delay, 0.1))
                if control is not None:
                    try:
                        control.sendall(serialize_close_camera())
                        time.sleep(0.15)
                    except OSError:
                        pass
                if control is not None:
                    _close_socket(control)
                if listener is not None:
                    _close_socket(listener)
                control = None
                listener = None
    finally:
        stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
