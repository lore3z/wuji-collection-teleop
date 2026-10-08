import socket
import struct
from unittest.mock import patch

from pico_raw_camera_publisher import (
    _pc_service_endpoint,
    publish_decoded_frames,
    iter_h264_access_units,
    put_latest,
    serialize_camera_request,
    serialize_close_camera,
    serialize_open_camera,
)


def test_pc_service_endpoint_uses_active_ipv4_mapped_session():
    output = "0 0 [::ffff:10.191.219.66]:63901 [::ffff:10.32.216.89]:41702\n"
    with patch("pico_raw_camera_publisher.subprocess.run") as run:
        run.return_value.stdout = output
        assert _pc_service_endpoint() == ("10.32.216.89", "10.191.219.66")


def test_pc_service_endpoint_accepts_plain_ipv4_ss_output():
    output = "0 0 10.191.219.66:63901 10.191.132.89:41702\n"
    with patch("pico_raw_camera_publisher.subprocess.run") as run:
        run.return_value.stdout = output
        assert _pc_service_endpoint() == ("10.191.132.89", "10.191.219.66")


def test_pc_service_endpoint_prefers_lan_peer_over_stale_tunnel_session():
    output = (
        "0 0 [::ffff:10.191.219.66]:63901 [::ffff:10.32.216.89]:41702\n"
        "0 0 [::ffff:10.191.219.66]:63901 [::ffff:10.191.208.82]:51984\n"
    )
    with patch("pico_raw_camera_publisher.subprocess.run") as run:
        run.return_value.stdout = output
        assert _pc_service_endpoint() == ("10.191.208.82", "10.191.219.66")


def test_put_latest_drops_oldest_without_blocking():
    from queue import Queue

    queue = Queue(maxsize=2)
    assert put_latest(queue, "oldest") == 0
    assert put_latest(queue, "middle") == 0
    assert put_latest(queue, "latest") == 1
    assert queue.get_nowait() == "middle"
    assert queue.get_nowait() == "latest"


def test_publish_drains_burst_without_pacing_or_timestamp_smoothing():
    from queue import Queue
    from threading import Event

    queue, stop = Queue(), Event()
    frames = [(1, 1_000_000_000, b'a'), (2, 1_001_000_000, b'b'),
              (4, 1_200_000_000, b'c')]
    for frame in frames:
        queue.put_nowait(frame)
    received = []

    def publish(frame):
        received.append(frame)
        if len(received) == len(frames):
            stop.set()

    with patch('pico_raw_camera_publisher.time.sleep', side_effect=AssertionError('pacing')):
        publish_decoded_frames(queue, stop, publish)
    assert received == frames
    assert queue.empty()


def test_publish_does_not_replay_backlog_after_stop():
    from queue import Queue
    from threading import Event

    queue, stop = Queue(), Event()
    queue.put_nowait((1, 1_000_000_000, b'a'))
    stop.set()
    received = []
    publish_decoded_frames(queue, stop, received.append)
    assert received == []


def test_camera_request_matches_unity_serializer_layout():
    payload = serialize_camera_request(
        width=2160, height=810, fps=60, bitrate=20_000_000, ip="192.168.1.5", port=45678
    )
    assert payload[:3] == b"\xca\xfe\x01"
    assert struct.unpack_from("<7i", payload, 3) == (2160, 810, 60, 20_000_000, 0, 2, 45678)
    assert payload[31:34] == b"\x02VR"


def test_open_and_close_commands_are_big_endian_framed():
    open_packet = serialize_open_camera(
        width=1280, height=480, fps=30, bitrate=1_000_000, ip="10.0.0.2", port=13580
    )
    assert struct.unpack(">I", open_packet[:4])[0] == len(open_packet) - 4
    assert open_packet[4:8] == struct.pack("<i", len(b"OPEN_CAMERA"))
    close_packet = serialize_close_camera()
    assert struct.unpack(">I", close_packet[:4])[0] == len(close_packet) - 4


def test_h264_access_units_read_exactly_with_partial_socket_writes():
    left, right = socket.socketpair()
    try:
        packet = struct.pack(">I", 3) + b"abc" + struct.pack(">I", 2) + b"de"
        right.sendall(packet)
        assert next(iter_h264_access_units(left)) == b"abc"
        assert next(iter_h264_access_units(left)) == b"de"
    finally:
        left.close()
        right.close()
