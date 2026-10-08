"""Decoder must emit live frames without waiting for future frames or EOF."""
import re
import selectors
import shutil
import subprocess
import time
import numpy as np
import pytest
from pico_raw_camera_publisher import decoder_command


def test_decoder_emits_jpeg_while_input_is_still_open():
    if not shutil.which('ffmpeg'):
        pytest.skip('FFmpeg unavailable')
    frames = np.random.default_rng(1).integers(0, 256, (8, 96, 128, 3), dtype=np.uint8)
    encoded = subprocess.run([
        'ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '128x96', '-r', '60', '-i', 'pipe:0', '-c:v', 'libx264',
        '-threads', '1', '-preset', 'ultrafast', '-tune', 'zerolatency',
        '-x264-params', 'aud=1:keyint=1', '-pix_fmt', 'yuv420p', '-f', 'h264', 'pipe:1'
    ], input=frames.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    starts = [m.start() for m in re.finditer(b'\x00\x00\x00\x01\x09', encoded)]
    assert len(starts) == 8
    process = subprocess.Popen(decoder_command(2), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        # Feed only four frames. The other four and EOF are deliberately withheld.
        process.stdin.write(encoded[:starts[4]])
        process.stdin.flush()
        data = bytearray()
        deadline = time.monotonic() + 2
        while b'\xff\xd9' not in data and time.monotonic() < deadline:
            if selector.select(timeout=.1):
                data.extend(process.stdout.read1(65536))
        assert data.startswith(b'\xff\xd8') and b'\xff\xd9' in data
    finally:
        selector.close()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
