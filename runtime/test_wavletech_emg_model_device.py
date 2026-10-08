from __future__ import annotations

import sys
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))

from wavletech_emg_model_device import BlockMeanDecimator, SampleClock


def frame(seq: int, arrival_ns: int) -> tuple[int, int, int, np.ndarray, int]:
    return arrival_ns, seq, 0, np.full(8, seq, dtype=np.int32), arrival_ns


def test_block_mean_decimator_outputs_200_hz_values_and_clock() -> None:
    clock = SampleClock()
    decimator = BlockMeanDecimator()
    base = 1_000_000_000
    rows = decimator.feed(
        [frame(i, base + i * 500_000) for i in range(20)], clock
    )
    assert len(rows) == 2
    np.testing.assert_allclose(rows[0][1], np.full(8, 4.5, dtype=np.float32))
    np.testing.assert_allclose(rows[1][1], np.full(8, 14.5, dtype=np.float32))
    assert rows[1][0] - rows[0][0] == 5_000_000


def test_decimator_keeps_partial_block_across_calls() -> None:
    clock = SampleClock()
    decimator = BlockMeanDecimator()
    base = 2_000_000_000
    assert decimator.feed([frame(i, base + i * 500_000) for i in range(6)], clock) == []
    rows = decimator.feed([frame(i, base + i * 500_000) for i in range(6, 10)], clock)
    assert len(rows) == 1
    np.testing.assert_allclose(rows[0][1], 4.5)


def test_sample_clock_smooths_batched_arrivals() -> None:
    clock = SampleClock()
    arrivals = [1_000_000_000] * 10 + [1_005_000_000] * 10
    timestamps = [clock.timestamp(value) for value in arrivals]
    steps = np.diff(timestamps)
    assert np.all((steps >= 490_000) & (steps <= 510_000))


def test_wavletech_launcher_help_exposes_safety_contract() -> None:
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_wavletech_emg_model_hand.sh"), "--help"],
        cwd=ROOT, text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0
    assert "--confirm-wavletech-model" in result.stdout
    assert "--emg-source" in result.stdout


def test_wavletech_confirmation_flag_does_not_require_second_prompt() -> None:
    launcher = (ROOT / "scripts/run_emg_model_hand.sh").read_text(encoding="utf-8")
    assert "Type WAVLETECH to ARM" not in launcher
    assert "--confirm-wavletech-model supplied; starting real-hand control" in launcher
