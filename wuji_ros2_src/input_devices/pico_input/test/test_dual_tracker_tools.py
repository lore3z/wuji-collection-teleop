import pytest

from pico_input.pico_dual_tracker_check import collect_synchronized_frames
from pico_input.pico_dual_tracker_pose_publisher import (
    select_dual_tracker_poses,
)
from pico_input.pico_tracker_pose_publisher import TrackerSelectionError


POSE_A = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
POSE_B = [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0]


def test_selects_two_named_trackers_from_one_frame():
    wrist, upper = select_dual_tracker_poses(
        [POSE_B, POSE_A], [b"UPPER", b"WRIST"], "WRIST", "UPPER")
    assert wrist[0] == "WRIST"
    assert upper[0] == "UPPER"
    assert wrist[1].tolist() == pytest.approx(POSE_A[:3])
    assert upper[1].tolist() == pytest.approx(POSE_B[:3])


@pytest.mark.parametrize("wrist,upper,error", [
    ("", "UPPER", "both required"),
    ("SAME", "SAME", "must be different"),
    ("WRIST", "MISSING", "not found"),
])
def test_rejects_invalid_dual_tracker_selection(wrist, upper, error):
    with pytest.raises(TrackerSelectionError, match=error):
        select_dual_tracker_poses(
            [POSE_A, POSE_B], ["WRIST", "UPPER"], wrist, upper)


class FakeClient:
    def __init__(self, serial_frames):
        self.serial_frames = list(serial_frames)
        self.index = 0

    def init(self):
        return True

    def get_motion_tracker_pose(self):
        return [POSE_A, POSE_B][:len(self.serial_frames[self.index])]

    def get_motion_tracker_serial_numbers(self):
        frame = self.serial_frames[self.index]
        if self.index < len(self.serial_frames) - 1:
            self.index += 1
        return frame


def test_quiet_check_requires_consecutive_paired_frames():
    client = FakeClient([
        ["WRIST"],
        ["WRIST", "UPPER"],
        ["WRIST"],
        ["WRIST", "UPPER"],
        ["WRIST", "UPPER"],
    ])
    assert collect_synchronized_frames(
        client, "WRIST", "UPPER", required_frames=2,
        timeout_sec=1.0, poll_rate_hz=1000.0) == 2


def test_quiet_check_rejects_pc_service_connection_failure():
    class DisconnectedClient:
        @staticmethod
        def init():
            return False

    with pytest.raises(RuntimeError, match="PC-Service"):
        collect_synchronized_frames(
            DisconnectedClient(), "WRIST", "UPPER",
            required_frames=1, timeout_sec=0.1)
