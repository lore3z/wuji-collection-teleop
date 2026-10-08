"""Three-state safety gate for the fake single-axis rehearsal."""

from enum import Enum


class MotionState(str, Enum):
    DISARMED = "DISARMED"
    ACTIVE = "ACTIVE"
    FAULT = "FAULT"


class MotionSafetyGate:
    def __init__(self):
        self.state = MotionState.DISARMED
        self.deadman = False
        self.fault_reason = ""

    def set_deadman(self, pressed):
        self.deadman = bool(pressed)
        if not self.deadman and self.state is MotionState.ACTIVE:
            self.state = MotionState.DISARMED

    def activate_if_ready(self, data_fresh):
        if self.state is MotionState.DISARMED and self.deadman and data_fresh:
            self.state = MotionState.ACTIVE
        return self.state is MotionState.ACTIVE

    def trip(self, reason):
        self.state = MotionState.FAULT
        self.fault_reason = str(reason)

    def reset(self):
        if self.deadman:
            raise RuntimeError("release deadman before resetting FAULT")
        self.state = MotionState.DISARMED
        self.fault_reason = ""

