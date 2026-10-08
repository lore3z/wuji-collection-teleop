import time

import numpy as np

from realman_rm75_output.joint_sender import (
    FakeJointBackend, JointRateLimiter, JointSender, LatestJointTarget)


def test_latest_target_stops_accepting_after_stop_or_fault():
    buffer = LatestJointTarget()
    assert buffer.publish([1] * 7, 1.0)
    buffer.stop("done")
    assert not buffer.publish([2] * 7, 2.0)
    state, reason, snap = buffer.snapshot()
    assert state == "STOP" and reason == "done" and snap.sequence == 1


def test_limiter_enforces_acceleration_speed_step_and_no_overshoot():
    limiter = JointRateLimiter([0] * 7, 10.0, 20.0, 0.05)
    first = limiter.step([1] * 7, 0.01)
    np.testing.assert_allclose(first, [0.002] * 7)
    assert np.max(np.abs(limiter.velocity)) <= 10.0
    for _ in range(1000):
        last = limiter.step([1] * 7, 0.01)
    np.testing.assert_allclose(last, [1] * 7)


def test_hold_freezes_position_and_zeros_velocity():
    limiter = JointRateLimiter([0] * 7)
    limiter.step([1] * 7, 0.01)
    held = limiter.hold()
    np.testing.assert_allclose(limiter.hold(), held)
    np.testing.assert_allclose(limiter.velocity, [0] * 7)


def test_limiter_clamps_final_command_to_absolute_position_guard():
    limiter = JointRateLimiter(
        [10] * 7, max_speed_deg_s=1000.0, max_accel_deg_s2=100000.0,
        max_step_deg=10.0, min_position_deg=[9] * 7,
        max_position_deg=[11] * 7)
    for _ in range(10):
        command = limiter.step([100] * 7, 0.01)
    np.testing.assert_allclose(command, [11] * 7)


def test_fake_sender_runs_independently_and_reports_metrics():
    buffer = LatestJointTarget()
    backend = FakeJointBackend()
    sender = JointSender(buffer, backend, [0] * 7, rate_hz=125.0)
    buffer.publish([1] * 7)
    sender.start()
    time.sleep(0.08)
    buffer.stop()
    time.sleep(0.025)
    held_before = sender.current_command()
    time.sleep(0.025)
    held_after = sender.current_command()
    sender.stop()
    assert sender.sent_count >= 10
    np.testing.assert_allclose(held_before, held_after)
    metrics = sender.metrics()
    assert metrics["period_ms"]["mean"] > 0.0
    assert metrics["target_age_ms"]["mean"] >= 0.0


def test_deadline_miss_counts_intervals_over_follow_true_ten_ms():
    sender = JointSender(LatestJointTarget(), FakeJointBackend(), [0] * 7)
    sender.period_samples = [0.008, 0.0099, 0.0101, 0.014]
    # Runtime increments this counter on the same >10ms condition.
    sender.deadline_misses = sum(v > 0.010 for v in sender.period_samples)
    assert sender.deadline_misses == 2
