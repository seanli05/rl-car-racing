"""Tests for il/teleop.py's steering math (no window needed). Run with: .venv/bin/python -m pytest tests/"""

import pytest

from il.teleop import TrackpadSteering


def test_sensitivity_scales_motion():
    wheel = TrackpadSteering(sensitivity=0.01, centering=0.0)
    assert wheel.update(20) == pytest.approx(0.2)
    assert wheel.update(-50) == pytest.approx(-0.3), "motion accumulates; left is negative"
    assert TrackpadSteering(sensitivity=0.002, centering=0.0).update(20) == pytest.approx(0.04)


def test_no_motion_holds_value_without_centering():
    wheel = TrackpadSteering(sensitivity=0.01, centering=0.0)
    wheel.update(30)
    for _ in range(100):
        wheel.update(0)
    assert wheel.value == pytest.approx(0.3)


def test_clamps_to_full_lock():
    wheel = TrackpadSteering(sensitivity=0.01, centering=0.0)
    assert wheel.update(500) == 1.0
    assert wheel.update(500) == 1.0
    # End stop: motion past full lock is discarded, so turning back unwinds immediately.
    assert wheel.update(-10) == pytest.approx(0.9)
    assert wheel.update(-1000) == -1.0


def test_centering_decays_toward_zero():
    wheel = TrackpadSteering(sensitivity=0.01, centering=0.1)
    assert wheel.update(50) == pytest.approx(0.5), "decay applies to the old value, not new motion"
    assert wheel.update(0) == pytest.approx(0.45)
    for _ in range(100):
        wheel.update(0)
    assert abs(wheel.value) < 1e-4
    wheel.update(-50)
    assert wheel.update(0) < 0, "centers from the left too, without overshooting"
    # Holding a turn against centering: steady state is sensitivity * dx / centering.
    for _ in range(200):
        wheel.update(2)
    assert wheel.value == pytest.approx(0.01 * 2 / 0.1)


def test_reset():
    wheel = TrackpadSteering(sensitivity=0.01, centering=0.0)
    wheel.update(40)
    wheel.reset()
    assert wheel.value == 0.0
