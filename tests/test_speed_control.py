"""Tests for il/speed_control.py. Run with: .venv/bin/python -m pytest tests/"""

import gymnasium as gym
import numpy as np
import pytest

from il.speed_control import SpeedControl

ENV_IDS = ["CarRacing-v3", "CarRacingObstacles-v0"]
TARGET = 35.0
STRAIGHT = np.array([0.0], dtype=np.float32)
LEFT = np.array([-0.5], dtype=np.float32)


def make_env(env_id="CarRacing-v3", **kwargs):
    return SpeedControl(gym.make(env_id), target_speed=TARGET, **kwargs)


def drive(env, steering):
    """Step through a steering sequence; return per-step speeds and applied actions."""
    speeds, actions = [], []
    for steer in steering:
        _, _, terminated, truncated, info = env.step(steer)
        assert not (terminated or truncated)
        speeds.append(info["speed"])
        actions.append(info["action"])
    return np.array(speeds), np.array(actions)


def test_action_space_and_info():
    env = make_env()
    assert env.action_space.shape == (1,)
    assert (env.action_space.low, env.action_space.high) == (-1.0, 1.0)
    env.reset(seed=0)
    speed_before = env.speed()
    _, _, _, _, info = env.step(np.array([2.0], dtype=np.float32))
    assert info["speed"] == speed_before, "info speed is the one the controller saw"
    steer, gas, brake = info["action"]
    assert (steer, gas, brake) == (1.0, 1.0, 0.0), "steering clipped; full gas at standstill"


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("seed", range(3))
def test_reaches_target_from_standing_start(env_id, seed):
    env = make_env(env_id)
    env.reset(seed=seed)
    speeds, _ = drive(env, [STRAIGHT] * 200)
    # Acceleration is friction-limited at ~0.875/step, so 95% of 35 takes at least ~36 steps.
    assert speeds[60] >= 0.95 * TARGET
    assert speeds.max() <= 1.01 * TARGET, "overshoot"


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_holds_speed_on_straight_with_no_gas(env_id):
    env = make_env(env_id)
    env.reset(seed=0)
    drive(env, [STRAIGHT] * 250)
    speeds, actions = drive(env, [STRAIGHT] * 50)
    assert np.all(actions[:, 1] < 0.01), "no drag: the controller should idle"
    assert np.all(np.abs(speeds - TARGET) < 0.1)


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_recovers_after_turn(env_id):
    env = make_env(env_id)
    env.reset(seed=0)
    drive(env, [STRAIGHT] * 100)
    turn_speeds, turn_actions = drive(env, [LEFT] * 20)
    after, _ = drive(env, [STRAIGHT] * 100)
    assert turn_speeds.min() < 0.99 * TARGET, "the turn must actually cost speed"
    assert turn_actions[:, 1].max() > 0, "the controller adds gas to compensate"
    assert np.all(np.abs(after[50:] - TARGET) < 0.005 * TARGET)


def test_same_seed_same_trajectory():
    rng = np.random.default_rng(0)
    steering = rng.uniform(-1, 1, size=(150, 1)).astype(np.float32)
    runs = []
    for _ in range(2):
        env = make_env("CarRacingObstacles-v0")
        env.reset(seed=0)
        speeds, actions = drive(env, steering)
        runs.append((speeds, actions, tuple(env.unwrapped.car.hull.position)))
    assert np.array_equal(runs[0][0], runs[1][0])
    assert np.array_equal(runs[0][1], runs[1][1])
    assert runs[0][2] == runs[1][2]
