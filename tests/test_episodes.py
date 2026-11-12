"""Tests for il/episodes.py. Run with: .venv/bin/python -m pytest tests/"""

import gymnasium as gym
import numpy as np

from il.episodes import car_state, end_reason, load_episode, make_step, save_episode
from il.replay import make_env as make_env_from_meta
from il.speed_control import SpeedControl

STEPS = 120  # time limit for the test episodes
SEED = 1234


def make_env(**kwargs):
    return SpeedControl(gym.make("CarRacingObstacles-v0", max_episode_steps=STEPS, **kwargs))


def obs_policy(obs):
    """Steering that depends on the exact pixels, so a misaligned obs/action pair is detectable."""
    return np.array([np.sin(obs.astype(np.float64).sum())], dtype=np.float32)


def record(env, seed):
    """Run obs_policy the way a recording loop does; return (steps, first obs, terminated, truncated, info)."""
    obs, info = env.reset(seed=seed)
    first_obs, steps = obs, []
    terminated = truncated = False
    while not (terminated or truncated):
        if env.render_mode == "rgb_array":
            env.render()  # as teleop does: rendering the window view must not affect the episode
        action = obs_policy(obs)
        state = car_state(env)
        next_obs, reward, terminated, truncated, info = env.step(action)
        steps.append(make_step(obs, action, state, reward, info))
        obs = next_obs
    return steps, first_obs, terminated, truncated, info


def record_and_save(dataset_dir, render_mode=None):
    env = make_env(render_mode=render_mode)
    steps, first_obs, terminated, truncated, info = record(env, SEED)
    path = save_episode(
        dataset_dir, steps, env, SEED, end_reason(terminated, truncated, info), terminated,
        truncated, info.get("lap_finished", False), source="test", sensitivity=0.005,
    )
    return path, steps, first_obs


def test_end_reason():
    assert end_reason(True, False, {"lap_finished": True}) == "lap_finished"
    assert end_reason(True, False, {"lap_finished": False}) == "off_playfield"
    assert end_reason(False, True, {}) == "time_limit"
    assert end_reason(True, True, {"lap_finished": True}) == "lap_finished", "termination wins"
    assert end_reason(False, False, {}) is None


def test_save_load_round_trip(tmp_path):
    path, steps, _ = record_and_save(tmp_path)
    assert path == tmp_path / "ep_0000.npz"
    arrays, meta = load_episode(path)

    T = len(steps)
    assert T == STEPS
    expected = {
        "obs": ((T, 96, 96, 3), np.uint8),
        "action": ((T, 1), np.float32),
        "applied_action": ((T, 3), np.float32),
        "speed": ((T,), None),
        "reward": ((T,), None),
        "obstacle_hits": ((T,), None),
        "position": ((T, 2), None),
        "angle": ((T,), None),
        "linear_velocity": ((T, 2), None),
        "angular_velocity": ((T,), None),
    }
    assert set(arrays) == set(expected)
    for key, (shape, dtype) in expected.items():
        assert arrays[key].shape == shape, key
        if dtype is not None:
            assert arrays[key].dtype == dtype, key
        assert np.array_equal(arrays[key], np.array([s[key] for s in steps], dtype=arrays[key].dtype)), key

    assert meta["seed"] == SEED
    assert meta["env_id"] == "CarRacingObstacles-v0"
    assert (meta["target_speed"], meta["k"]) == (35.0, 0.04)
    assert (meta["num_obstacles"], meta["obstacle_size"]) == (5, 2.5)
    assert meta["max_episode_steps"] == STEPS
    assert meta["end_reason"] == "time_limit"
    assert (meta["terminated"], meta["truncated"], meta["lap_finished"]) == (False, True, False)
    assert meta["steps"] == T
    assert (meta["source"], meta["sensitivity"]) == ("test", 0.005), "extra settings saved"
    assert "timestamp" in meta

    path2, _, _ = record_and_save(tmp_path)
    assert path2 == tmp_path / "ep_0001.npz", "auto-incrementing"


def test_obs_action_alignment(tmp_path):
    path, _, first_obs = record_and_save(tmp_path)
    arrays, _ = load_episode(path)
    obs, action = arrays["obs"], arrays["action"]
    assert np.array_equal(obs[0], first_obs), "obs[0] is the reset frame"
    assert len(np.unique(action)) > 0.9 * len(action), "actions must vary for this test to mean anything"
    for t in range(len(obs)):
        assert np.array_equal(action[t], obs_policy(obs[t])), f"action[{t}] was not chosen from obs[{t}]"
    assert np.array_equal(arrays["applied_action"][:, 0], action[:, 0])
    # Car state is read at the same instant as info["speed"] (before the physics step).
    assert np.allclose(arrays["speed"], np.linalg.norm(arrays["linear_velocity"], axis=1))


def test_replay_reproduces_observations(tmp_path):
    # Record with rendering on (as teleop does), replay headless from the saved metadata alone.
    path, _, _ = record_and_save(tmp_path, render_mode="rgb_array")
    arrays, meta = load_episode(path)
    env = make_env_from_meta(meta)  # what il/replay.py uses
    obs, _ = env.reset(seed=meta["seed"])
    for t, action in enumerate(arrays["action"]):
        assert np.array_equal(obs, arrays["obs"][t]), f"obs differs at step {t}"
        assert car_state(env)["position"] == tuple(arrays["position"][t])
        obs, _, terminated, truncated, info = env.step(action)
    assert end_reason(terminated, truncated, info) == meta["end_reason"]
    assert info["obstacle_hits"] == arrays["obstacle_hits"][-1]
