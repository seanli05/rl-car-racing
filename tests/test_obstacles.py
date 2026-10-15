"""Tests for il/obstacles.py. Run with: .venv/bin/python -m pytest tests/"""

import gymnasium as gym
import numpy as np
import pytest

import il.obstacles as obstacles

SEEDS = range(10)
AHEAD_IDX = 8  # track point ~28 units ahead of the start: in view, and hit when driving straight
GAS = np.array([0.0, 0.5, 0.0], dtype=np.float32)


def make_env(**kwargs):
    return gym.make("CarRacingObstacles-v0", **kwargs)


def obstacle_positions(env):
    return [(o["x"], o["y"], o["track_idx"]) for o in env.unwrapped.obstacles]


def test_same_seed_same_obstacles():
    env = make_env()
    env.reset(seed=0)
    first, track = obstacle_positions(env), env.unwrapped.track
    env.reset(seed=1)
    other = obstacle_positions(env)
    env.reset(seed=0)
    assert obstacle_positions(env) == first
    assert env.unwrapped.track == track
    assert other != first


@pytest.mark.parametrize("seed", SEEDS)
def test_placement_invariants(seed):
    env = make_env(num_obstacles=5)
    env.reset(seed=seed)
    u = env.unwrapped
    n = len(u.track)
    assert len(u.obstacles) == 5

    segments = np.array_split(np.arange(int(obstacles.START_SKIP_FRAC * n), n), 5)
    for o, segment in zip(u.obstacles, segments):
        idx = o["track_idx"]
        _, _, x, y = u.track[idx]
        assert (o["x"], o["y"]) == (x, y), "obstacle must sit on the centerline point"
        assert idx in segment, "one obstacle per segment, outside the start region"
        margin = int(obstacles.SEGMENT_MARGIN_FRAC * len(segment))
        middle = segment[margin : len(segment) - margin]
        curvature = [u._curvature(i) for i in middle]
        if u._curvature(idx) > obstacles.MAX_CURVATURE:
            # Only allowed when the segment has no straight point: then it's the least-curved one.
            assert min(curvature) > obstacles.MAX_CURVATURE
            assert u._curvature(idx) == min(curvature)


def test_obstacle_visible_in_observation():
    env = make_env()
    env.reset(seed=0)
    env.unwrapped._place_obstacles([AHEAD_IDX])
    for _ in range(50):  # let the start-of-episode camera zoom finish
        obs, *_ = env.step(np.zeros(3, dtype=np.float32))
    road_area = obs[:84]  # rows below this are the indicator bar
    brown = np.all(np.abs(road_area.astype(int) - obstacles.OBSTACLE_COLOR) <= 10, axis=-1)
    assert brown.sum() >= 10


def test_hit_counts_once_without_terminating():
    env = make_env()
    env.reset(seed=0)
    env.unwrapped._place_obstacles([AHEAD_IDX])
    hit_steps = []
    for t in range(150):  # long enough to drive fully through the obstacle
        _, _, terminated, truncated, info = env.step(GAS)
        assert not terminated
        if info["obstacle_hit"]:
            hit_steps.append(t)
    assert len(hit_steps) == 1
    assert info["obstacle_hits"] == 1
    assert env.unwrapped.obstacles[0]["hit"]


def test_obstacles_dont_change_dynamics_or_reward():
    # Drive the same actions through an obstacle and through the plain env: everything must match.
    plain, with_obs = gym.make("CarRacing-v3"), make_env()
    plain.reset(seed=0)
    with_obs.reset(seed=0)
    with_obs.unwrapped._place_obstacles([AHEAD_IDX])
    for _ in range(150):
        _, r1, *_ = plain.step(GAS)
        _, r2, _, _, info = with_obs.step(GAS)
        assert r1 == r2
        assert tuple(plain.unwrapped.car.hull.position) == tuple(
            with_obs.unwrapped.car.hull.position
        )
    assert info["obstacle_hits"] == 1, "the run must actually pass through the obstacle"
    assert plain.unwrapped.tile_visited_count == with_obs.unwrapped.tile_visited_count


def test_random_episode_and_repeated_reset():
    env = make_env(max_episode_steps=300)
    env.action_space.seed(0)
    env.reset(seed=0)
    truncated = terminated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
    assert "obstacle_hits" in info

    body_counts = []
    for seed in range(5):
        env.reset(seed=seed)
        body_counts.append(len(env.unwrapped.world.bodies) - len(env.unwrapped.track))
    assert len(set(body_counts)) == 1, f"Box2D bodies leaking across resets: {body_counts}"
    env.close()
