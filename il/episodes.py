"""Saving and loading recorded episodes: one compressed .npz per episode in a dataset folder.

Every loop that records (teleop, policy evaluation rollouts, HG-DAgger) keeps a list of steps and
calls save_episode once at the end, so all datasets share one format.

Alignment: index t of every per-step array describes the same instant, the moment action[t] was
chosen. obs[t] is the frame the action was chosen from; speed[t] and the car state are read at that
same instant (before the physics step), and reward[t] / obstacle_hits[t] are what the step then
produced. So the loop must read the car state and append the step BEFORE overwriting obs:

    obs, _ = env.reset(seed=seed)
    steps = []
    while not done:
        action = choose(obs)
        state = car_state(env)                  # same instant as obs
        next_obs, reward, terminated, truncated, info = env.step(action)
        steps.append(make_step(obs, action, state, reward, info))
        obs = next_obs                          # only now
    save_episode(dataset_dir, steps, env, seed, end_reason(terminated, truncated, info), ...)

The final next_obs is not saved: no action was chosen from it.

Per-step arrays (T = number of actions):
  obs               (T, 96, 96, 3) uint8
  action            (T, 1) float32   steering the loop chose (the policy target)
  applied_action    (T, 3) float32   info["action"]: [steer, gas, brake] after SpeedControl
  speed             (T,)             info["speed"]
  reward            (T,)
  obstacle_hits     (T,) int         cumulative, after the step
  position          (T, 2)           car hull, world coordinates
  angle             (T,)
  linear_velocity   (T, 2)
  angular_velocity  (T,)
Metadata: a JSON string under "meta" (see save_episode), so loading never needs pickle.
"""

import datetime
import json
from pathlib import Path

import numpy as np


def car_state(env):
    """The car hull's pose and velocity right now. Call it before env.step (see module docstring)."""
    hull = env.unwrapped.car.hull
    return {
        "position": tuple(hull.position),
        "angle": hull.angle,
        "linear_velocity": tuple(hull.linearVelocity),
        "angular_velocity": hull.angularVelocity,
    }


def make_step(obs, action, state, reward, info):
    """One step's record: obs is the frame action was chosen from, state is car_state before the step."""
    return {
        "obs": obs,
        "action": np.asarray(action, dtype=np.float32).reshape(1),
        "applied_action": info["action"],
        "speed": info["speed"],
        "reward": reward,
        "obstacle_hits": info["obstacle_hits"],
        **state,
    }


def end_reason(terminated, truncated, info):
    """Why the env ended the episode, or None if it didn't (the loop then supplies user_reset/user_quit)."""
    if terminated:  # CarRacing terminates on a finished lap or on leaving the playfield
        return "lap_finished" if info.get("lap_finished") else "off_playfield"
    if truncated:
        return "time_limit"
    return None


def env_config(env):
    """Settings that determine the episode's dynamics, read from a SpeedControl(gym.make(...)) env."""
    return {
        "env_id": env.spec.id,
        "target_speed": float(env.target_speed),
        "k": float(env.k),
        "num_obstacles": int(env.unwrapped.num_obstacles),
        "obstacle_size": float(env.unwrapped.obstacle_size),
        "max_episode_steps": int(env.spec.max_episode_steps),
    }


def save_episode(dataset_dir, steps, env, seed, end_reason, terminated, truncated, lap_finished, **extra):
    """Write steps to the next free dataset_dir/ep_NNNN.npz and return its path.

    The metadata records the seed, env_config(env), how the episode ended, a timestamp, and any
    loop-specific settings passed as extra keyword arguments (e.g. teleop's fps and sensitivity).
    """
    assert steps, "empty episode: nothing to save"
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.array([step[key] for step in steps]) for key in steps[0]}
    arrays["obs"] = arrays["obs"].astype(np.uint8)
    arrays["action"] = arrays["action"].astype(np.float32)
    arrays["applied_action"] = arrays["applied_action"].astype(np.float32)
    meta = {
        "seed": int(seed),
        **env_config(env),
        "end_reason": end_reason,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "lap_finished": bool(lap_finished),
        "steps": len(steps),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        **extra,
    }
    existing = [int(p.stem.removeprefix("ep_")) for p in dataset_dir.glob("ep_*.npz")]
    path = dataset_dir / f"ep_{max(existing, default=-1) + 1:04d}.npz"
    with open(path, "xb") as f:  # "x": fail rather than overwrite an existing episode
        np.savez_compressed(f, meta=json.dumps(meta), **arrays)
    return path


def load_episode(path):
    """Return (arrays, meta): a dict of the per-step arrays and the metadata dict."""
    with np.load(path, allow_pickle=False) as f:
        arrays = {key: f[key] for key in f.files if key != "meta"}
        meta = json.loads(str(f["meta"]))
    return arrays, meta
