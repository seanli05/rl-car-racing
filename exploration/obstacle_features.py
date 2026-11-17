"""Per-frame distance to the obstacle ahead, for exploring steering around obstacles.

Obstacle positions aren't saved in episode files, so they are rebuilt by resetting the
recorded env with the episode's seed (the env is deterministic).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from il.episodes import load_episode
from il.replay import make_env

PASS_COLORS = {"left": "#2a78d6", "right": "#eb6834"}


def obstacle_positions(meta):
    """(N, 2) obstacle centers for the episode described by meta."""
    env = make_env(meta)
    try:
        env.reset(seed=meta["seed"])
        return np.array([[o["x"], o["y"]] for o in env.unwrapped.obstacles])
    finally:
        env.close()


def episode_features(path, max_lateral=8.0):
    """Per-frame features for one episode, as a dict of (T,) arrays.

    For each frame, the obstacle considered is the nearest one ahead of the car
    (forward > 0) and within max_lateral of its heading line; frames with none get NaN.

      steering   the recorded action (-1 = full left)
      forward    distance to that obstacle along the car's heading
      lateral    its offset sideways; positive = obstacle is to the car's LEFT
      obstacle   which obstacle it is (-1 if none)
      passed     side the car went around it on: "left", "right", or "" if none
      episode    episode file name
    """
    arrays, meta = load_episode(path)
    obstacles = obstacle_positions(meta)
    pos, angle = arrays["position"], arrays["angle"]
    T = len(pos)

    heading = np.stack([-np.sin(angle), np.cos(angle)], axis=1)  # car's forward axis
    left = np.stack([-np.cos(angle), -np.sin(angle)], axis=1)
    rel = obstacles[None, :, :] - pos[:, None, :]  # (T, N, 2)
    fwd = np.einsum("tnd,td->tn", rel, heading)
    lat = np.einsum("tnd,td->tn", rel, left)

    candidate = (fwd > 0) & (np.abs(lat) <= max_lateral)
    fwd_masked = np.where(candidate, fwd, np.inf)
    nearest = fwd_masked.argmin(axis=1)
    has = np.isfinite(fwd_masked.min(axis=1))
    rows = np.arange(T)

    # Side each obstacle was passed on: at the car's closest approach, an obstacle on the
    # car's right (lateral < 0) means the car went around it on the left.
    dist = np.linalg.norm(rel, axis=2)
    closest = dist.argmin(axis=0)
    side_of = np.where(lat[closest, np.arange(len(obstacles))] < 0, "left", "right")

    return {
        "steering": arrays["action"].ravel().astype(np.float64),
        "forward": np.where(has, fwd[rows, nearest], np.nan),
        "lateral": np.where(has, lat[rows, nearest], np.nan),
        "obstacle": np.where(has, nearest, -1),
        "passed": np.where(has, side_of[nearest], ""),
        "episode": np.full(T, Path(path).name),
    }


def all_features(folder="data/plain", max_lateral=8.0):
    """episode_features for every episode in folder, concatenated."""
    parts = [episode_features(p, max_lateral) for p in sorted(Path(folder).glob("*.npz"))]
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def kde(x, grid=None, n=512):
    """(grid, density) Gaussian KDE with Scott's-rule bandwidth."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    bw = 1.06 * x.std() * x.size ** (-1 / 5) or 1e-3
    if grid is None:
        grid = np.linspace(x.min() - 3 * bw, x.max() + 3 * bw, n)
    density = np.zeros_like(grid)
    for chunk in np.array_split(x, max(1, x.size // 5000)):
        density += np.exp(-0.5 * ((grid[:, None] - chunk[None, :]) / bw) ** 2).sum(axis=1)
    return grid, density / (x.size * bw * np.sqrt(2 * np.pi))


def _style(ax):
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0)


def plot_steering_by_distance(feats, bins=(0, 10, 20, 30, 45), split_by_side=True, min_frames=30):
    """One KDE panel of steering per forward-distance bin, sharing both axes.

    split_by_side=True draws one curve per passing side (left / right) in each panel;
    False draws a single curve of all frames in the bin.
    """
    edges = list(zip(bins[:-1], bins[1:]))
    fig, axes = plt.subplots(1, len(edges), figsize=(3.6 * len(edges), 3.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    steer, fwd, passed = feats["steering"], feats["forward"], feats["passed"]
    near = (fwd >= bins[0]) & (fwd < bins[-1])
    reach = np.abs(steer[near]).max() * 1.15 if near.any() else 1.0
    grid = np.linspace(-reach, reach, 512)
    ymax = 0.0

    for ax, (lo, hi) in zip(axes, edges):
        in_bin = (fwd >= lo) & (fwd < hi)
        groups = [(s, in_bin & (passed == s)) for s in ("left", "right")] if split_by_side else [(None, in_bin)]
        for side, mask in groups:
            if mask.sum() < min_frames:
                continue
            color = PASS_COLORS[side] if side else PASS_COLORS["left"]
            _, d = kde(steer[mask], grid)
            ymax = max(ymax, d.max())
            ax.plot(grid, d, color=color, linewidth=2)
            ax.fill_between(grid, d, color=color, alpha=0.12, linewidth=0)
            ax.text(0.98, 0.95 - 0.08 * (side == "right"), f"n={mask.sum():,}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="#52514e")
        ax.set_title(f"obstacle {lo}–{hi} ahead", loc="left")
        ax.set_xlabel("steering  (← left · right →)")
        _style(ax)
    axes[0].set_ylim(0, ymax * 1.05)  # set once, after every panel is drawn (axes are shared)
    axes[0].set_ylabel("density")
    if split_by_side:
        handles = [plt.Line2D([], [], color=c, linewidth=2) for c in PASS_COLORS.values()]
        fig.legend(handles, [f"passed {s}" for s in PASS_COLORS], frameon=False,
                   loc="upper right", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92 if split_by_side else 1))
    return fig


def plot_steering_vs_distance(feats, max_forward=45):
    """Scatter of steering against distance to the obstacle ahead, colored by passing side."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for side, color in PASS_COLORS.items():
        m = (feats["passed"] == side) & (feats["forward"] <= max_forward)
        ax.scatter(feats["forward"][m], feats["steering"][m], s=6, alpha=0.35, color=color,
                   linewidths=0, label=f"passed {side} (n={m.sum():,})")
    ax.axhline(0, color="#c9c8c2", linewidth=1)
    ax.invert_xaxis()  # time flows left to right as the car approaches
    ax.set_xlabel("distance to obstacle ahead (car approaches →)")
    ax.set_ylabel("steering  (− left · right +)")
    ax.set_title("Steering while approaching obstacles", loc="left")
    ax.legend(frameon=False, markerscale=3, loc="upper left")
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    return fig
