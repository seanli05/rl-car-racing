"""Sweep the speed controller's gain k: log every run, plot every k, replay any run.

  .venv/bin/python -m il.tune_speed_control sweep [--target-speed 35] [--ks 0.03 0.04] [--seeds 0 1]
  .venv/bin/python -m il.tune_speed_control plot [--target-speed 35]   # redo metrics + plots from logs
  .venv/bin/python -m il.tune_speed_control replay --k 0.04 [--seed 0] [--render-mode human]

Steering comes from a fixed centerline-following rule (pursuit), so runs differ only in k and seed.
Each run is saved as one .npz with every step's applied action and car state. The env is
deterministic, so seed + logged steering reproduces a run exactly: `replay` re-drives it in the
game window and checks every step's car state against the log. (Box2D can't be restored from
body states alone, and pixel observations would be ~28 MB/run, so neither is stored.)

Outputs go to runs/tune_speed_control/target{target}/ (gitignored): k{k}_seed{seed}.npz,
metrics.csv, and plots/.
"""

import argparse
import csv
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.car_dynamics import SIZE, WHEEL_R

from il.speed_control import TARGET_SPEED, SpeedControl

ENV_ID = "CarRacingObstacles-v0"  # obstacles are sensors: same dynamics as CarRacing-v3
KS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.1, 0.3, 1.0]
SEEDS = [0, 1, 2, 3, 4]
LOOKAHEAD = 6  # pursuit aims at the centerline point this many points ahead (~20 units)
STEER_GAIN = 2.0  # steering per radian of heading error
SETTLE_STEP = 200  # metrics after this step measure holding speed, not the standing start
TURN_STEER = 0.3  # |steer| above this counts as turning (grey bands in plots)
BAND = 0.02  # "holding speed" means within +-2% of target
OUT_DIR = Path("runs/tune_speed_control")
VIDEO_DIR = Path("videos/tune_speed_control")
SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical, fixed order
K_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",  # ordinal blue ramp: light = small k
          "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
STATE_KEYS = ["position", "angle", "linear_velocity", "angular_velocity",
              "wheel_omega", "wheel_gas", "wheel_joint_angle"]


def pursuit(env):
    """Steer toward the track centerline LOOKAHEAD points ahead of the nearest one."""
    u = env.unwrapped
    p = np.array(u.car.hull.position)
    points = np.array([(x, y) for _, _, x, y in u.track])
    i = int(np.argmin(np.linalg.norm(points - p, axis=1)))
    to_target = points[(i + LOOKAHEAD) % len(points)] - p
    fwd = np.array(u.car.hull.GetWorldVector((0, 1)))
    # Signed angle from heading to target: + means the target is to the left (steer -1 = left).
    err = math.atan2(fwd[0] * to_target[1] - fwd[1] * to_target[0], fwd @ to_target)
    return float(np.clip(-STEER_GAIN * err, -1.0, 1.0)), i


def car_state(car):
    return {
        "position": tuple(car.hull.position),
        "angle": car.hull.angle,
        "linear_velocity": tuple(car.hull.linearVelocity),
        "angular_velocity": car.hull.angularVelocity,
        "wheel_omega": [w.omega for w in car.wheels],
        "wheel_gas": [w.gas for w in car.wheels],
        "wheel_joint_angle": [w.joint.angle for w in car.wheels],
    }


def run(k, target, seed):
    """Drive one full episode with pursuit steering; return the per-step log."""
    env = SpeedControl(gym.make(ENV_ID), target_speed=target, k=k)
    env.reset(seed=seed)
    keys = ["steer_input", "action", "speed", "reward", "track_idx", "on_road"] + STATE_KEYS
    log = {key: [] for key in keys}
    terminated = truncated = False
    while not (terminated or truncated):
        steer, i = pursuit(env)
        _, reward, terminated, truncated, info = env.step(np.array([steer], dtype=np.float32))
        car = env.unwrapped.car
        for key, value in car_state(car).items():
            log[key].append(value)
        log["steer_input"].append(steer)
        log["action"].append(info["action"])
        log["speed"].append(info["speed"])
        log["reward"].append(reward)
        log["track_idx"].append(i)
        log["on_road"].append(all(len(w.tiles) > 0 for w in car.wheels))
    env.close()
    log = {key: np.array(value) for key, value in log.items()}
    log.update(env_id=ENV_ID, seed=seed, k=k, target_speed=target)
    return log


def metrics(log):
    target, v = float(log["target_speed"]), log["speed"]
    reached = v >= 0.95 * target
    turning = np.abs(log["steer_input"]) > 0.5
    late = v[SETTLE_STEP:]
    return {
        "rise_steps": int(np.argmax(reached)) if reached.any() else -1,
        "overshoot_pct": 100 * (v[:SETTLE_STEP].max() - target) / target,
        "tracking_err_pct": 100 * np.mean(np.abs(late - target)) / target,
        "time_in_band_pct": 100 * np.mean(np.abs(late - target) <= BAND * target),
        "max_gas_turning": float(log["action"][turning, 1].max()) if turning.any() else 0.0,
        "offroad_steps": int((~log["on_road"]).sum()),
    }


def rise_floor(target, seed=0):
    """Fewest steps any controller can take to reach 95% of target: full gas from standstill.

    Counted like `rise_steps` (the step at which the pre-step speed first reaches 95%)."""
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    car = env.unwrapped.car
    t = 0
    while np.linalg.norm(car.hull.linearVelocity) < 0.95 * target:
        steer, _ = pursuit(env)
        env.step(np.array([steer, 1.0, 0.0], dtype=np.float32))
        t += 1
    env.close()
    return t


def rear_slip(log):
    """Rear wheel surface speed minus car speed: the stored wheelspin that causes overshoot."""
    rear_surface = log["wheel_omega"][:, 2:].mean(axis=1) * WHEEL_R * SIZE
    return rear_surface - np.linalg.norm(log["linear_velocity"], axis=1)


def shade_turns(axes, log):
    turning = np.abs(log["steer_input"]) > TURN_STEER
    for ax in axes:
        ax.fill_between(np.arange(len(turning)), 0, 1, where=turning,
                        transform=ax.get_xaxis_transform(), color="#e4e4e4", lw=0)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#eee", lw=0.8)
    ax.set_axisbelow(True)


def plot_k(logs, k, target, path):
    """Speed, gas and rear-wheel slip vs step for every seed at one k."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, height_ratios=[2, 1, 1])
    shade_turns(axes, logs[0])
    for color, log in zip(SEED_COLORS, logs):
        axes[0].plot(log["speed"], color=color, lw=1.3, label=f"seed {log['seed']}")
        axes[1].plot(log["action"][:, 1], color=color, lw=1.0)
        axes[2].plot(rear_slip(log), color=color, lw=1.0)
    axes[0].axhline(target, color="#777", lw=1, ls="--")
    axes[0].set_title(f"k = {k:g}, target {target:g}: speed, gas and rear-wheel slip per seed")
    axes[0].set_ylabel("speed")
    axes[0].legend(frameon=False, loc="lower right", ncol=len(logs))
    axes[1].set_ylabel("gas")
    axes[2].set_ylabel("rear slip")
    axes[2].set_xlabel("step")
    for ax in axes:
        tidy(ax)
    fig.text(0.01, 0.005, f"grey bands: seed {logs[0]['seed']} turning (|steer| > {TURN_STEER}). "
             "Rear slip = rear wheel surface speed - car speed.", color="#666", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_compare(logs_by_k, target, path, steps=None):
    """Speed and gas vs step for every k on one seed."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[2, 1])
    first = next(iter(logs_by_k.values()))
    end = steps or len(first["speed"])
    shade_turns(axes, {"steer_input": first["steer_input"][:end]})
    for color, (k, log) in zip(K_RAMP, logs_by_k.items()):
        axes[0].plot(log["speed"][:end], color=color, lw=1.4, label=f"k={k:g}")
        axes[1].plot(log["action"][:end, 1], color=color, lw=1.0)
    axes[0].axhline(target, color="#777", lw=1, ls="--")
    zoom = f", steps 0-{end}" if steps else ""
    axes[0].set_title(f"All k, seed {first['seed']}, target {target:g}{zoom} (darker = larger k)")
    axes[0].set_ylabel("speed")
    axes[0].legend(frameon=False, loc="lower right", ncol=5)
    axes[1].set_ylabel("gas")
    axes[1].set_xlabel("step")
    for ax in axes:
        tidy(ax)
    fig.text(0.01, 0.005, f"grey bands: turning (|steer| > {TURN_STEER}, k={next(iter(logs_by_k)):g})",
             color="#666", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_metrics(rows, target, floor, path):
    """Each metric vs k: mean over seeds, band = min to max."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    panels = [("rise_steps", "rise time (steps to 95% of target)"),
              ("overshoot_pct", "standing-start overshoot %"),
              ("tracking_err_pct", f"tracking error % (mean |speed - target|, after step {SETTLE_STEP})"),
              ("time_in_band_pct", f"% of time within ±{BAND:.0%} of target (after step {SETTLE_STEP})")]
    ks = sorted({r["k"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title) in zip(axes.flat, panels):
        values = [[r[key] for r in rows if r["k"] == k] for k in ks]
        ax.fill_between(ks, [min(v) for v in values], [max(v) for v in values],
                        color="#b7d3f6", lw=0, label="min-max over seeds")
        ax.plot(ks, [np.mean(v) for v in values], color="#2a78d6", lw=2, marker="o", ms=5,
                label="mean")
        ax.set_xscale("log")
        ax.set_xticks(ks, [f"{k:g}" for k in ks], rotation=60, fontsize=8)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("k (log scale; ticks = tested values)")
        ax.set_title(title, fontsize=10)
        tidy(ax)
    rise, over, err, band = axes.flat
    rise.axhline(floor, color="#777", lw=1, ls="--")
    rise.annotate(f"physical floor: {floor} steps (full gas from standstill)", (ks[-1], floor),
                  xytext=(0, 8), textcoords="offset points", ha="right", va="bottom",
                  color="#555", fontsize=8)
    rise.set_ylim(0, 1.05 * max(r["rise_steps"] for r in rows))
    over.axhline(0, color="#777", lw=1, ls="--")
    err.set_ylim(bottom=0)
    band.set_ylim(0, 102)
    rise.legend(frameon=False, loc="upper right")
    fig.suptitle(f"Controller metrics vs k, target {target:g}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_track(log, track, path):
    """Car path over the track centerline, colored by speed (80-105% of target); x = off-road."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#3987e5", "#0d366b"])
    target = float(log["target_speed"])
    points = np.array([(x, y) for _, _, x, y in track] + [track[0][2:]])
    pos, off = log["position"], ~log["on_road"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(points[:, 0], points[:, 1], color="#ddd", lw=2, label="track centerline")
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=log["speed"], cmap=cmap, s=6, zorder=2,
                    vmin=0.8 * target, vmax=1.05 * target, label="car (color = speed)")
    ax.scatter(pos[off, 0], pos[off, 1], marker="x", color="#eb6834", s=14, lw=1, zorder=3,
               label="off-road")
    ax.scatter(*pos[0], color="black", s=30, zorder=4, label="start")
    fig.colorbar(sc, ax=ax, shrink=0.7, extend="both", label="speed")
    ax.set_aspect("equal")
    ax.set_title(f"k = {float(log['k']):g}, seed {int(log['seed'])}: path, {off.sum()} off-road steps")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def sweep(target, ks, seeds):
    out = OUT_DIR / f"target{target:g}"
    out.mkdir(parents=True, exist_ok=True)
    for k in ks:
        for seed in seeds:
            log = run(k, target, seed)
            np.savez(out / f"k{k:g}_seed{seed}.npz", **log)
            print(f"k={k:g} seed={seed}: {len(log['speed'])} steps")
    plot_all(target)


def plot_all(target):
    """Metrics table, metrics.csv and every plot, from the saved logs (no simulation reruns)."""
    out = OUT_DIR / f"target{target:g}"
    (out / "plots").mkdir(exist_ok=True)
    logs = {}
    for path in out.glob("k*_seed*.npz"):
        log = dict(np.load(path))
        logs[float(log["k"]), int(log["seed"])] = log
    ks, seeds = sorted({k for k, _ in logs}), sorted({s for _, s in logs})
    rows = [{"k": k, "seed": seed, **metrics(logs[k, seed])} for k in ks for seed in seeds]

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = [key for key in rows[0] if key not in ("k", "seed")]
    print(f"\ntarget {target:g}, mean over seeds {seeds}")
    print(f"{'k':>6} " + " ".join(f"{key:>16}" for key in keys))
    for k in ks:
        mean = {key: np.mean([r[key] for r in rows if r["k"] == k]) for key in keys}
        print(f"{k:>6g} " + " ".join(f"{mean[key]:>16.2f}" for key in keys))

    for k in ks:
        plot_k([logs[k, s] for s in seeds], k, target, out / "plots" / f"k{k:g}.png")
        env = gym.make(ENV_ID)
        env.reset(seed=seeds[0])  # regenerate the seed's track (it isn't stored in the log)
        plot_track(logs[k, seeds[0]], env.unwrapped.track, out / "plots" / f"track_k{k:g}.png")
        env.close()
    by_k = {k: logs[k, seeds[0]] for k in ks}
    plot_compare(by_k, target, out / "plots" / f"compare_seed{seeds[0]}.png")
    plot_compare(by_k, target, out / "plots" / f"compare_start_seed{seeds[0]}.png", steps=150)
    plot_metrics(rows, target, rise_floor(target, seeds[0]), out / "plots" / "metrics.png")
    print(f"\nlogs, metrics.csv and plots/ in {out}")


def replay(target, k, seed, render_mode="rgb_array"):
    """Re-drive a logged run, checking every step's car state against the log.

    render_mode "rgb_array" saves an MP4 (no window, faster than real time); "human" shows the
    game window in real time instead."""
    out = OUT_DIR / f"target{target:g}"
    path = out / f"k{k:g}_seed{seed}.npz"
    if not path.exists():
        done = [dict(np.load(p, allow_pickle=False)) for p in out.glob("k*_seed*.npz")]
        ks = sorted({float(d["k"]) for d in done})
        seeds = sorted({int(d["seed"]) for d in done})
        raise SystemExit(f"no finished run for k={k:g} seed={seed} at target {target:g}. "
                         f"Finished ks: {[f'{x:g}' for x in ks] or 'none'}, seeds: {seeds or 'none'} "
                         "(run `sweep` first)")
    log = dict(np.load(path))
    env = SpeedControl(gym.make(str(log["env_id"]), render_mode=render_mode),
                       target_speed=float(log["target_speed"]), k=float(log["k"]))
    env.reset(seed=int(log["seed"]))

    writer = None
    if render_mode == "rgb_array":
        import imageio_ffmpeg

        video = VIDEO_DIR / f"target{target:g}" / f"k{k:g}_seed{seed}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        frame = env.render()
        # 600x400 frames: block size 8 divides both, so ffmpeg doesn't resize them.
        writer = imageio_ffmpeg.write_frames(str(video), (frame.shape[1], frame.shape[0]),
                                             fps=env.metadata["render_fps"], macro_block_size=8)
        writer.send(None)  # start the generator
        writer.send(frame)
    else:
        import pygame
    print(f"replaying k={k:g} seed={seed}: {len(log['speed'])} steps"
          + (" (Esc to stop)" if writer is None else ""))

    for t, steer in enumerate(log["steer_input"]):
        if writer is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    env.close()
                    return
        env.step(np.array([steer], dtype=np.float32))
        state = car_state(env.unwrapped.car)
        for key in STATE_KEYS:
            assert np.array_equal(np.array(state[key]), log[key][t]), f"step {t}: {key} differs from log"
        if writer is not None:
            writer.send(env.render())
        if t % 100 == 0:
            print(f"step {t}: speed {log['speed'][t]:.1f}, gas {log['action'][t, 1]:.2f}")
    print("replay matched the log at every step")
    if writer is not None:
        writer.close()
        print(f"saved {video}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("sweep")
    p.add_argument("--target-speed", type=float, default=TARGET_SPEED)
    p.add_argument("--ks", type=float, nargs="+", default=KS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p = commands.add_parser("plot")
    p.add_argument("--target-speed", type=float, default=TARGET_SPEED)
    p = commands.add_parser("replay")
    p.add_argument("--k", type=float, required=True)
    p.add_argument("--seed", type=int, default=SEEDS[0])
    p.add_argument("--target-speed", type=float, default=TARGET_SPEED)
    p.add_argument("--render-mode", choices=["rgb_array", "human"], default="rgb_array",
                   help="rgb_array saves an MP4 to videos/; human shows the game window")
    args = parser.parse_args()
    if args.command == "sweep":
        sweep(args.target_speed, args.ks, args.seeds)
    elif args.command == "plot":
        plot_all(args.target_speed)
    else:
        replay(args.target_speed, args.k, args.seed, args.render_mode)
