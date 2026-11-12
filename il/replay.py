"""Watch a recorded episode in a pygame window, with the recorded steering bar below.

  --view full (default)  re-simulates the episode: rebuilds the env from the saved metadata, resets
                         with the saved seed and steps the saved actions. The env is deterministic, so
                         this reproduces the drive exactly; every frame is checked against the saved
                         obs, and a mismatch (e.g. the env code changed since recording) stops it.
  --view obs             plays the saved 96x96 frames upscaled (what the policy trains on). No
                         simulation, so it works even if the env has changed.

Plays at the recorded fps unless --fps is given. Esc or closing the window stops early.

  .venv/bin/python -m il.replay data/plain/ep_0000.npz
  .venv/bin/python -m il.replay data/plain/ep_0000.npz --view obs --fps 10
"""

import argparse

import gymnasium as gym
import numpy as np
import pygame

from il.episodes import load_episode
from il.speed_control import SpeedControl
from il.teleop import BAR_H, VIEW_H, VIEW_W, draw


def make_env(meta, render_mode=None):
    """The env an episode was recorded in, rebuilt from its metadata."""
    return SpeedControl(
        gym.make(
            meta["env_id"], render_mode=render_mode, max_episode_steps=meta["max_episode_steps"],
            num_obstacles=meta["num_obstacles"], obstacle_size=meta["obstacle_size"],
        ),
        target_speed=meta["target_speed"], k=meta["k"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="an episode file, e.g. data/plain/ep_0000.npz")
    parser.add_argument("--view", choices=["full", "obs"], default="full", help="obs: the saved frames")
    parser.add_argument("--fps", type=int, help="playback speed (default: the recorded fps)")
    args = parser.parse_args()

    arrays, meta = load_episode(args.path)
    fps = args.fps or meta.get("fps", 50)  # episodes from non-teleop loops may not record an fps
    print(f"seed {meta['seed']}: {meta['end_reason']}, {meta['steps']} steps, "
          f"{arrays['obstacle_hits'][-1]} obstacle hits, recorded {meta['timestamp']}")

    if args.view == "full":
        env = make_env(meta, render_mode="rgb_array")
        obs, _ = env.reset(seed=meta["seed"])
    pygame.init()
    screen = pygame.display.set_mode((VIEW_W, VIEW_H + BAR_H))
    pygame.display.set_caption(f"replay: {args.path} (Esc stops)")
    clock = pygame.time.Clock()
    for t, action in enumerate(arrays["action"]):
        if args.view == "full":
            assert np.array_equal(obs, arrays["obs"][t]), (
                f"replay diverged from the recording at step {t}: the env has changed since recording"
            )
            frame = env.render()
        else:
            frame = arrays["obs"][t]
        draw(screen, frame, float(action[0]))
        clock.tick(fps)
        if any(
            event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
            for event in pygame.event.get()
        ):
            print(f"stopped at step {t}")
            break
        if args.view == "full":
            obs, *_ = env.step(action)
    else:
        print("done" + (": every frame matched the recording" if args.view == "full" else ""))
    pygame.quit()
