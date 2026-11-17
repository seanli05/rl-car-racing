"""Watch a trained policy drive, live, in a pygame window.

Same env and window as il/teleop.py (SpeedControl(CarRacingObstacles-v0)), but the model steers
instead of the trackpad; the bar below the game shows its steering. Each step: preprocess obs_t the
same way training did (models/dataset.py: preprocess), run the model, clip to [-1, 1], step.

Tracks come from the evaluation seeds (0-999), which teleop never records on, so you're watching
the policy on tracks it has never seen. Episodes start at --seed and count up. A summary line is
printed after each one.

Keys: Space pauses, Enter skips to the next track, Esc quits.

  .venv/bin/python -m model_testing.watch models.firsttest:CNN firsttest.pth
  .venv/bin/python -m model_testing.watch models.firsttest:CNN firsttest.pth --seed 5 --fps 25 --view obs
"""

import argparse
import importlib

import gymnasium as gym
import numpy as np
import pygame
import torch

from il.episodes import end_reason
from il.speed_control import TARGET_SPEED, SpeedControl
from il.teleop import BAR_H, FIRST_TRAIN_SEED, VIEW_H, VIEW_W, draw
from models.dataset import preprocess


def load_policy(spec, checkpoint, device):
    """Build the model class named by "module:Class" and load its weights (a state_dict file).

    A state_dict holds only the weights, not the architecture, so we need the class to rebuild the
    network before loading into it. Importing the module must not train: keep training code under
    `if __name__ == "__main__":`.
    """
    module_name, class_name = spec.split(":")
    model = getattr(importlib.import_module(module_name), class_name)()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()  # eval(): dropout/batchnorm switch to inference behavior


@torch.no_grad()  # inference only: don't build the graph for backprop
def act(model, obs, device):
    x = preprocess(obs).unsqueeze(0).to(device)  # (1, 3, 96, 96): a batch of one frame
    steer = float(np.clip(model(x).item(), -1.0, 1.0))  # MSE outputs are unbounded; the env isn't
    return np.array([steer], dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help='model class as "module:Class", e.g. models.firsttest:CNN')
    parser.add_argument("checkpoint", help="state_dict file saved with torch.save, e.g. firsttest.pth")
    parser.add_argument("--seed", type=int, default=0, help=f"first evaluation seed (0-{FIRST_TRAIN_SEED - 1})")
    parser.add_argument("--fps", type=int, default=50, help="50 = real time; lower to watch closely")
    parser.add_argument("--target-speed", type=float, default=TARGET_SPEED)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--view", choices=["full", "obs"], default="full", help="obs: what the policy sees")
    args = parser.parse_args()
    if not 0 <= args.seed < FIRST_TRAIN_SEED:
        parser.error(f"--seed must be an evaluation seed, 0-{FIRST_TRAIN_SEED - 1}")

    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
    model = load_policy(args.model, args.checkpoint, device)
    env = SpeedControl(
        gym.make(
            "CarRacingObstacles-v0", render_mode="rgb_array", max_episode_steps=args.max_episode_steps
        ),
        target_speed=args.target_speed,
    )
    pygame.init()
    screen = pygame.display.set_mode((VIEW_W, VIEW_H + BAR_H))
    pygame.display.set_caption(f"watch {args.model}: Space pauses, Enter next track, Esc quits")
    clock = pygame.time.Clock()

    seed, quit = args.seed, False
    while not quit:
        obs, info = env.reset(seed=seed)
        steer, steps, paused, reason = 0.0, 0, False, None
        while reason is None:
            draw(screen, env.render() if args.view == "full" else obs, steer)
            clock.tick(args.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    reason = "user_quit"
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    reason = "user_reset"
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused
            if reason is not None or paused:
                continue
            action = act(model, obs, device)
            steer = float(action[0])
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            reason = end_reason(terminated, truncated, info)
        quit = reason == "user_quit"
        u = env.unwrapped
        print(
            f"seed {seed}: {reason}, {steps} steps, tiles {u.tile_visited_count}/{len(u.track)}, "
            f"obstacle hits {info.get('obstacle_hits', 0)}"
        )
        seed += 1
    env.close()
    pygame.quit()
