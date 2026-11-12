"""Drive with the trackpad as a virtual steering wheel; with --record, save episodes.

The env is SpeedControl(CarRacingObstacles-v0): the trackpad steers, the speed controller handles
gas. The loop draws into its own pygame window from render_mode="rgb_array" and paces itself with
clock.tick(fps). render_mode="human" isn't used because it ticks its own clock at 50 FPS. Physics
is always 1/50 s per step, so a lower --fps just slows the game down and gives more reaction time;
the recorded data is the same kind either way.

Each frame: draw obs_t, wait for the clock, read the trackpad, step. So the action is chosen from
the frame that was on screen, matching the alignment il/episodes.py expects.

Keys: Enter restarts (end_reason user_reset), Esc quits and discards the episode in progress. The mouse is in relative
mode while the window is open (cursor hidden, raw motion only), so steering never stops at an edge.

  .venv/bin/python -m il.teleop                       # test-drive, nothing saved
  .venv/bin/python -m il.teleop --record data/plain   # save episodes to data/plain/ep_NNNN.npz
"""

import argparse

import gymnasium as gym
import numpy as np
import pygame

from il.episodes import car_state, end_reason, make_step, save_episode
from il.speed_control import TARGET_SPEED, SpeedControl

FIRST_TRAIN_SEED = 1000  # seeds 0-999 are reserved for evaluation
VIEW_W, VIEW_H = 900, 600  # game area: 1.5x the env's 600x400 render (the obs view is letterboxed)
BAR_H = 40  # strip below the game area for the steering bar (display only, never in obs)


class TrackpadSteering:
    """A virtual steering wheel turned by horizontal trackpad motion.

    Each frame: value = clip(value * (1 - centering) + sensitivity * dx, -1, 1).
    sensitivity is wheel units per pixel of motion (1 / sensitivity pixels = full lock from center).
    centering is the fraction of the value removed each frame, i.e. per physics step, so it acts
    the same on the car at any --fps (0 = off: the wheel stays where you leave it).
    The clamp works like a wheel's end stop: extra motion past full lock is discarded, so moving back
    starts unwinding immediately.
    """

    def __init__(self, sensitivity=0.005, centering=0.05):
        self.sensitivity = sensitivity
        self.centering = centering
        self.value = 0.0

    def update(self, dx):
        """Advance one frame given the horizontal motion dx (pixels; positive = right)."""
        decayed = self.value * (1.0 - self.centering)
        self.value = float(np.clip(decayed + self.sensitivity * dx, -1.0, 1.0))
        return self.value

    def read(self):
        """Advance one frame from the mouse motion since the last call."""
        return self.update(pygame.mouse.get_rel()[0])

    def reset(self):
        self.value = 0.0


def draw(screen, frame, steer):
    """Draw a frame (env render or 96x96 obs) scaled to the game area, and the steering bar below."""
    h, w = frame.shape[:2]
    scale = min(VIEW_W / w, VIEW_H / h)
    size = (int(w * scale), int(h * scale))
    # surfarray is (x, y), numpy images are (y, x); transform.scale is nearest-neighbor, so the obs
    # view shows the policy's actual pixels.
    image = pygame.transform.scale(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), size)
    screen.fill((0, 0, 0))
    screen.blit(image, ((VIEW_W - size[0]) // 2, (VIEW_H - size[1]) // 2))
    cx, y, half, bar_h = VIEW_W // 2, VIEW_H + 10, VIEW_W // 2 - 20, BAR_H - 20
    fill = int(steer * half)
    pygame.draw.rect(screen, (60, 60, 60), (cx - half, y, 2 * half, bar_h))
    pygame.draw.rect(screen, (0, 200, 0), (min(cx, cx + fill), y, abs(fill), bar_h))
    pygame.draw.line(screen, (255, 255, 255), (cx, y - 4), (cx, y + bar_h + 4), 2)
    pygame.display.flip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", metavar="DATASET_DIR", help="save episodes here (omit to test-drive)")
    parser.add_argument("--fps", type=int, default=35, help="game speed; physics is 50 steps per sim second")
    parser.add_argument("--sensitivity", type=float, default=0.005, help="wheel units per pixel")
    parser.add_argument("--centering", type=float, default=0.07, help="per-frame decay toward 0 (0 = off)")
    parser.add_argument("--target-speed", type=float, default=TARGET_SPEED)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--view", choices=["full", "obs"], default="full", help="obs: what the policy sees")
    args = parser.parse_args()

    env = SpeedControl(
        gym.make(
            "CarRacingObstacles-v0", render_mode="rgb_array", max_episode_steps=args.max_episode_steps
        ),
        target_speed=args.target_speed,
    )
    pygame.init()
    screen = pygame.display.set_mode((VIEW_W, VIEW_H + BAR_H))
    pygame.display.set_caption("teleop: trackpad steers, Enter restarts, Esc quits")
    # Relative mode hides the cursor and reports raw motion with no position, so it never stops at
    # an edge. (set_grab + set_visible(False) doesn't reliably enable it on macOS: the hidden cursor
    # still hit the window edge and get_rel returned 0 there.)
    try:
        pygame.mouse.set_relative_mode(True)
    except pygame.error:  # e.g. SDL's headless dummy driver
        print("warning: no relative mouse mode; steering will stop at the window edges")
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()
    wheel = TrackpadSteering(args.sensitivity, args.centering)
    rng = np.random.default_rng()  # unseeded on purpose: a fresh track every episode
    quit = False
    while not quit:
        seed = int(rng.integers(FIRST_TRAIN_SEED, 2**31))
        obs, info = env.reset(seed=seed)
        wheel.reset()
        pygame.mouse.get_rel()  # discard motion from between episodes
        steps, reason = [], None
        terminated = truncated = False
        while reason is None:
            draw(screen, env.render() if args.view == "full" else obs, wheel.value)
            clock.tick(args.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    reason = "user_quit"
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    reason = "user_reset"
            if reason is not None:
                break
            action = np.array([wheel.read()], dtype=np.float32)
            state = car_state(env)
            next_obs, reward, terminated, truncated, info = env.step(action)
            steps.append(make_step(obs, action, state, reward, info))  # before obs is overwritten
            obs = next_obs
            reason = end_reason(terminated, truncated, info)
        quit = reason == "user_quit"
        hits = steps[-1]["obstacle_hits"] if steps else 0
        print(f"seed {seed}: {reason}, {len(steps)} steps, {hits} obstacle hits")
        if args.record and quit:
            print("  not saved (Esc discards the episode in progress)")
        elif args.record and steps:
            path = save_episode(
                args.record, steps, env, seed, reason, terminated, truncated,
                info.get("lap_finished", False), source="teleop", fps=args.fps,
                sensitivity=args.sensitivity, centering=args.centering, view=args.view,
            )
            print(f"  saved {path}")
    env.close()
    pygame.quit()
