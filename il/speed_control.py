"""Steering-only action interface: a proportional speed controller handles the gas.

Works on CarRacing-v3 and CarRacingObstacles-v0. A wrapper rather than a subclass: it changes the
action interface, not the world, so it composes with any CarRacing variant.

Each step, from the car's current speed (norm of the hull's linear velocity):
  gas = clip(k * (target_speed - speed), 0, 1),  brake = 0
The car has no drag, so on straights it coasts at constant speed and the controller idles; it only
adds gas after turns (which bleed speed) and from a standing start.

  action                  Box(-1, 1, shape=(1,)): steering (-1 = full left)
  info["action"]          the 3D action actually applied: [steer, gas, brake]
  info["speed"]           the speed the controller saw this step (before the physics step), i.e.
                          the speed matching the observation the steering was chosen from
"""

import gymnasium as gym
import numpy as np

import il.obstacles  # noqa: F401  (registers CarRacingObstacles-v0)

# Measured over 5 seeds with a centerline-following steerer (il/tune_speed_control.py; see
# notes/log.md, 2026-10-15 and 2026-10-16): from a standing start, acceleration is friction-limited
# at ~0.875 speed/step, so no k reaches 95% of the target faster than full gas does (36 steps at
# 35). k >= 0.05 hits that floor but overshoots (stored rear-wheel spin keeps pushing after the
# gas is cut), and with no drag or brake the overshoot persists until turns bleed it off.
# k = 0.04 hits the floor with ~0% overshoot at targets 30-50 and spends ~98% of the time within
# +-2% of 35 after settling.
# Target 35: chosen by test-driving (fastest speed that still handles hairpins and obstacles).
TARGET_SPEED = 35.0
K = 0.04


class SpeedControl(gym.Wrapper):
    def __init__(self, env, target_speed: float = TARGET_SPEED, k: float = K):
        super().__init__(env)
        self.target_speed = target_speed
        self.k = k
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def speed(self):
        return float(np.linalg.norm(self.env.unwrapped.car.hull.linearVelocity))

    def step(self, action):
        speed = self.speed()
        steer = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        gas = float(np.clip(self.k * (self.target_speed - speed), 0.0, 1.0))
        applied = np.array([steer, gas, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(applied)
        info["action"] = applied
        info["speed"] = speed
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    # Test-drive: left/right arrows steer, the controller handles speed. Enter restarts, Esc quits.
    import argparse

    import pygame

    parser = argparse.ArgumentParser()
    parser.add_argument("--target-speed", type=float, default=TARGET_SPEED)
    args = parser.parse_args()

    env = SpeedControl(
        gym.make("CarRacingObstacles-v0", render_mode="human"), target_speed=args.target_speed
    )
    quit = False
    while not quit:
        env.reset()
        total_reward, info, t = 0.0, {}, 0
        restart = terminated = truncated = False
        while not (terminated or truncated or restart or quit):
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    quit = True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    restart = True
            keys = pygame.key.get_pressed()
            steer = -1.0 * keys[pygame.K_LEFT] + 1.0 * keys[pygame.K_RIGHT]
            _, reward, terminated, truncated, info = env.step(np.array([steer], dtype=np.float32))
            total_reward += reward
            t += 1
            if t % 50 == 0:
                print(f"step {t}: speed {info['speed']:.1f} (target {env.target_speed:.0f})")
            if info["obstacle_hit"]:
                print(f"hit obstacle! ({info['obstacle_hits']}/{len(env.unwrapped.obstacles)})")
        print(f"episode done: reward {total_reward:+.1f}, obstacles hit {info.get('obstacle_hits', 0)}")
    env.close()
