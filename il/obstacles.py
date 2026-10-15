"""CarRacing with obstacles on the road centerline.

Obstacles sit exactly in the middle of straight-ish road segments, so passing on the left and
passing on the right are equally valid. This makes human demonstrations deliberately bimodal.

Obstacles are Box2D sensors: not solid, never terminate the episode, and don't change the reward.
Each obstacle counts as hit at most once per episode:
  info["obstacle_hit"]   True only on the step a new obstacle is first hit
  info["obstacle_hits"]  cumulative number of obstacles hit this episode
Per-obstacle records live in `env.unwrapped.obstacles`: dicts with x, y, track_idx, hit.
"""

import math

import gymnasium as gym
import numpy as np
from Box2D.b2 import fixtureDef, polygonShape
from gymnasium.envs.box2d.car_racing import CarRacing, FrictionDetector
from gymnasium.utils import EzPickle

OBSTACLE_COLOR = (139, 69, 19)  # brown; not used anywhere else in CarRacing's render
START_SKIP_FRAC = 0.1  # skip the first 10% of the track (camera zoom-in at the start)
SEGMENT_MARGIN_FRAC = 0.15  # pick from the middle 70% of each segment -> min spacing between obstacles
CURVATURE_WINDOW = 5  # track points on each side used to measure curvature (~17.5 units)
MAX_CURVATURE = 0.15  # radians of total heading change in the window to count as "straight"


class ObstacleDetector(FrictionDetector):
    """FrictionDetector that also marks obstacles as hit when the car touches them.

    FrictionDetector itself ignores obstacle contacts: it only acts when one body has
    `road_friction`, which obstacles don't, so obstacles never count as tile visits.
    """

    def BeginContact(self, contact):
        super().BeginContact(contact)
        for body in (contact.fixtureA.body, contact.fixtureB.body):
            if body.userData is not None and hasattr(body.userData, "obstacle_id"):
                # Static bodies only touch dynamic ones, so the other body is part of the car.
                self.env.obstacles[body.userData.obstacle_id]["hit"] = True


class CarRacingObstacles(CarRacing):
    def __init__(self, num_obstacles: int = 5, obstacle_size: float = 2.5, **kwargs):
        super().__init__(**kwargs)
        EzPickle.__init__(
            self, num_obstacles=num_obstacles, obstacle_size=obstacle_size, **kwargs
        )
        self.num_obstacles = num_obstacles
        self.obstacle_size = obstacle_size
        self.obstacles = []
        self.obstacle_bodies = []

    def _create_track(self):
        success = super()._create_track()
        if success:
            self._place_obstacles(self._pick_obstacle_indices())
        return success

    def _curvature(self, i):
        """Total absolute heading change over the track points around index i."""
        n = len(self.track)
        total = 0.0
        for k in range(i - CURVATURE_WINDOW, i + CURVATURE_WINDOW):
            d = self.track[(k + 1) % n][1] - self.track[k % n][1]
            total += abs((d + math.pi) % (2 * math.pi) - math.pi)  # wrap to [-pi, pi]
        return total

    def _pick_obstacle_indices(self):
        n = len(self.track)
        usable = np.arange(int(START_SKIP_FRAC * n), n)
        indices = []
        for segment in np.array_split(usable, self.num_obstacles):
            margin = int(SEGMENT_MARGIN_FRAC * len(segment))
            middle = segment[margin : len(segment) - margin]
            curvature = np.array([self._curvature(i) for i in middle])
            straight = middle[curvature <= MAX_CURVATURE]
            if len(straight) > 0:
                indices.append(int(self.np_random.choice(straight)))
            else:  # whole segment curves: fall back to its least-curved point
                indices.append(int(middle[np.argmin(curvature)]))
        return indices

    def _place_obstacles(self, track_indices):
        """Replace all obstacles with squares centered on the given track points."""
        self._destroy_obstacles()
        half = self.obstacle_size / 2
        for i, track_idx in enumerate(track_indices):
            _, beta, x, y = self.track[track_idx]
            # (cos beta, sin beta) points across the road; (-sin beta, cos beta) along it.
            across = np.array([math.cos(beta), math.sin(beta)]) * half
            along = np.array([-math.sin(beta), math.cos(beta)]) * half
            center = np.array([x, y])
            vertices = [
                tuple(center - across - along),
                tuple(center + across - along),
                tuple(center + across + along),
                tuple(center - across + along),
            ]
            body = self.world.CreateStaticBody(
                fixtures=fixtureDef(shape=polygonShape(vertices=vertices), isSensor=True)
            )
            body.userData = body
            body.obstacle_id = i
            body.vertices = vertices
            self.obstacle_bodies.append(body)
            self.obstacles.append(
                {"x": float(x), "y": float(y), "track_idx": int(track_idx), "hit": False}
            )

    def _destroy_obstacles(self):
        for body in self.obstacle_bodies:
            self.world.DestroyBody(body)
        self.obstacle_bodies = []
        self.obstacles = []

    def _destroy(self):
        self._destroy_obstacles()
        super()._destroy()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # CarRacing.reset installs a fresh FrictionDetector each time; swap in ours.
        self.world.contactListener_bug_workaround = ObstacleDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        return obs, info

    def step(self, action):
        hits_before = sum(o["hit"] for o in self.obstacles)
        obs, reward, terminated, truncated, info = super().step(action)
        hits = sum(o["hit"] for o in self.obstacles)
        info["obstacle_hit"] = hits > hits_before
        info["obstacle_hits"] = hits
        return obs, reward, terminated, truncated, info

    def _render_road(self, zoom, translation, angle):
        super()._render_road(zoom, translation, angle)
        for body in self.obstacle_bodies:
            self._draw_colored_polygon(
                self.surf, body.vertices, OBSTACLE_COLOR, zoom, translation, angle
            )


gym.register(
    id="CarRacingObstacles-v0",
    entry_point="il.obstacles:CarRacingObstacles",
    max_episode_steps=1000,
)


if __name__ == "__main__":
    # Play with the keyboard: arrows to steer/gas/brake, Enter to restart, Esc to quit.
    import pygame

    env = CarRacingObstacles(render_mode="human")
    quit = False
    while not quit:
        env.reset()
        total_reward, info = 0.0, {}
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
            action = np.array([steer, 1.0 * keys[pygame.K_UP], 0.8 * keys[pygame.K_DOWN]])
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info["obstacle_hit"]:
                print(f"hit obstacle! ({info['obstacle_hits']}/{len(env.obstacles)})")
        print(f"episode done: reward {total_reward:+.1f}, obstacles hit {info.get('obstacle_hits', 0)}")
    env.close()
