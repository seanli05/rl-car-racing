# Lab notebook

## 2026-10-14
- Set up repo: venv, gymnasium[box2d], torch, numpy, matplotlib.
- CarRacing-v3: obs is 96x96x3 uint8; action is [steer in [-1,1], gas in [0,1], brake in [0,1]].

## 2026-10-15
- Added il/obstacles.py: CarRacing subclass with obstacles, registered as CarRacingObstacles-v0
  (import il.obstacles first). Tests in tests/test_obstacles.py (`.venv/bin/python -m pytest tests/`).
- Placement: N obstacles (default 5, 3.5-unit squares) on the track centerline, aligned to track
  heading. Skip first 10% of track (camera zoom), split the rest into N segments, pick a random
  "straight" point (heading change <= 0.15 rad over +-5 points) from the middle 70% of each
  segment (min spacing); fall back to the segment's least-curved point so there are always N.
  Seeded via np_random after track generation, so same seed -> same track and obstacles.
- Why centered on straights: passing left vs right is equally valid -> deliberately bimodal demos.
- Collision: Box2D sensors (not solid), no termination, reward unchanged. Each obstacle counts at
  most once/episode. info["obstacle_hit"] (new hit this step), info["obstacle_hits"] (cumulative).
  Per-obstacle records (x, y, track_idx, hit) in env.unwrapped.obstacles.
- Why no termination: terminating only evaluates obstacles before the first hit, and conflates
  off-track and obstacle failures.
- Color: brown (139, 69, 19), since red is already used by the car, kerbs and gyro indicator.
  Caveat: in grayscale it's close to the road's luminance (~84 vs 102), so keep RGB inputs.
- Tracks are 251-350 points; ~50% of points count as straight at the 0.15 rad threshold.
