# Lab notebook

## 2026-10-14
- Set up repo: venv, gymnasium[box2d], torch, numpy, matplotlib.
- CarRacing-v3: obs is 96x96x3 uint8; action is [steer in [-1,1], gas in [0,1], brake in [0,1]].

## 2026-10-15
- Added il/obstacles.py: CarRacing subclass with obstacles, registered as CarRacingObstacles-v0
  (import il.obstacles first). Tests in tests/test_obstacles.py (`.venv/bin/python -m pytest tests/`).
- Placement: N obstacles (default 5, 2.5-unit squares) on the track centerline, aligned to track
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
- Added il/speed_control.py: SpeedControl wrapper (not a subclass: changes the action interface,
  not the world; works on CarRacing-v3 and CarRacingObstacles-v0). Action = steering only,
  Box(-1, 1, (1,)). Each step gas = clip(k * (target_speed - speed), 0, 1), brake = 0, where
  speed = norm of hull linearVelocity. info["action"] = applied [steer, gas, brake];
  info["speed"] = speed the controller saw (before the physics step, i.e. matching the obs the
  steering was chosen from). Test-drive: `.venv/bin/python -m il.speed_control --target-speed 50`.
  Tests in tests/test_speed_control.py.
- Choosing k (5 seeds, centerline-following steerer, plots kept out of the repo):
  - Standing start is friction-limited: rear wheels spin, speed rises ~0.875/step regardless of
    gas. So 95% of target takes >= ~target/0.9 steps (53 steps at 50) for any k.
  - Larger k only adds overshoot: gas stays at 1 until close, wheelspin stored in the rear wheels
    keeps pushing after the gas is cut. No drag/brake -> overshoot persists until turns bleed it.
    Target 50: k=0.1 +10%, 0.3 +15%, 1.0 +17%. Target 30: k=0.05 +7%, 0.07 +16%.
  - k=0.04: at the rise floor, ~0% overshoot at targets 30-50, back within ~3% after turns, gas
    mid-turn <= ~0.1 (no rear slides). k=0.01-0.02 too sluggish to replace speed lost in turns.
  - Brake not needed at k=0.04 (left at 0).
- Default target_speed = 50, a starting point only: the simple steerer holds the road at 40-50,
  starts spinning out at 60, and fails badly at 70. Tune by test-driving hairpins and obstacles.

## 2026-10-16
- Settled on target_speed = 35 (test-driving). Added il/tune_speed_control.py to re-run the k sweep
  reproducibly: `sweep` (k x seed grid, pursuit steering, obstacles env, full 1000-step episodes)
  and `replay --k K [--seed S]` (re-drives a logged run, asserting the car state matches the log at
  every step; refuses a k/seed with no finished run). Default saves an MP4 (600x400, 50 fps, ~0.5 MB,
  ~15 s to encode) to videos/tune_speed_control/target35/ via imageio-ffmpeg (added to
  requirements.txt; bundles its own ffmpeg); `--render-mode human` shows the game window instead. One .npz per run in runs/tune_speed_control/target35/ (gitignored):
  applied action, speed, reward, track idx, on-road, hull pose/velocity, per-wheel omega, gas and
  steering angle. Replay works because the env is deterministic (seed + steering), not by
  restoring Box2D state. Plus metrics.csv and plots/ (per-k speed/gas/rear-slip over seeds,
  all-k comparison, metrics vs k, track maps). `plot` redoes metrics.csv and plots/ from the saved
  logs in ~5 s, so changing a metric or plot never needs a sweep rerun.
- Results at 35 (10 ks x 5 seeds): rise floor is 36 steps (full gas from standstill; reached by
  every k >= 0.04). k=0.04: +0.07% overshoot, 0.26% tracking error, 97.6% of time within +-2% of
  target after step 200 (the best; 0.03: 96.8%, 0.01: 82%). k=0.05 already +6.6% overshoot and
  62% in band, 0.1 +20% / 22%, 1.0 +31% / 9%: that overshoot never fully bleeds off in 1000 steps. k=0.01-0.03 slower (50-160 steps) with
  no gain. Why 0.04 is the edge: gas only saturates while error > 1/k = 25, so it starts easing
  off at speed 10 and the rear wheelspin is gone by the time the car reaches 35 (rear slip peaks
  ~27 vs ~33 at k=0.1). Same edge as at targets 30 and 50, so k=0.04 stays the default.
