# rl-car-racing

Learning imitation learning on Gymnasium's `CarRacing-v3`: record human demonstrations,
train behavioral cloning policies (MSE CNN, then Gaussian, mixture of Gaussians,
discretized/autoregressive, and flow-matching heads), and study compounding errors with HG-DAgger.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install swig            # needed first: Box2D builds against it
pip install -r requirements.txt
```

## Run

Drive the car yourself with the built-in demo (arrow keys to steer/gas/brake):

```bash
python -m gymnasium.envs.box2d.car_racing
```

## Layout

- `il/` — project code
- `notes/log.md` — lab notebook
- `data/`, `runs/`, `videos/` — demonstrations, training runs, recordings (git-ignored)
