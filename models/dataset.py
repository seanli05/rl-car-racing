from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from il.episodes import load_episode


def preprocess(obs):
    """One env observation -> model input: (96, 96, 3) uint8 -> (3, 96, 96) float32 in [0, 1].

    Training (Episodes) and driving (model_testing/watch.py) both call this, so the model always sees inputs
    prepared the same way. Change preprocessing here, never in just one of them.
    """
    return torch.from_numpy(obs).permute(2, 0, 1).float() / 255


def episode_files(folder="data/plain"):
    """Every episode file in folder, sorted so the order is the same on every machine."""
    return sorted(Path(folder).glob("*.npz"))


def split_episodes(folder="data/plain", test_frac=0.2, seed=0):
    """(train_files, test_files): whole episodes, shuffled with a fixed seed.

    Splitting by episode instead of by frame keeps near-identical neighboring frames
    from landing on both sides of the split.
    """
    files = episode_files(folder)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(files))
    n_test = max(1, round(len(files) * test_frac))
    test = [files[i] for i in sorted(order[:n_test])]
    train = [files[i] for i in sorted(order[n_test:])]
    return train, test


class Episodes(Dataset):
    """Frames and steering labels from a list of episode files (default: all of data/plain)."""

    def __init__(self, files=None):
        self.files = list(files) if files is not None else episode_files()
        if not self.files:
            raise ValueError("Episodes got no episode files")

        f, l = [], []
        for path in self.files:
            arrays, meta = load_episode(path)
            f.append(arrays["obs"])
            l.append(arrays["action"])

        self.features = np.concatenate(f)  # (N, 96, 96, 3) uint8
        self.labels = np.concatenate(l)    # (N, 1) float32

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = preprocess(self.features[idx])  # (3, 96, 96), the layout Conv2d expects
        label = torch.from_numpy(self.labels[idx]).float()  # (1,) steering
        return (feature, label)
