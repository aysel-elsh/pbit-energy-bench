"""Conventional baseline: a small deterministic MLP denoiser (numpy only)."""

from __future__ import annotations

import numpy as np

from .task import DenoiseTask

__all__ = ["MLP"]


class MLP:
    """N -> H -> N, tanh activations, trained with plain SGD on MSE."""

    def __init__(self, n: int, hidden: int, rng: np.random.Generator):
        self.n, self.hidden = n, hidden
        self.w1 = (rng.standard_normal((n, hidden)) / np.sqrt(n)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.w2 = (rng.standard_normal((hidden, n)) / np.sqrt(hidden)).astype(np.float32)
        self.b2 = np.zeros(n, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(x @ self.w1 + self.b1)
        return np.tanh(h @ self.w2 + self.b2)

    def predict(self, noisy: np.ndarray) -> np.ndarray:
        return np.where(self.forward(noisy) >= 0, 1.0, -1.0).astype(np.float32)

    def fit(self, task: DenoiseTask, rng: np.random.Generator, steps: int = 2000,
            batch: int = 64, lr: float = 0.05) -> "MLP":
        for _ in range(steps):
            x, y = task.sample(batch, rng)
            h = np.tanh(x @ self.w1 + self.b1)
            out = np.tanh(h @ self.w2 + self.b2)
            d_out = (out - y) * (1 - out**2) / batch
            d_h = (d_out @ self.w2.T) * (1 - h**2)
            self.w2 -= lr * h.T @ d_out
            self.b2 -= lr * d_out.sum(0)
            self.w1 -= lr * x.T @ d_h
            self.b1 -= lr * d_h.sum(0)
        return self

    def macs_per_item(self) -> int:
        return 2 * self.n * self.hidden
