"""The shared task: recover a stored +-1 pattern from a corrupted copy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DenoiseTask", "make_task"]


@dataclass(frozen=True)
class DenoiseTask:
    patterns: np.ndarray  # (K, N) stored +-1 patterns
    flip_prob: float

    @property
    def n(self) -> int:
        return self.patterns.shape[1]

    @property
    def k(self) -> int:
        return self.patterns.shape[0]

    def sample(self, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """``count`` (noisy, clean) pairs, each (count, N)."""
        idx = rng.integers(0, self.k, size=count)
        clean = self.patterns[idx]
        flips = rng.random(clean.shape) < self.flip_prob
        noisy = np.where(flips, -clean, clean)
        return noisy.astype(np.float32), clean.astype(np.float32)


def make_task(n: int, k: int | None = None, flip_prob: float = 0.1, seed: int = 0) -> DenoiseTask:
    """Random task with ``k`` patterns of length ``n`` (default k = max(2, n // 16))."""
    if n < 4:
        raise ValueError("n must be >= 4")
    if not 0 <= flip_prob < 0.5:
        raise ValueError("flip_prob must be in [0, 0.5)")
    k = k or max(2, n // 16)
    rng = np.random.default_rng(seed)
    patterns = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(k, n))
    return DenoiseTask(patterns, flip_prob)


def bit_accuracy(pred: np.ndarray, clean: np.ndarray) -> float:
    """Fraction of bits recovered correctly."""
    return float(np.mean(np.sign(pred) == clean))
