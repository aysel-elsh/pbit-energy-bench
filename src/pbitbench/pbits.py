"""Probabilistic bits (p-bits) and block Gibbs sampling on a bipartite
energy-based model, simulated in software.

A p-bit is a binary unit that randomly flips, biased by its input:

    m = sign(tanh(beta * I) - u),   u ~ Uniform(-1, 1)

(Camsari et al., "Stochastic p-bits for invertible logic", PRX 2017).
Hardware p-bits get ``u`` for free from physical noise; here we draw it from a
PRNG, which is exactly the overhead the benchmark is designed to measure.

The model is bipartite (visible <-> hidden), so each layer can be updated in
one vectorised block, like the block Gibbs samplers in Extropic's ``thrml``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["pbit", "bernoulli_pbit", "PatternRBM"]


def pbit(field: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """+-1 p-bits driven by ``field`` (already multiplied by beta)."""
    u = rng.uniform(-1.0, 1.0, size=field.shape).astype(field.dtype)
    return np.where(np.tanh(field) > u, 1.0, -1.0).astype(field.dtype)


def bernoulli_pbit(field: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """0/1 p-bits: on with probability sigmoid(field)."""
    p = 1.0 / (1.0 + np.exp(-field))
    return (rng.random(field.shape) < p).astype(field.dtype)


@dataclass
class PatternRBM:
    """Restricted Boltzmann machine whose hidden units are stored templates.

    Hidden unit k switches on when the visible state overlaps pattern k by more
    than ``threshold``; visible units are then pulled toward the active
    templates. Recall is block Gibbs sampling started from the noisy input.
    """

    patterns: np.ndarray  # (K, N)
    beta_hidden: float = 20.0
    beta_visible: float = 3.0
    threshold: float = 0.5
    sweeps: int = 5

    def __post_init__(self) -> None:
        self.patterns = np.asarray(self.patterns, dtype=np.float32)
        self._w = self.patterns.T.copy()  # (N, K)

    @property
    def n(self) -> int:
        return self.patterns.shape[1]

    @property
    def k(self) -> int:
        return self.patterns.shape[0]

    def recall(self, noisy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Denoise a batch (B, N) of +-1 inputs; returns (B, N) +-1."""
        v = np.asarray(noisy, dtype=np.float32)
        for _ in range(self.sweeps):
            overlap = (v @ self._w) / self.n  # (B, K) in [-1, 1]
            h = bernoulli_pbit(self.beta_hidden * (overlap - self.threshold), rng)
            v = pbit(self.beta_visible * (h @ self.patterns), rng)
        return v

    def macs_per_item(self) -> int:
        """Multiply-accumulates per input: two N*K products per sweep."""
        return 2 * self.n * self.k * self.sweeps
