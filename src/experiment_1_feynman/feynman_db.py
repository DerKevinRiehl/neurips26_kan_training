"""Vectorized Feynman benchmark functions for experiment 1.

The registry below starts with the Feynman equations used in the KAN
paper's Feynman benchmark table. It is intentionally data-like: adding the
remaining AI-Feynman equations should only require appending entries.

The domains are current safe plotting/training ranges chosen from the
dimensionless formulas and singularities. They are not yet the canonical
AI-Feynman ranges; use ``feynman_db_visualization.py`` to inspect and tune them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


Array = np.ndarray
EvalFn = Callable[[Array], Array]


@dataclass(frozen=True)
class FeynmanFunction:
    """One vectorized scalar or vector-valued Feynman equation."""

    name: str
    expression: str
    variables: tuple[str, ...]
    domains: tuple[tuple[float, float], ...]
    fn: EvalFn

    @property
    def input_dim(self) -> int:
        return len(self.variables)

    def evaluate(self, x: Array) -> Array:
        """Evaluate on shape ``(n_samples, input_dim)`` and return 2D output."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"{self.name} expects input shape (n, {self.input_dim}), got {x.shape}."
            )
        y = np.asarray(self.fn(x), dtype=np.float64)
        if y.ndim == 0:
            y = np.full((x.shape[0], 1), float(y), dtype=np.float64)
        elif y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                f"{self.name} returned {y.shape[0]} rows for {x.shape[0]} inputs."
            )
        return y

    def sample(
        self,
        n_samples: int,
        seed: int | np.random.Generator | None = None,
        *,
        dtype: np.dtype = np.float32,
        max_attempts: int = 100,
    ) -> tuple[Array, Array]:
        """Draw finite samples uniformly from the equation domain."""
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        lows = np.array([domain[0] for domain in self.domains], dtype=np.float64)
        highs = np.array([domain[1] for domain in self.domains], dtype=np.float64)
        xs: list[Array] = []
        ys: list[Array] = []
        remaining = n_samples

        for _ in range(max_attempts):
            if remaining <= 0:
                break
            batch_size = max(remaining * 2, 128)
            candidates = rng.uniform(lows, highs, size=(batch_size, self.input_dim))
            values = self.evaluate(candidates)
            mask = np.isfinite(values).all(axis=1)
            if not np.any(mask):
                continue
            accepted_x = candidates[mask][:remaining]
            accepted_y = values[mask][:remaining]
            xs.append(accepted_x)
            ys.append(accepted_y)
            remaining -= accepted_x.shape[0]

        if remaining > 0:
            raise RuntimeError(
                f"Could not sample {n_samples} finite points for {self.name}; "
                f"{remaining} points missing after {max_attempts} attempts."
            )

        return np.vstack(xs).astype(dtype), np.vstack(ys).astype(dtype)


def _col(x: Array, idx: int) -> Array:
    return x[:, idx]


def _gaussian(theta: Array, sigma: Array) -> Array:
    return np.exp(-(theta**2) / (2.0 * sigma**2)) / np.sqrt(2.0 * np.pi * sigma**2)


def _sinc_sq(z: Array) -> Array:
    out = np.ones_like(z, dtype=np.float64)
    mask = np.abs(z) > 1.0e-12
    out[mask] = (np.sin(z[mask]) / z[mask]) ** 2
    return out


def _entry(
    name: str,
    expression: str,
    variables: Iterable[str],
    domains: Iterable[tuple[float, float]],
    fn: EvalFn,
) -> FeynmanFunction:
    return FeynmanFunction(
        name=name,
        expression=expression,
        variables=tuple(variables),
        domains=tuple(domains),
        fn=fn,
    )


FEYNMAN_FUNCTIONS: tuple[FeynmanFunction, ...] = (
    _entry(
        "I.6.2",
        "exp(-theta^2/(2*sigma^2))/sqrt(2*pi*sigma^2)",
        ("theta", "sigma"),
        ((-3.0, 3.0), (0.5, 2.0)),
        lambda x: _gaussian(_col(x, 0), _col(x, 1)),
    ),
    _entry(
        "I.6.2b",
        "normal(theta; 0, sigma) / normal(theta; theta1, sigma)",
        ("theta", "theta1", "sigma"),
        ((-3.0, 3.0), (-3.0, 3.0), (0.5, 2.0)),
        lambda x: _gaussian(_col(x, 0), _col(x, 2))
        / _gaussian(_col(x, 0) - _col(x, 1), _col(x, 2)),
    ),
    _entry(
        "I.9.18",
        "a / ((b - 1)^2 + (c - d)^2 + (e - f)^2)",
        ("a", "b", "c", "d", "e", "f"),
        ((0.1, 2.0), (-1.0, 0.5), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        lambda x: _col(x, 0)
        / ((_col(x, 1) - 1.0) ** 2 + (_col(x, 2) - _col(x, 3)) ** 2 + (_col(x, 4) - _col(x, 5)) ** 2),
    ),
    _entry(
        "I.12.11",
        "1 + a*sin(theta)",
        ("a", "theta"),
        ((-1.0, 1.0), (-np.pi, np.pi)),
        lambda x: 1.0 + _col(x, 0) * np.sin(_col(x, 1)),
    ),
    _entry(
        "I.13.12",
        "a*(1/b - 1)",
        ("a", "b"),
        ((0.1, 2.0), (0.2, 3.0)),
        lambda x: _col(x, 0) * (1.0 / _col(x, 1) - 1.0),
    ),
    _entry(
        "I.15.3x",
        "sqrt(1 - a) / (1 - b^2)",
        ("a", "b"),
        ((0.0, 0.95), (-0.9, 0.9)),
        lambda x: np.sqrt(1.0 - _col(x, 0)) / (1.0 - _col(x, 1) ** 2),
    ),
    _entry(
        "I.16.6",
        "(a + b) / (1 + a*b)",
        ("a", "b"),
        ((-0.8, 0.8), (-0.8, 0.8)),
        lambda x: (_col(x, 0) + _col(x, 1)) / (1.0 + _col(x, 0) * _col(x, 1)),
    ),
    _entry(
        "I.18.4",
        "(1 + a*b) / (1 + a)",
        ("a", "b"),
        ((0.1, 3.0), (-2.0, 2.0)),
        lambda x: (1.0 + _col(x, 0) * _col(x, 1)) / (1.0 + _col(x, 0)),
    ),
    _entry(
        "I.26.2",
        "arcsin(n*sin(theta))",
        ("n", "theta"),
        ((0.1, 0.9), (-1.0, 1.0)),
        lambda x: np.arcsin(_col(x, 0) * np.sin(_col(x, 1))),
    ),
    _entry(
        "I.27.6",
        "1 / (1 + a*b)",
        ("a", "b"),
        ((-0.8, 0.8), (-0.8, 0.8)),
        lambda x: 1.0 / (1.0 + _col(x, 0) * _col(x, 1)),
    ),
    _entry(
        "I.29.16",
        "sqrt(1 + a^2 - 2*a*cos(theta1 - theta2))",
        ("a", "theta1", "theta2"),
        ((0.1, 2.0), (-np.pi, np.pi), (-np.pi, np.pi)),
        lambda x: np.sqrt(1.0 + _col(x, 0) ** 2 - 2.0 * _col(x, 0) * np.cos(_col(x, 1) - _col(x, 2))),
    ),
    _entry(
        "I.30.3",
        "sin(n*theta/2)^2 / sin(theta/2)^2",
        ("n", "theta"),
        ((1.0, 5.0), (0.2, 2.8)),
        lambda x: (np.sin(_col(x, 0) * _col(x, 1) / 2.0) ** 2) / (np.sin(_col(x, 1) / 2.0) ** 2),
    ),
    _entry(
        "I.30.5",
        "arcsin(a/n)",
        ("a", "n"),
        ((-0.9, 0.9), (1.0, 2.0)),
        lambda x: np.arcsin(_col(x, 0) / _col(x, 1)),
    ),
    _entry(
        "I.37.4",
        "sqrt(1 + a + 2*sqrt(a)*cos(delta))",
        ("a", "delta"),
        ((0.05, 3.0), (-np.pi, np.pi)),
        lambda x: np.sqrt(np.maximum(0.0, 1.0 + _col(x, 0) + 2.0 * np.sqrt(_col(x, 0)) * np.cos(_col(x, 1)))),
    ),
    _entry(
        "I.40.1",
        "n0*exp(-a)",
        ("n0", "a"),
        ((0.1, 3.0), (-2.0, 2.0)),
        lambda x: _col(x, 0) * np.exp(-_col(x, 1)),
    ),
    _entry(
        "I.44.4",
        "n*log(a)",
        ("n", "a"),
        ((0.1, 3.0), (0.2, 3.0)),
        lambda x: _col(x, 0) * np.log(_col(x, 1)),
    ),
    _entry(
        "I.50.26",
        "cos(a) + alpha*cos(a)^2",
        ("a", "alpha"),
        ((-np.pi, np.pi), (-2.0, 2.0)),
        lambda x: np.cos(_col(x, 0)) + _col(x, 1) * np.cos(_col(x, 0)) ** 2,
    ),
    _entry(
        "II.2.42",
        "(a - 1)*b",
        ("a", "b"),
        ((-2.0, 2.0), (-2.0, 2.0)),
        lambda x: (_col(x, 0) - 1.0) * _col(x, 1),
    ),
    _entry(
        "II.6.15a",
        "((a - 1)*b) / (4*pi*c*sqrt(a^2 + b^2))",
        ("a", "b", "c"),
        ((0.2, 3.0), (0.2, 3.0), (0.2, 3.0)),
        lambda x: ((_col(x, 0) - 1.0) * _col(x, 1))
        / (4.0 * np.pi * _col(x, 2) * np.sqrt(_col(x, 0) ** 2 + _col(x, 1) ** 2)),
    ),
    _entry(
        "II.11.7",
        "n0*(1 + a*cos(theta))",
        ("n0", "a", "theta"),
        ((0.1, 3.0), (-0.9, 0.9), (-np.pi, np.pi)),
        lambda x: _col(x, 0) * (1.0 + _col(x, 1) * np.cos(_col(x, 2))),
    ),
    _entry(
        "II.11.27",
        "n*alpha / (1 - n*alpha/3)",
        ("n", "alpha"),
        ((0.1, 1.5), (-1.0, 1.0)),
        lambda x: (_col(x, 0) * _col(x, 1)) / (1.0 - _col(x, 0) * _col(x, 1) / 3.0),
    ),
    _entry(
        "II.35.18",
        "n0 / (exp(a) + exp(-a))",
        ("n0", "a"),
        ((0.1, 3.0), (-3.0, 3.0)),
        lambda x: _col(x, 0) / (np.exp(_col(x, 1)) + np.exp(-_col(x, 1))),
    ),
    _entry(
        "II.36.38",
        "a + alpha*b",
        ("a", "alpha", "b"),
        ((-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)),
        lambda x: _col(x, 0) + _col(x, 1) * _col(x, 2),
    ),
    _entry(
        "II.38.3",
        "a*b",
        ("a", "b"),
        ((-2.0, 2.0), (-2.0, 2.0)),
        lambda x: _col(x, 0) * _col(x, 1),
    ),
    _entry(
        "III.9.52",
        "(a/b)*sinc((b - c)/2)^2",
        ("a", "b", "c"),
        ((0.1, 3.0), (0.2, 3.0), (-3.0, 3.0)),
        lambda x: (_col(x, 0) / _col(x, 1)) * _sinc_sq((_col(x, 1) - _col(x, 2)) / 2.0),
    ),
    _entry(
        "III.10.19",
        "sqrt(1 + a^2 + b^2)",
        ("a", "b"),
        ((-2.0, 2.0), (-2.0, 2.0)),
        lambda x: np.sqrt(1.0 + _col(x, 0) ** 2 + _col(x, 1) ** 2),
    ),
    _entry(
        "III.17.37",
        "beta*(1 + alpha*cos(theta))",
        ("alpha", "beta", "theta"),
        ((-0.9, 0.9), (0.1, 3.0), (-np.pi, np.pi)),
        lambda x: _col(x, 1) * (1.0 + _col(x, 0) * np.cos(_col(x, 2))),
    ),
)

_BY_NAME = {function.name: function for function in FEYNMAN_FUNCTIONS}


def list_functions(
    *,
    min_inputs: int | None = None,
    max_inputs: int | None = None,
) -> tuple[FeynmanFunction, ...]:
    """Return registered equations, optionally filtered by input dimension."""
    functions = FEYNMAN_FUNCTIONS
    if min_inputs is not None:
        functions = tuple(f for f in functions if f.input_dim >= min_inputs)
    if max_inputs is not None:
        functions = tuple(f for f in functions if f.input_dim <= max_inputs)
    return functions


def get_function(name: str) -> FeynmanFunction:
    """Look up one equation by Feynman identifier."""
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BY_NAME))
        raise KeyError(f"Unknown Feynman function {name!r}. Available: {available}") from exc


def sample_function(
    name: str,
    n_samples: int,
    seed: int | np.random.Generator | None = None,
    *,
    dtype: np.dtype = np.float32,
) -> tuple[Array, Array]:
    """Convenience wrapper to sample a registered equation by name."""
    return get_function(name).sample(n_samples, seed=seed, dtype=dtype)
