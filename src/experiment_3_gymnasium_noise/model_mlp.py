"""Configurable multilayer perceptron for Feynman experiments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn


class GaussianActivation(nn.Module):
    """Elementwise Gaussian radial activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(x**2))


def make_activation(name: str) -> nn.Module:
    """Create an activation module from a compact string name."""
    normalized = name.strip().lower().replace("_", "").replace("-", "")
    if normalized == "relu":
        return nn.ReLU()
    if normalized in {"sigmoid", "logistic"}:
        return nn.Sigmoid()
    if normalized in {"gauss", "gaussian", "rbf"}:
        return GaussianActivation()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized in {"silu", "swish"}:
        return nn.SiLU()
    if normalized in {"identity", "linear", "none"}:
        return nn.Identity()
    raise ValueError(
        "Unknown activation "
        f"{name!r}. Supported: relu, sigmoid, gauss, tanh, silu, identity."
    )


def _load_torch_checkpoint(path: str | Path, map_location: str | torch.device | None = None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class MLP(nn.Module):
    """Fully connected MLP with a configurable hidden activation.

    Parameters
    ----------
    layer_dims:
        Neuron counts including input and output dimensions.
        For example, ``[5, 3, 3]`` creates Linear(5, 3), activation,
        Linear(3, 3).
    base_function:
        Hidden activation function name: ``relu``, ``sigmoid``, ``gauss``,
        ``tanh``, ``silu``, or ``identity``.
    output_activation:
        Optional activation applied after the final linear layer.
    """

    def __init__(
        self,
        layer_dims: Sequence[int],
        base_function: str = "relu",
        *,
        output_activation: str | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must include at least input and output dimensions.")
        if any(dim <= 0 for dim in layer_dims):
            raise ValueError(f"All layer dimensions must be positive, got {layer_dims}.")

        modules: list[nn.Module] = []
        for idx, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            modules.append(nn.Linear(in_dim, out_dim, bias=bias))
            is_last = idx == len(layer_dims) - 2
            if not is_last:
                modules.append(make_activation(base_function))
            elif output_activation is not None:
                modules.append(make_activation(output_activation))

        self.layer_dims = tuple(int(dim) for dim in layer_dims)
        self.base_function = base_function
        self.output_activation = output_activation
        self.bias = bias
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(parameter.numel() for parameter in params if parameter.requires_grad)
        return sum(parameter.numel() for parameter in params)

    def get_n_parameters(self, trainable_only: bool = True) -> int:
        return self.count_parameters(trainable_only=trainable_only)

    def get_config(self) -> dict[str, object]:
        return {
            "layer_dims": list(self.layer_dims),
            "base_function": self.base_function,
            "output_activation": self.output_activation,
            "bias": self.bias,
        }

    def save(self, path: str | Path) -> None:
        checkpoint = {
            "class_name": self.__class__.__name__,
            "config": self.get_config(),
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> "MLP":
        checkpoint = _load_torch_checkpoint(Path(path), map_location=map_location)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model


def get_n_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in params if parameter.requires_grad)
    return sum(parameter.numel() for parameter in params)


"""
Example:

from model_mlp import MLP

model = MLP(layer_dims=[5, 32, 32, 1], base_function="relu")
y_hat = model(x_batch)
num_params = model.get_n_parameters()
"""
