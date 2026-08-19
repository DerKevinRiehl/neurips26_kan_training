"""Configurable KAN wrapper for Gymnasium experiments.

This module wraps selected models from the cloned All-KAN repository:
https://github.com/hoangthangta/All-KAN
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


_ALL_KAN_MODELS = Path(__file__).resolve().parent.parent / "external" / "All-KAN" / "models"
if _ALL_KAN_MODELS.exists() and str(_ALL_KAN_MODELS) not in sys.path:
    sys.path.insert(0, str(_ALL_KAN_MODELS))

try:
    from bsrbf_kan import BSRBF_KAN
    from efficient_kan import EfficientKAN
    from fast_kan import FastKAN
except ImportError as exc:  # pragma: no cover - clearer error for missing clone
    raise ImportError(
        "Could not import All-KAN models. Expected the repository at "
        f"{_ALL_KAN_MODELS.parent}. Clone it with: "
        "git clone https://github.com/hoangthangta/All-KAN.git "
        "src/external/All-KAN"
    ) from exc


def _activation_class(name: str) -> type[nn.Module]:
    normalized = name.strip().lower().replace("_", "").replace("-", "")
    if normalized in {"silu", "swish"}:
        return nn.SiLU
    if normalized == "relu":
        return nn.ReLU
    if normalized == "gelu":
        return nn.GELU
    if normalized == "tanh":
        return nn.Tanh
    if normalized in {"sigmoid", "logistic"}:
        return nn.Sigmoid
    if normalized == "softplus":
        return nn.Softplus
    raise ValueError(
        "Unknown base activation "
        f"{name!r}. Supported: silu, relu, gelu, tanh, sigmoid, softplus."
    )


def _activation_function(name: str):
    normalized = name.strip().lower().replace("_", "").replace("-", "")
    if normalized in {"silu", "swish"}:
        return F.silu
    if normalized == "relu":
        return F.relu
    if normalized == "gelu":
        return F.gelu
    if normalized == "tanh":
        return torch.tanh
    if normalized in {"sigmoid", "logistic"}:
        return torch.sigmoid
    if normalized == "softplus":
        return F.softplus
    raise ValueError(
        "Unknown base activation "
        f"{name!r}. Supported: silu, relu, gelu, tanh, sigmoid, softplus."
    )


def _load_torch_checkpoint(path: str | Path, map_location: str | torch.device | None = None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class KAN(nn.Module):
    """KAN model with a small, experiment-friendly constructor.

    Parameters
    ----------
    layer_dims:
        Neuron counts including input and output dimensions.
        For example, ``[5, 3, 3]`` creates a 5-to-3-to-3 network.
    base_function:
        ``"bspline"`` uses All-KAN's EfficientKAN.
        ``"gaussrbf"`` or ``"rbf"`` uses All-KAN's FastKAN.
        ``"bsrbf"`` uses All-KAN's BSRBF_KAN hybrid.
    grid_size:
        Spline grid size for B-spline variants.
    spline_order:
        Spline order for B-spline variants.
    num_grids:
        RBF grid count for FastKAN. Defaults to ``grid_size + spline_order``.
    """

    def __init__(
        self,
        layer_dims: Sequence[int],
        base_function: str = "bspline",
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        num_grids: int | None = None,
        grid_range: tuple[float, float] = (-1.5, 1.5),
        base_activation: str = "silu",
        use_layernorm: bool = False,
        norm_type: str = "none",
    ) -> None:
        super().__init__()
        if len(layer_dims) < 2:
            raise ValueError("layer_dims must include at least input and output dimensions.")
        if any(dim <= 0 for dim in layer_dims):
            raise ValueError(f"All layer dimensions must be positive, got {layer_dims}.")

        self.layer_dims = tuple(int(dim) for dim in layer_dims)
        self.base_function = base_function
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.num_grids = num_grids or (grid_size + spline_order)
        self.grid_range = (float(grid_range[0]), float(grid_range[1]))
        self.base_activation = base_activation
        self.use_layernorm = use_layernorm
        self.norm_type = norm_type

        normalized = base_function.strip().lower().replace("_", "").replace("-", "")
        activation = _activation_class(base_activation)

        if normalized in {"bspline", "bsplines", "spline", "efficientkan"}:
            self.model = EfficientKAN(
                list(self.layer_dims),
                grid_size=grid_size,
                spline_order=spline_order,
                base_activation=activation,
                grid_range=list(self.grid_range),
            )
            self.model_family = "efficient_kan"
        elif normalized in {"gaussrbf", "gaussianrbf", "rbf", "fastkan"}:
            self.model = FastKAN(
                list(self.layer_dims),
                grid_min=self.grid_range[0],
                grid_max=self.grid_range[1],
                num_grids=self.num_grids,
                base_activation=_activation_function(base_activation),
                use_layernorm=use_layernorm,
            )
            self.model_family = "fast_kan"
        elif normalized in {"bsrbf", "bsrbfkan"}:
            self.model = BSRBF_KAN(
                list(self.layer_dims),
                grid_size=grid_size,
                spline_order=spline_order,
                base_activation=activation,
                norm_type=norm_type,
            )
            self.model_family = "bsrbf_kan"
        else:
            raise ValueError(
                "Unknown KAN base_function "
                f"{base_function!r}. Supported: bspline, gaussrbf, bsrbf."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "num_grids": self.num_grids,
            "grid_range": self.grid_range,
            "base_activation": self.base_activation,
            "use_layernorm": self.use_layernorm,
            "norm_type": self.norm_type,
        }

    def save(self, path: str | Path) -> None:
        checkpoint = {
            "class_name": self.__class__.__name__,
            "model_family": self.model_family,
            "config": self.get_config(),
            "state_dict": self.state_dict(),
        }
        torch.save(checkpoint, Path(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> "KAN":
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
Examples:

from model_kan import KAN

bspline_model = KAN(layer_dims=[5, 32, 32, 1], base_function="bspline")
rbf_model = KAN(layer_dims=[5, 32, 32, 1], base_function="gaussrbf")
hybrid_model = KAN(layer_dims=[5, 32, 32, 1], base_function="bsrbf")

y_hat = bspline_model(x_batch)
num_params = bspline_model.get_n_parameters()
"""
