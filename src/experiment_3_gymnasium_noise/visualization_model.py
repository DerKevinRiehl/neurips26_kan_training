"""Matplotlib visualizers for MLP and KAN experiment models."""

from __future__ import annotations

import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


class ModelVisualizer:
    """Visualize MLP structure and KAN edge transfer functions."""

    def __init__(
        self,
        *,
        max_nodes_per_layer: int = 12,
        edge_alpha: float = 0.16,
        node_size: float = 220.0,
    ) -> None:
        self.max_nodes_per_layer = max_nodes_per_layer
        self.edge_alpha = edge_alpha
        self.node_size = node_size

    def visualize(self, model: nn.Module, **kwargs):
        if self._looks_like_kan(model):
            return self.visualize_kan(model, **kwargs)
        return self.visualize_mlp(model, **kwargs)

    def visualize_mlp(
        self,
        model: nn.Module,
        *,
        show_transfer_function: bool = True,
        activation_range: tuple[float, float] = (-4.0, 4.0),
        num_points: int = 400,
        show: bool = True,
        max_nodes_per_layer: int | None = None,
    ):
        layer_dims = self._infer_mlp_layer_dims(model)
        if show_transfer_function:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(max(8.5, 1.8 * len(layer_dims) + 3.0), 4.8),
                gridspec_kw={"width_ratios": [2.5, 1.0]},
            )
            structure_ax, transfer_ax = axes
        else:
            fig, structure_ax = plt.subplots(figsize=(max(6.0, 1.8 * len(layer_dims)), 4.8))
            transfer_ax = None
            axes = structure_ax

        self._draw_network_structure(
            layer_dims=layer_dims,
            ax=structure_ax,
            title="MLP structure",
            max_nodes_per_layer=max_nodes_per_layer or self.max_nodes_per_layer,
            edge_label_fn=None,
            layer_transfer_label=getattr(model, "base_function", None),
        )

        if transfer_ax is not None:
            self._plot_mlp_transfer_function(
                model,
                transfer_ax,
                activation_range=activation_range,
                num_points=num_points,
            )

        fig.tight_layout()
        if show:
            plt.show()
        return fig, axes

    def visualize_kan(
        self,
        model: nn.Module,
        *,
        num_points: int = 300,
        include_base: bool = True,
        max_nodes_per_layer: int | None = None,
        max_edge_labels: int = 80,
        max_transfer_columns: int = 4,
        show: bool = True,
    ) -> dict[str, object]:
        structure = self.visualize_kan_structure(
            model,
            max_nodes_per_layer=max_nodes_per_layer,
            max_edge_labels=max_edge_labels,
            show=False,
        )
        transfer_functions = self.visualize_kan_transfer_functions(
            model,
            num_points=num_points,
            include_base=include_base,
            max_columns=max_transfer_columns,
            show=False,
        )
        if show:
            plt.show()
        return {
            "structure": structure,
            "transfer_functions": transfer_functions,
        }

    def visualize_kan_structure(
        self,
        model: nn.Module,
        *,
        show: bool = True,
        max_nodes_per_layer: int | None = None,
        max_edge_labels: int = 80,
    ):
        layer_dims = self._infer_kan_network_dims(model)
        visible_layer_dims = [
            min(dim, max_nodes_per_layer or self.max_nodes_per_layer) for dim in layer_dims
        ]
        visible_edge_count = sum(
            left * right for left, right in zip(visible_layer_dims[:-1], visible_layer_dims[1:])
        )
        show_edge_labels = visible_edge_count <= max_edge_labels

        fig, ax = plt.subplots(figsize=(max(7.0, 2.0 * len(layer_dims)), 5.2))
        self._draw_network_structure(
            layer_dims=layer_dims,
            ax=ax,
            title="KAN structure with edge transfer-function names",
            max_nodes_per_layer=max_nodes_per_layer or self.max_nodes_per_layer,
            edge_label_fn=(
                lambda layer_idx, input_idx, output_idx: self._edge_name(
                    layer_idx,
                    input_idx,
                    output_idx,
                )
                if show_edge_labels
                else None
            ),
            layer_transfer_label=None,
        )
        if not show_edge_labels:
            ax.text(
                0.5,
                -0.17,
                f"Edge labels hidden because {visible_edge_count} visible edges would overlap.",
                ha="center",
                va="top",
                fontsize=9,
            )
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def visualize_kan_transfer_functions(
        self,
        model: nn.Module,
        *,
        layer_index: int | None = None,
        num_points: int = 300,
        include_base: bool = True,
        max_columns: int = 4,
        max_edges: int | None = None,
        show: bool = True,
    ):
        edge_infos = list(self._iter_kan_edge_infos(model, layer_index=layer_index))
        if max_edges is not None:
            edge_infos = edge_infos[:max_edges]
        if not edge_infos:
            raise ValueError("No KAN edges found to visualize.")

        n_edges = len(edge_infos)
        n_cols = min(max_columns, n_edges)
        n_rows = math.ceil(n_edges / n_cols)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(max(4.0, 3.1 * n_cols), max(2.6, 2.45 * n_rows)),
            squeeze=False,
        )

        for ax, edge_info in zip(axes.ravel(), edge_infos):
            details = self._edge_details(
                edge_info["layer"],
                input_index=edge_info["input_index"],
                output_index=edge_info["output_index"],
                num_points=num_points,
                include_base=include_base,
            )
            ax.plot(details["x"], details["y"], color="#2563eb", linewidth=1.35)
            self._draw_basis_markers(ax, details["basis_markers"])
            ax.axhline(0.0, color="#94a3b8", linewidth=0.8, alpha=0.7)
            ax.set_title(
                f"{edge_info['edge_name']}: {edge_info['source']} -> {edge_info['target']}\n"
                f"{details['parameter_note']}; {details['basis_note']}",
                fontsize=8,
            )
            ax.tick_params(axis="both", labelsize=7)

        for ax in axes.ravel()[n_edges:]:
            ax.axis("off")

        title_layer = "all layers" if layer_index is None else f"layer {layer_index}"
        fig.suptitle(f"KAN transfer functions, {title_layer}", fontsize=12)
        fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
        if show:
            plt.show()
        return fig, axes

    def visualize_kan_edges(
        self,
        model: nn.Module,
        *,
        layer_index: int = 0,
        max_inputs: int = 6,
        max_outputs: int = 6,
        num_points: int = 300,
        include_base: bool = True,
        show: bool = True,
    ):
        layer = self._get_kan_layer(model, layer_index)
        input_dim, output_dim = self._infer_kan_layer_dims(layer)
        input_count = min(input_dim, max_inputs)
        output_count = min(output_dim, max_outputs)

        fig, axes = plt.subplots(
            output_count,
            input_count,
            figsize=(max(4.0, 2.55 * input_count), max(2.6, 2.2 * output_count)),
            squeeze=False,
        )

        for output_index in range(output_count):
            for input_index in range(input_count):
                edge_name = self._edge_name(layer_index, input_index, output_index)
                ax = axes[output_index, input_index]
                details = self._edge_details(
                    layer,
                    input_index=input_index,
                    output_index=output_index,
                    num_points=num_points,
                    include_base=include_base,
                )
                ax.plot(details["x"], details["y"], color="#2563eb", linewidth=1.35)
                self._draw_basis_markers(ax, details["basis_markers"])
                ax.axhline(0.0, color="#94a3b8", linewidth=0.8, alpha=0.7)
                ax.set_title(
                    f"{edge_name}: x{input_index} -> h{output_index}\n"
                    f"{details['parameter_note']}; {details['basis_note']}",
                    fontsize=8,
                )
                ax.tick_params(axis="both", labelsize=7)

        if input_dim > input_count or output_dim > output_count:
            fig.text(
                0.5,
                0.01,
                f"Showing {output_count}/{output_dim} outputs and {input_count}/{input_dim} inputs.",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.suptitle(f"KAN transfer functions, layer {layer_index}", fontsize=12)
        fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        if show:
            plt.show()
        return fig, axes

    def _draw_network_structure(
        self,
        *,
        layer_dims: list[int],
        ax,
        title: str,
        max_nodes_per_layer: int,
        edge_label_fn,
        layer_transfer_label: str | None,
    ) -> None:
        node_positions: list[list[tuple[float, float]]] = []
        x_positions = np.linspace(0.0, 1.0, len(layer_dims))

        for layer_index, (x_pos, layer_width) in enumerate(zip(x_positions, layer_dims)):
            visible_nodes = min(layer_width, max_nodes_per_layer)
            y_positions = (
                np.array([0.5]) if visible_nodes == 1 else np.linspace(0.0, 1.0, visible_nodes)
            )
            positions = [(x_pos, y_pos) for y_pos in y_positions]
            node_positions.append(positions)

            ax.scatter(
                [position[0] for position in positions],
                [position[1] for position in positions],
                s=self.node_size,
                color="#ffffff",
                edgecolor="#1f2937",
                linewidth=1.2,
                zorder=3,
            )

            for node_index, (node_x, node_y) in enumerate(positions):
                ax.text(
                    node_x,
                    node_y,
                    self._node_name(layer_index, node_index, len(layer_dims)),
                    ha="center",
                    va="center",
                    fontsize=7,
                    zorder=4,
                )

            if layer_width > visible_nodes:
                ax.text(
                    x_pos,
                    -0.12,
                    f"{layer_width} units",
                    ha="center",
                    va="top",
                    fontsize=9,
                )

            layer_name = self._layer_label(layer_index, len(layer_dims))
            ax.text(
                x_pos,
                1.12,
                f"{layer_name}\n{layer_width}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        for layer_index, (left_positions, right_positions) in enumerate(
            zip(node_positions[:-1], node_positions[1:])
        ):
            for input_index, (x_left, y_left) in enumerate(left_positions):
                for output_index, (x_right, y_right) in enumerate(right_positions):
                    ax.plot(
                        [x_left, x_right],
                        [y_left, y_right],
                        color="#64748b",
                        alpha=self.edge_alpha,
                        linewidth=0.8,
                        zorder=1,
                    )
                    if edge_label_fn is not None:
                        edge_label = edge_label_fn(layer_index, input_index, output_index)
                        if edge_label is not None:
                            ax.text(
                                (x_left + x_right) / 2.0,
                                (y_left + y_right) / 2.0,
                                edge_label,
                                ha="center",
                                va="center",
                                fontsize=6,
                                color="#334155",
                                alpha=0.82,
                                zorder=2,
                            )

            if layer_transfer_label is not None and layer_index < len(layer_dims) - 2:
                ax.text(
                    (x_positions[layer_index] + x_positions[layer_index + 1]) / 2.0,
                    1.02,
                    f"transfer: {layer_transfer_label}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#334155",
                )

        ax.set_title(title)
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.22, 1.2)
        ax.axis("off")

    def _plot_mlp_transfer_function(
        self,
        model: nn.Module,
        ax,
        *,
        activation_range: tuple[float, float],
        num_points: int,
    ) -> None:
        activation_module, activation_name = self._infer_mlp_activation(model)
        x_values = torch.linspace(activation_range[0], activation_range[1], num_points)
        with torch.no_grad():
            y_values = activation_module(x_values).detach().cpu().numpy()

        x_numpy = x_values.detach().cpu().numpy()
        ax.plot(x_numpy, y_values, color="#2563eb", linewidth=1.6)
        ax.axhline(0.0, color="#94a3b8", linewidth=0.8, alpha=0.7)
        ax.axvline(0.0, color="#94a3b8", linewidth=0.8, alpha=0.7)
        ax.set_title(f"MLP transfer function\n{activation_name}")
        ax.set_xlabel("pre-activation")
        ax.set_ylabel("post-activation")
        ax.grid(True, alpha=0.25)

    @staticmethod
    def _infer_mlp_activation(model: nn.Module) -> tuple[nn.Module, str]:
        activation_name = getattr(model, "base_function", None)
        for module in model.modules():
            if module is model:
                continue
            if isinstance(module, (nn.Sequential, nn.Linear)):
                continue
            if list(module.children()):
                continue
            label = activation_name or module.__class__.__name__
            return module.cpu(), str(label)
        return nn.Identity(), str(activation_name or "identity")

    @staticmethod
    def _layer_label(layer_index: int, layer_count: int) -> str:
        if layer_index == 0:
            return "input"
        if layer_index == layer_count - 1:
            return "output"
        return f"hidden {layer_index}"

    @staticmethod
    def _node_name(layer_index: int, node_index: int, layer_count: int) -> str:
        if layer_index == 0:
            return f"x{node_index}"
        if layer_index == layer_count - 1:
            return f"y{node_index}"
        return f"h{layer_index}_{node_index}"

    @staticmethod
    def _edge_name(layer_index: int, input_index: int, output_index: int) -> str:
        return f"e{layer_index}_{input_index}_{output_index}"

    @staticmethod
    def _infer_mlp_layer_dims(model: nn.Module) -> list[int]:
        if hasattr(model, "layer_dims"):
            return [int(dim) for dim in getattr(model, "layer_dims")]

        linear_layers = [module for module in model.modules() if isinstance(module, nn.Linear)]
        if not linear_layers:
            raise ValueError("Could not infer MLP layer dimensions from this model.")
        return [linear_layers[0].in_features] + [
            layer.out_features for layer in linear_layers
        ]

    @staticmethod
    def _looks_like_kan(model: nn.Module) -> bool:
        if hasattr(model, "model_family"):
            return True
        backend = getattr(model, "model", model)
        return hasattr(backend, "layers") and not hasattr(model, "net")

    @staticmethod
    def _get_kan_layers(model: nn.Module):
        backend = getattr(model, "model", model)
        layers = getattr(backend, "layers", None)
        if layers is None:
            raise ValueError("Could not find a KAN layers attribute on this model.")
        return layers

    def _get_kan_layer(self, model: nn.Module, layer_index: int) -> nn.Module:
        layers = self._get_kan_layers(model)
        if layer_index < 0 or layer_index >= len(layers):
            raise IndexError(f"KAN layer_index {layer_index} is out of range.")
        return layers[layer_index]

    def _infer_kan_network_dims(self, model: nn.Module) -> list[int]:
        if hasattr(model, "layer_dims"):
            return [int(dim) for dim in getattr(model, "layer_dims")]
        layers = self._get_kan_layers(model)
        first_input, _ = self._infer_kan_layer_dims(layers[0])
        return [first_input] + [self._infer_kan_layer_dims(layer)[1] for layer in layers]

    @staticmethod
    def _infer_kan_layer_dims(layer: nn.Module) -> tuple[int, int]:
        if hasattr(layer, "input_dim") and hasattr(layer, "output_dim"):
            return int(layer.input_dim), int(layer.output_dim)
        if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
            return int(layer.in_features), int(layer.out_features)
        raise ValueError("Could not infer input and output dimensions for this KAN layer.")

    def _iter_kan_edge_infos(self, model: nn.Module, *, layer_index: int | None):
        layers = self._get_kan_layers(model)
        layer_dims = self._infer_kan_network_dims(model)
        for current_layer_index, layer in enumerate(layers):
            if layer_index is not None and current_layer_index != layer_index:
                continue
            input_dim, output_dim = self._infer_kan_layer_dims(layer)
            for input_index in range(input_dim):
                for output_index in range(output_dim):
                    yield {
                        "layer": layer,
                        "layer_index": current_layer_index,
                        "input_index": input_index,
                        "output_index": output_index,
                        "edge_name": self._edge_name(
                            current_layer_index,
                            input_index,
                            output_index,
                        ),
                        "source": self._node_name(
                            current_layer_index,
                            input_index,
                            len(layer_dims),
                        ),
                        "target": self._node_name(
                            current_layer_index + 1,
                            output_index,
                            len(layer_dims),
                        ),
                    }

    def _edge_details(
        self,
        layer: nn.Module,
        *,
        input_index: int,
        output_index: int,
        num_points: int,
        include_base: bool,
    ) -> dict[str, object]:
        with torch.no_grad():
            if hasattr(layer, "spline_linear") and hasattr(layer, "rbf"):
                x_values, y_values = self._fastkan_edge_curve(
                    layer,
                    input_index=input_index,
                    output_index=output_index,
                    num_points=num_points,
                    include_base=include_base,
                )
                centers = layer.rbf.grid.detach().cpu().numpy()
                return {
                    "x": x_values,
                    "y": y_values,
                    "basis_markers": [("RBF centers", centers, "#dc2626", "--")],
                    "basis_note": f"{len(centers)} RBF centers",
                    "parameter_note": self._edge_parameter_note(layer),
                }

            if hasattr(layer, "scaled_spline_weight"):
                x_values, y_values = self._efficientkan_edge_curve(
                    layer,
                    input_index=input_index,
                    output_index=output_index,
                    num_points=num_points,
                    include_base=include_base,
                )
                knots = layer.grid[input_index].detach().cpu().numpy()
                return {
                    "x": x_values,
                    "y": y_values,
                    "basis_markers": [("B-spline knots", knots, "#dc2626", ":")],
                    "basis_note": f"{len(knots)} knot positions",
                    "parameter_note": self._edge_parameter_note(layer),
                }

            if hasattr(layer, "spline_weight") and hasattr(layer, "b_splines"):
                x_values, y_values = self._bsrbf_edge_curve(
                    layer,
                    input_index=input_index,
                    output_index=output_index,
                    num_points=num_points,
                    include_base=include_base,
                )
                knots = layer.grid[input_index].detach().cpu().numpy()
                centers = layer.rbf.grid.detach().cpu().numpy()
                return {
                    "x": x_values,
                    "y": y_values,
                    "basis_markers": [
                        ("B-spline knots", knots, "#dc2626", ":"),
                        ("RBF centers", centers, "#16a34a", "--"),
                    ],
                    "basis_note": f"{len(knots)} knots, {len(centers)} centers",
                    "parameter_note": self._edge_parameter_note(layer),
                }

        raise ValueError(f"Unsupported KAN layer type: {type(layer).__name__}.")

    @staticmethod
    def _draw_basis_markers(ax, basis_markers: list[tuple[str, np.ndarray, str, str]]) -> None:
        used_labels: set[str] = set()
        for label, positions, color, linestyle in basis_markers:
            for position_index, position in enumerate(positions):
                marker_label = label if label not in used_labels and position_index == 0 else None
                ax.axvline(
                    float(position),
                    color=color,
                    linestyle=linestyle,
                    linewidth=0.7,
                    alpha=0.28,
                    label=marker_label,
                )
            used_labels.add(label)
        if basis_markers:
            ax.legend(fontsize=6, loc="best")

    @staticmethod
    def _edge_parameter_note(layer: nn.Module) -> str:
        if hasattr(layer, "spline_linear") and hasattr(layer, "rbf"):
            count = int(layer.rbf.num_grids)
            if getattr(layer, "use_base_update", False):
                count += 1
            note = f"{count} edge params"
            if getattr(layer, "use_base_update", False) and layer.base_linear.bias is not None:
                note += " + output bias"
            return note

        if hasattr(layer, "scaled_spline_weight"):
            count = int(layer.spline_weight.shape[-1]) + 1
            if getattr(layer, "enable_standalone_scale_spline", False):
                count += 1
            return f"{count} edge params"

        if hasattr(layer, "spline_weight") and hasattr(layer, "b_splines"):
            basis_width = int(layer.grid_size + layer.spline_order)
            return f"{basis_width + 1} edge params"

        return "unknown edge params"

    @staticmethod
    def _fastkan_edge_curve(
        layer: nn.Module,
        *,
        input_index: int,
        output_index: int,
        num_points: int,
        include_base: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        weight = layer.spline_linear.weight
        device = weight.device
        dtype = weight.dtype
        grid_count = layer.rbf.num_grids
        spacing = layer.rbf.denominator
        x_values = torch.linspace(
            layer.rbf.grid_min - 2 * spacing,
            layer.rbf.grid_max + 2 * spacing,
            num_points,
            device=device,
            dtype=dtype,
        )
        edge_weights = weight[
            output_index,
            input_index * grid_count : (input_index + 1) * grid_count,
        ]
        y_values = (layer.rbf(x_values) * edge_weights).sum(dim=-1)

        if include_base and getattr(layer, "use_base_update", False):
            base_weight = layer.base_linear.weight[output_index, input_index]
            y_values = y_values + base_weight * layer.base_activation(x_values)

        return (
            x_values.detach().cpu().numpy(),
            y_values.detach().cpu().numpy(),
        )

    @staticmethod
    def _efficientkan_edge_curve(
        layer: nn.Module,
        *,
        input_index: int,
        output_index: int,
        num_points: int,
        include_base: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        device = layer.base_weight.device
        dtype = layer.base_weight.dtype
        x_values = _grid_interval(layer, input_index, num_points, device, dtype)
        x_matrix = torch.zeros(num_points, layer.in_features, device=device, dtype=dtype)
        x_matrix[:, input_index] = x_values

        basis = layer.b_splines(x_matrix)[:, input_index, :]
        spline_weights = layer.scaled_spline_weight[output_index, input_index, :]
        y_values = basis @ spline_weights

        if include_base:
            base_weight = layer.base_weight[output_index, input_index]
            y_values = y_values + base_weight * layer.base_activation(x_values)

        return (
            x_values.detach().cpu().numpy(),
            y_values.detach().cpu().numpy(),
        )

    @staticmethod
    def _bsrbf_edge_curve(
        layer: nn.Module,
        *,
        input_index: int,
        output_index: int,
        num_points: int,
        include_base: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        device = layer.base_weight.device
        dtype = layer.base_weight.dtype
        x_values = _grid_interval(layer, input_index, num_points, device, dtype)
        x_matrix = torch.zeros(num_points, layer.input_dim, device=device, dtype=dtype)
        x_matrix[:, input_index] = x_values

        basis = layer.b_splines(x_matrix)[:, input_index, :]
        rbf_values = layer.rbf(x_matrix)[:, input_index, :]
        combined_basis = basis + rbf_values
        basis_width = layer.grid_size + layer.spline_order
        start = input_index * basis_width
        end = start + basis_width
        edge_weights = layer.spline_weight[output_index, start:end]
        y_values = combined_basis @ edge_weights

        if include_base:
            base_weight = layer.base_weight[output_index, input_index]
            y_values = y_values + base_weight * layer.base_activation(x_values)

        return (
            x_values.detach().cpu().numpy(),
            y_values.detach().cpu().numpy(),
        )


def _grid_interval(
    layer: nn.Module,
    input_index: int,
    num_points: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    grid = layer.grid[input_index].to(device=device, dtype=dtype)
    spline_order = int(getattr(layer, "spline_order", 0))
    if spline_order > 0 and grid.numel() > 2 * spline_order:
        left = grid[spline_order]
        right = grid[-spline_order - 1]
    else:
        left = grid[0]
        right = grid[-1]
    return torch.linspace(left.item(), right.item(), num_points, device=device, dtype=dtype)


# ###########################################################################
# Spyder demo 1: visualize an MLP with ReLU activations and transfer function.
# Uncomment this block and run the file or cell in Spyder.

# from model_mlp import MLP

# visualizer = ModelVisualizer(max_nodes_per_layer=8)
# mlp_relu = MLP(layer_dims=[2, 16, 16, 1], base_function="relu")
# visualizer.visualize_mlp(mlp_relu)


# ###########################################################################
# Spyder demo 2: visualize KAN structure and all edge transfer functions.
# These are untrained models, so the curves show random initial edge functions.
# Uncomment this block and run the file or cell in Spyder.

# from model_kan import KAN

# visualizer = ModelVisualizer(max_nodes_per_layer=8)
# kan_spline = KAN(layer_dims=[2, 4, 1], base_function="bspline")
# visualizer.visualize_kan(kan_spline)

# kan_gaussrbf = KAN(layer_dims=[2, 4, 1], base_function="gaussrbf")
# visualizer.visualize_kan(kan_gaussrbf)
