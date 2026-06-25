"""Regression metrics for the Feynman function-approximation benchmark."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


EPSILON = 1e-12


def _to_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _validate_inputs(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true_array = _to_numpy(y_true)
    y_pred_array = _to_numpy(y_pred)
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have the same flattened shape, "
            f"got {y_true_array.shape} and {y_pred_array.shape}."
        )
    if y_true_array.size == 0:
        raise ValueError("y_true and y_pred must contain at least one value.")
    return y_true_array, y_pred_array


def mse(y_true, y_pred) -> float:
    y_true_array, y_pred_array = _validate_inputs(y_true, y_pred)
    return float(np.mean((y_pred_array - y_true_array) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    y_true_array, y_pred_array = _validate_inputs(y_true, y_pred)
    return float(np.mean(np.abs(y_pred_array - y_true_array)))


def nrmse(y_true, y_pred, epsilon: float = EPSILON) -> float:
    y_true_array, y_pred_array = _validate_inputs(y_true, y_pred)
    denominator = np.std(y_true_array) + epsilon
    return float(np.sqrt(np.mean((y_pred_array - y_true_array) ** 2)) / denominator)


def relative_l2(y_true, y_pred, epsilon: float = EPSILON) -> float:
    y_true_array, y_pred_array = _validate_inputs(y_true, y_pred)
    numerator = np.linalg.norm(y_pred_array - y_true_array)
    denominator = np.linalg.norm(y_true_array) + epsilon
    return float(numerator / denominator)


def r2_score(y_true, y_pred, epsilon: float = EPSILON) -> float:
    y_true_array, y_pred_array = _validate_inputs(y_true, y_pred)
    residual_sum_squares = np.sum((y_pred_array - y_true_array) ** 2)
    total_sum_squares = np.sum((y_true_array - np.mean(y_true_array)) ** 2)
    return float(1.0 - residual_sum_squares / (total_sum_squares + epsilon))


def max_absolute_error(y_true, y_pred) -> float:
    y_true_array, y_pred_array = _validate_inputs(y_true, y_pred)
    return float(np.max(np.abs(y_pred_array - y_true_array)))


def evaluate_all(y_true, y_pred) -> Mapping[str, float]:
    return {
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "relative_l2": relative_l2(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "max_absolute_error": max_absolute_error(y_true, y_pred),
    }


class Evaluation:
    """Namespace-style access to the benchmark metrics."""

    mse = staticmethod(mse)
    rmse = staticmethod(rmse)
    mae = staticmethod(mae)
    nrmse = staticmethod(nrmse)
    relative_l2 = staticmethod(relative_l2)
    r2_score = staticmethod(r2_score)
    max_absolute_error = staticmethod(max_absolute_error)
    evaluate_all = staticmethod(evaluate_all)
