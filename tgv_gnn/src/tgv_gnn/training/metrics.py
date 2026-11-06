from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # Average ties without requiring scipy.
    for value in np.unique(values):
        mask = values == value
        ranks[mask] = ranks[mask].mean()
    return ranks


def _base_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")
    if len(y_true) >= 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        spearman = float(np.corrcoef(_rank(y_true), _rank(y_pred))[0, 1])
    else:
        spearman = float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman, "n": int(len(y_true))}


def regression_metrics(y_true: Any, y_pred: Any, groups: Any | None = None) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("y_true 与 y_pred 必须为等长非空一维数组。")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("指标输入必须为有限数值。")
    report: dict[str, Any] = _base_metrics(truth, pred)
    report["by_group"] = {}
    if groups is not None:
        labels = np.asarray(groups, dtype=str)
        if labels.shape != truth.shape:
            raise ValueError("groups 必须与目标等长。")
        report["by_group"] = {
            str(label): _base_metrics(truth[labels == label], pred[labels == label])
            for label in sorted(np.unique(labels))
        }
    return report
