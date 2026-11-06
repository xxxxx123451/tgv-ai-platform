from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class GNNEnsemble:
    """Aggregate independently seeded models into epistemic mean/std predictions."""

    def __init__(self, models: Sequence[Any]) -> None:
        if len(models) < 2:
            raise ValueError("可靠的模型集成不确定度至少需要 2 个独立模型。")
        self.models = list(models)

    def predict_with_uncertainty(self, loader: Any) -> dict[str, np.ndarray]:
        predictions = np.stack([model.predict(loader) for model in self.models], axis=0)
        return {
            "prediction_mean": predictions.mean(axis=0),
            "prediction_std": predictions.std(axis=0, ddof=0),
            "uncertainty_available": np.ones(predictions.shape[1], dtype=bool),
        }
