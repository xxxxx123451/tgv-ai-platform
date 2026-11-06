from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TargetScaler:
    """Fold-local target standardizer; call ``fit`` on training labels only."""

    mean_: float | None = None
    std_: float | None = None

    def fit(self, values: Any) -> "TargetScaler":
        array = np.asarray(values, dtype=float).reshape(-1)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError("用于拟合 TargetScaler 的训练标签必须为有限非空数组。")
        self.mean_ = float(array.mean())
        std = float(array.std(ddof=0))
        self.std_ = std if std > 1e-12 else 1.0
        return self

    def _check(self) -> tuple[float, float]:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TargetScaler 尚未在训练折上拟合。")
        return self.mean_, self.std_

    def transform(self, values: Any) -> np.ndarray:
        mean, std = self._check()
        return (np.asarray(values, dtype=float) - mean) / std

    def inverse_transform(self, values: Any) -> np.ndarray:
        mean, std = self._check()
        return np.asarray(values, dtype=float) * std + mean

    def to_dict(self) -> dict[str, float]:
        mean, std = self._check()
        return {"mean": mean, "std": std}

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "TargetScaler":
        return cls(mean_=float(payload["mean"]), std_=float(payload["std"]))
