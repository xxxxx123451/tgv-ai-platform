from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseGNNModel(ABC):
    """Uniform interface for P0 GNN models."""

    @abstractmethod
    def fit(self, train_loader: Any, val_loader: Any, config: dict[str, Any]):
        raise NotImplementedError

    @abstractmethod
    def predict(self, loader: Any):
        raise NotImplementedError

    @abstractmethod
    def predict_with_uncertainty(self, loader: Any):
        raise NotImplementedError

    @abstractmethod
    def save(self, output_dir: str | Path, **kwargs: Any) -> Path:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, model_dir: str | Path, map_location: str = "cpu"):
        raise NotImplementedError

    @abstractmethod
    def get_model_card(self) -> dict[str, Any]:
        raise NotImplementedError
