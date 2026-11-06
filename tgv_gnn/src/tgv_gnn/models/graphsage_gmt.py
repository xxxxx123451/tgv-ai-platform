from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GraphMultisetTransformer, SAGEConv, global_mean_pool

from tgv_gnn.data.graph_builder import GAME_NET_ELEMENT_VOCAB
from tgv_gnn.models.base import BaseGNNModel


def _schema_hash(vocab: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(list(vocab), separators=(",", ":")).encode()).hexdigest()


class GraphSAGEGMTRegressor(nn.Module, BaseGNNModel):
    """GAME-Net-inspired topology baseline.

    GMT means Graph Multiset Transformer. ``global_mean`` is exposed as a
    distinct ablation and never aliases GMT.
    """

    def __init__(
        self,
        in_features: int = 19,
        hidden_dim: int = 160,
        n_conv: int = 3,
        head_hidden_dim: int = 64,
        pooling: str = "gmt",
        dropout: float = 0.0,
        gmt_k: int = 8,
        element_vocab: list[str] | tuple[str, ...] = GAME_NET_ELEMENT_VOCAB,
    ) -> None:
        super().__init__()
        if in_features != len(element_vocab):
            raise ValueError("in_features 必须与 element_vocab 长度一致。")
        if pooling not in {"gmt", "global_mean"}:
            raise ValueError("pooling 只能是 'gmt' 或 'global_mean'。")
        if n_conv < 1:
            raise ValueError("n_conv 至少为 1。")
        self.config = {
            "in_features": int(in_features), "hidden_dim": int(hidden_dim),
            "n_conv": int(n_conv), "head_hidden_dim": int(head_hidden_dim),
            "pooling": pooling, "dropout": float(dropout), "gmt_k": int(gmt_k),
            "element_vocab": list(element_vocab),
        }
        self.input_layer = nn.Linear(in_features, hidden_dim, bias=False)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim, aggr="mean", bias=False) for _ in range(n_conv)])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.pooling_name = pooling
        self.gmt = GraphMultisetTransformer(hidden_dim, k=gmt_k, heads=1) if pooling == "gmt" else None
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, 1),
        )
        self.training_history: list[dict[str, float]] = []
        self.target_scaler = None

    def forward(self, data) -> torch.Tensor:
        x = self.activation(self.input_layer(data.x.float()))
        for conv in self.convs:
            x = self.dropout(self.activation(conv(x, data.edge_index)))
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        pooled = self.gmt(x, index=batch) if self.gmt is not None else global_mean_pool(x, batch)
        return self.head(pooled).view(-1)

    def fit(self, train_loader: Any, val_loader: Any, config: dict[str, Any]):
        from tgv_gnn.training.trainer import train_regressor

        self.training_history = train_regressor(self, train_loader, val_loader, config)
        return self

    def predict(self, loader: Any) -> np.ndarray:
        device = next(self.parameters()).device
        self.eval()
        values: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                values.append(self(batch.to(device)).cpu().numpy())
        prediction = np.concatenate(values) if values else np.array([], dtype=float)
        if self.target_scaler is not None:
            prediction = self.target_scaler.inverse_transform(prediction)
        return prediction

    def predict_with_uncertainty(self, loader: Any) -> dict[str, np.ndarray]:
        prediction = self.predict(loader)
        return {
            "prediction_mean": prediction,
            "prediction_std": np.full_like(prediction, np.nan, dtype=float),
            "uncertainty_available": np.zeros(len(prediction), dtype=bool),
        }

    def get_model_card(self) -> dict[str, Any]:
        vocab = self.config["element_vocab"]
        return {
            "model_type": "GraphSAGEGMTRegressor",
            "version": "0.1.0",
            "task": "single-configuration adsorption-property regression",
            "element_vocab": vocab,
            "schema_hash": _schema_hash(vocab),
            "graph_schema": {
                "builder": "pbc_radius",
                "cutoff_angstrom": 3.5,
                "pbc": [True, True, False],
                "edge_direction": "bidirectional",
            },
            "pooling": self.pooling_name,
            "uncertainty": "single model; unavailable unless wrapped by GNNEnsemble",
            "target_scaling": self.target_scaler.to_dict() if self.target_scaler is not None else None,
            "limitations": [
                "Static graph predictions do not represent adsorption kinetics or additive synergy.",
                "Predicted low-energy configurations require DFT verification.",
                "Unknown elements are rejected rather than mapped to zero vectors.",
            ],
        }

    def save(self, output_dir: str | Path, **kwargs: Any) -> Path:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output / "model_state.pt")
        (output / "model_config.json").write_text(
            json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        card = self.get_model_card()
        card.update(kwargs.get("extra_model_card", {}))
        (output / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.training_history:
            (output / "training_history.json").write_text(
                json.dumps(self.training_history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        if self.target_scaler is not None:
            (output / "target_scaler.json").write_text(
                json.dumps(self.target_scaler.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return output

    @classmethod
    def load(cls, model_dir: str | Path, map_location: str = "cpu") -> "GraphSAGEGMTRegressor":
        source = Path(model_dir)
        config = json.loads((source / "model_config.json").read_text(encoding="utf-8"))
        vocab = config.get("element_vocab")
        if not vocab or len(vocab) != int(config["in_features"]):
            raise ValueError("模型包 element_vocab 与 in_features 不匹配。")
        card_path = source / "model_card.json"
        if card_path.exists():
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if card.get("schema_hash") != _schema_hash(vocab):
                raise ValueError("模型包 schema hash 校验失败。")
        model = cls(**config)
        state = torch.load(source / "model_state.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        scaler_path = source / "target_scaler.json"
        if scaler_path.exists():
            from tgv_gnn.training.scaling import TargetScaler

            model.target_scaler = TargetScaler.from_dict(json.loads(scaler_path.read_text(encoding="utf-8")))
        model.to(map_location)
        model.eval()
        return model
