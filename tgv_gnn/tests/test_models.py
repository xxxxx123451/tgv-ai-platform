from __future__ import annotations

import numpy as np
import pytest
import torch

pyg = pytest.importorskip("torch_geometric")
from torch_geometric.data import Batch

from tgv_gnn.data.graph_builder import build_radius_graph
from tgv_gnn.models.graphsage_gmt import GraphSAGEGMTRegressor
from tgv_gnn.training.scaling import TargetScaler


def test_graphsage_gmt_forward_and_safe_bundle_roundtrip(tmp_path):
    g1 = build_radius_graph(["Cu", "S"], np.array([[0, 0, 0], [0, 0, 1.8]]), np.eye(3) * 8, cutoff=2.5)
    g2 = build_radius_graph(["Cu", "O"], np.array([[0, 0, 0], [0, 0, 1.9]]), np.eye(3) * 8, cutoff=2.5)
    batch = Batch.from_data_list([g1, g2])
    model = GraphSAGEGMTRegressor(in_features=19, hidden_dim=16, n_conv=2, head_hidden_dim=8, pooling="gmt")
    pred = model(batch)
    assert pred.shape == (2,)
    model.target_scaler = TargetScaler(mean_=2.0, std_=3.0)
    assert np.allclose(model.predict([batch]), pred.detach().numpy() * 3.0 + 2.0)
    model.save(tmp_path, extra_model_card={"task": "adsorption"})
    restored = GraphSAGEGMTRegressor.load(tmp_path)
    with torch.no_grad():
        assert torch.allclose(model(batch), restored(batch))
    assert restored.target_scaler.to_dict() == {"mean": 2.0, "std": 3.0}
    assert (tmp_path / "model_config.json").exists()
    assert (tmp_path / "model_card.json").exists()
    assert (tmp_path / "target_scaler.json").exists()
