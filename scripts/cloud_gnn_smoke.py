"""Deployment smoke test for the P0 GNN web adapter; no scientific data."""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import time
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gnn_web import load_model_bundle, parse_graph_csv, predict_graphs
from tgv_gnn.models.graphsage_gmt import GraphSAGEGMTRegressor


def main() -> None:
    frame = pd.DataFrame([{
        "molecule_id": "deployment_smoke_only",
        "symbols": json.dumps(["Cu", "S"]),
        "positions_angstrom": json.dumps([[0, 0, 0], [0, 0, 1.8]]),
        "cell_angstrom": json.dumps([[8, 0, 0], [0, 8, 0], [0, 0, 12]]),
        "pbc": json.dumps([True, True, False]),
        "atom_roles": json.dumps([1, 0]),
    }])
    graphs, _ = parse_graph_csv(frame.to_csv(index=False).encode("utf-8"))
    with tempfile.TemporaryDirectory(prefix="tgv_gnn_cloud_smoke_") as temp_dir:
        model_dir = Path(temp_dir) / "model"
        GraphSAGEGMTRegressor(hidden_dim=16, n_conv=2, head_hidden_dim=8).save(model_dir)
        payload = BytesIO()
        with ZipFile(payload, "w", ZIP_DEFLATED) as archive:
            for name in ("model_config.json", "model_state.pt", "model_card.json"):
                archive.write(model_dir / name, arcname=name)
        model, card = load_model_bundle(payload.getvalue())
    start = time.perf_counter()
    prediction = predict_graphs(model, graphs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(json.dumps({
        "status": "ok",
        "model_type": card["model_type"],
        "prediction_shape": list(prediction.shape),
        "elapsed_ms": round(elapsed_ms, 3),
        "scientific_use": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
