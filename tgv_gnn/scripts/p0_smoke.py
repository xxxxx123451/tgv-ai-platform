from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tgv_gnn.data.graph_builder import build_radius_graph
from tgv_gnn.dft_interface.poscar import export_poscar
from tgv_gnn.data.surfaces import build_cu_slab
from tgv_gnn.models.graphsage_gmt import GraphSAGEGMTRegressor


def main() -> None:
    graph = build_radius_graph(
        ["Cu", "S"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]]),
        np.diag([8.0, 8.0, 12.0]), cutoff=2.5, atom_roles=[1, 0],
        molecule_id="smoke", config_id="config_0000", facet="Cu111",
    )
    model = GraphSAGEGMTRegressor(hidden_dim=16, n_conv=2, head_hidden_dim=8)
    model.eval()
    with torch.no_grad():
        prediction = float(model(graph)[0])
    output = ROOT / "runs" / "p0_smoke"
    model.save(output / "model", extra_model_card={"purpose": "software smoke test only"})
    export_poscar(build_cu_slab("Cu111", size=(2, 2, 3), fixed_layers=1), output / "dft")
    print(f"P0 smoke test completed; untrained prediction={prediction:.6f}; output={output}")


if __name__ == "__main__":
    main()
