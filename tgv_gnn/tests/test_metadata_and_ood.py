from __future__ import annotations

import numpy as np
import json
from pathlib import Path

from tgv_gnn.inference.applicability import ApplicabilityDomain
from tgv_gnn.models.game_net_adapter import read_game_net_metadata


def test_game_net_metadata_reader_distinguishes_gmt(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text(
        repr({
            "graph": {"voronoi_tol": 0.5, "scaling_factor": 1.5},
            "train": {"loss_function": "mae"},
            "architecture": {"pool_layer": "GMT", "dim": 160, "n_conv": 3},
        }), encoding="utf-8",
    )
    metadata = read_game_net_metadata(path)
    assert metadata.architecture["pool_layer"] == "GMT"
    assert "Zn" in metadata.element_vocab and "Cr" not in metadata.element_vocab


def test_applicability_domain_flags_far_embedding():
    rng = np.random.default_rng(42)
    train = rng.normal(0, 0.2, size=(50, 4))
    domain = ApplicabilityDomain().fit(train)
    flags = domain.predict_in_domain(np.vstack([np.zeros(4), np.full(4, 20.0)]))
    assert flags.tolist() == [True, False]


def test_all_json_schemas_are_valid_json_objects():
    schema_dir = Path(__file__).resolve().parents[1] / "schemas"
    schemas = list(schema_dir.glob("*.schema.json"))
    assert len(schemas) == 4
    for path in schemas:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["type"] == "object"
