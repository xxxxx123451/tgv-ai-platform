from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gnn_web import (
    export_graphs_as_poscar_zip,
    load_model_bundle,
    parse_graph_csv,
    predict_graphs,
    validate_adsorption_csv,
    validate_candidate_csv,
)


pytest.importorskip("torch_geometric")
from tgv_gnn.models.graphsage_gmt import GraphSAGEGMTRegressor


def _row(symbols=None, pbc=None):
    return {
        "molecule_id": "probe_001",
        "symbols": json.dumps(symbols or ["Cu", "S"]),
        "positions_angstrom": json.dumps([[0, 0, 0], [0, 0, 1.8]]),
        "cell_angstrom": json.dumps([[8, 0, 0], [0, 8, 0], [0, 0, 12]]),
        "pbc": json.dumps(pbc or [True, True, False]),
    }


def _csv_bytes(row):
    import pandas as pd

    return pd.DataFrame([row]).to_csv(index=False).encode("utf-8")


def test_graph_csv_validation_and_cpu_inference():
    graphs, metadata = parse_graph_csv(_csv_bytes(_row()))
    assert len(graphs) == 1
    assert metadata.loc[0, "molecule_id"] == "probe_001"
    model = GraphSAGEGMTRegressor(hidden_dim=16, n_conv=2, head_hidden_dim=8)
    prediction = predict_graphs(model, graphs)
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()
    poscar = export_graphs_as_poscar_zip(graphs)
    assert b"POSCAR" in poscar


def test_uploaded_candidate_and_adsorption_contracts():
    candidate = (
        "molecule_id,canonical_smiles,additive_role,formal_charge,protonation_state,"
        "counterion_policy,chemical_family,source\n"
        "m1,CCO,unknown,0,neutral,none,alcohol,test\n"
    ).encode()
    assert validate_candidate_csv(candidate).loc[0, "molecule_id"] == "m1"
    label = (
        "molecule_id,config_id,facet,E_slab_adsorbate,E_clean_slab,"
        "E_isolated_molecule,E_ads,energy_unit,calculation_version\n"
        "m1,c1,Cu111,-20,-15,-4,-1,eV,v1\n"
    ).encode()
    assert validate_adsorption_csv(label).loc[0, "E_ads"] == -1


def test_graph_csv_rejects_unknown_elements_and_wrong_pbc():
    with pytest.raises(ValueError, match="未知元素"):
        parse_graph_csv(_csv_bytes(_row(symbols=["Cu", "Cl"])))
    with pytest.raises(ValueError, match="pbc 与模型卡要求"):
        parse_graph_csv(_csv_bytes(_row(pbc=[True, True, True])))


def test_safe_model_zip_roundtrip_and_path_traversal_rejection(tmp_path):
    model_dir = tmp_path / "model"
    GraphSAGEGMTRegressor(hidden_dim=16, n_conv=2, head_hidden_dim=8).save(model_dir)
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name in ("model_config.json", "model_state.pt", "model_card.json"):
            archive.write(model_dir / name, arcname=name)
    restored, card = load_model_bundle(buffer.getvalue())
    assert isinstance(restored, GraphSAGEGMTRegressor)
    assert card["graph_schema"]["cutoff_angstrom"] == 3.5

    malicious = BytesIO()
    with ZipFile(malicious, "w", ZIP_DEFLATED) as archive:
        archive.writestr("../model_config.json", "{}")
    with pytest.raises(ValueError, match="路径穿越"):
        load_model_bundle(malicious.getvalue())
