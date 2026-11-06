from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tgv_gnn.data.graph_builder import GAME_NET_ELEMENT_VOCAB, build_radius_graph
from tgv_gnn.data.surfaces import build_cu_slab
from tgv_gnn.data.validators import validate_adsorption_labels, validate_candidates
from tgv_gnn.config import load_config


def test_official_game_net_vocab_is_stable_and_has_no_al_or_cr():
    assert len(GAME_NET_ELEMENT_VOCAB) == 19
    assert set(GAME_NET_ELEMENT_VOCAB) == {
        "Ag", "Au", "Cd", "Co", "Cu", "Fe", "Ir", "Ni", "Os",
        "Pd", "Pt", "Rh", "Ru", "Zn", "C", "H", "O", "N", "S",
    }
    assert "Al" not in GAME_NET_ELEMENT_VOCAB
    assert "Cr" not in GAME_NET_ELEMENT_VOCAB


def test_unknown_elements_are_rejected_in_compatibility_mode():
    positions = np.array([[0, 0, 0], [0, 0, 1.8]], dtype=float)
    with pytest.raises(ValueError, match="未知元素"):
        build_radius_graph(["Cu", "Cl"], positions, np.eye(3) * 10, cutoff=3.0)


def test_graph_contains_bidirectional_edges_positions_cell_and_roles():
    positions = np.array([[0, 0, 0], [0, 0, 1.8], [7, 7, 7]], dtype=float)
    graph = build_radius_graph(
        ["Cu", "S", "Cu"], positions, np.eye(3) * 12, cutoff=2.5,
        atom_roles=[1, 0, 1], molecule_id="sps", config_id="c1", facet="Cu111",
    )
    edges = {tuple(pair) for pair in graph.edge_index.t().tolist()}
    assert (0, 1) in edges and (1, 0) in edges
    assert graph.pos.shape == (3, 3)
    assert graph.cell.shape[-2:] == (3, 3)
    assert graph.atom_role.tolist() == [1, 0, 1]
    assert graph.molecule_id == "sps"


def test_candidate_and_adsorption_contracts_reject_bad_records():
    candidates = pd.DataFrame({
        "molecule_id": ["m1"], "canonical_smiles": ["CCO"],
        "additive_role": ["accelerator"], "formal_charge": [0],
        "protonation_state": ["neutral"], "counterion_policy": ["none"],
        "chemical_family": ["aliphatic"], "source": ["test"],
    })
    assert validate_candidates(candidates).iloc[0]["molecule_id"] == "m1"
    labels = pd.DataFrame({
        "molecule_id": ["m1"], "config_id": ["c1"], "facet": ["Cu111"],
        "E_slab_adsorbate": [-20.0], "E_clean_slab": [-15.0],
        "E_isolated_molecule": [-4.0], "E_ads": [-1.0],
        "energy_unit": ["eV"], "calculation_version": ["v1"],
    })
    assert validate_adsorption_labels(labels).iloc[0]["E_ads"] == -1.0
    labels.loc[0, "energy_unit"] = "kJ/mol"
    with pytest.raises(ValueError, match="eV"):
        validate_adsorption_labels(labels)


def test_versioned_configs_load_and_cu111_slab_has_fixed_bottom_layers():
    assert load_config("data")["primary_facet"] == "Cu111"
    slab = build_cu_slab("Cu111", size=(2, 2, 3), vacuum=8.0, fixed_layers=1)
    assert len(slab.symbols) == 12
    assert slab.fixed_mask.sum() == 4
    assert np.linalg.det(slab.cell) > 0
