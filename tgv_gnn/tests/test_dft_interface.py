from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tgv_gnn.dft_interface.poscar import StructureRecord, export_poscar, read_poscar
from tgv_gnn.dft_interface.results import append_dft_labels


def test_poscar_roundtrip_preserves_cell_species_positions_and_constraints(tmp_path):
    structure = StructureRecord(
        symbols=["Cu", "Cu", "S"],
        positions=np.array([[0, 0, 0], [1.8, 0, 0], [0.9, 0.9, 2.0]]),
        cell=np.diag([5.0, 5.0, 12.0]),
        fixed_mask=np.array([True, True, False]),
        molecule_id="m1", config_id="c2", facet="Cu111",
    )
    path = export_poscar(structure, tmp_path)
    loaded = read_poscar(path)
    assert loaded.symbols == ["Cu", "Cu", "S"]
    assert np.allclose(loaded.cell, structure.cell)
    assert np.allclose(loaded.positions, structure.positions)
    assert loaded.fixed_mask.tolist() == [True, True, False]
    assert path.with_suffix(".json").exists()


def test_append_dft_labels_validates_keys_units_and_versions():
    dataset = pd.DataFrame({
        "molecule_id": ["m1"], "config_id": ["c1"], "facet": ["Cu111"],
        "y_ads": [np.nan], "calculation_version": ["v1"],
    })
    results = pd.DataFrame({
        "molecule_id": ["m1"], "config_id": ["c1"], "facet": ["Cu111"],
        "dft_E_ads": [-0.75], "energy_unit": ["eV"], "calculation_version": ["v1"],
    })
    updated = append_dft_labels(results, dataset)
    assert updated.loc[0, "y_ads"] == -0.75
    results.loc[0, "energy_unit"] = "kJ/mol"
    with pytest.raises(ValueError, match="eV"):
        append_dft_labels(results, dataset)
