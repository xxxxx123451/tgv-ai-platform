from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tgv_gnn.dft_interface.poscar import StructureRecord


def build_cu_slab(
    facet: str = "Cu111",
    *,
    size: Sequence[int] = (3, 3, 4),
    vacuum: float = 10.0,
    fixed_layers: int = 2,
    lattice_constant: float = 3.615,
) -> StructureRecord:
    """Create a Cu slab with ASE and mark the lowest layers as fixed."""
    try:
        from ase.build import fcc100, fcc111
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Cu slab 生成需要 ASE。") from exc
    size_tuple = tuple(int(value) for value in size)
    if len(size_tuple) != 3 or any(value < 1 for value in size_tuple):
        raise ValueError("size 必须为三个正整数。")
    if not 0 <= fixed_layers < size_tuple[2]:
        raise ValueError("fixed_layers 必须小于 slab 层数。")
    if facet == "Cu111":
        atoms = fcc111("Cu", size=size_tuple, a=lattice_constant, vacuum=vacuum, orthogonal=False)
    elif facet == "Cu100":
        atoms = fcc100("Cu", size=size_tuple, a=lattice_constant, vacuum=vacuum)
    else:
        raise ValueError("facet 只能是 Cu111 或 Cu100。")
    positions = atoms.get_positions()
    unique_z = np.unique(np.round(positions[:, 2], decimals=6))
    fixed_z = set(unique_z[:fixed_layers])
    fixed = np.array([round(z, 6) in fixed_z for z in positions[:, 2]], dtype=bool)
    return StructureRecord(
        symbols=atoms.get_chemical_symbols(), positions=positions, cell=atoms.cell.array,
        fixed_mask=fixed, molecule_id="clean_cu_slab", config_id="slab_0000", facet=facet,
    )
