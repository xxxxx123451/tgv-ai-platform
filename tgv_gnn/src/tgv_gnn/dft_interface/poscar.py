from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class StructureRecord:
    symbols: list[str]
    positions: np.ndarray
    cell: np.ndarray
    fixed_mask: np.ndarray
    molecule_id: str
    config_id: str
    facet: str

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        self.cell = np.asarray(self.cell, dtype=float)
        self.fixed_mask = np.asarray(self.fixed_mask, dtype=bool)
        n = len(self.symbols)
        if self.positions.shape != (n, 3) or self.cell.shape != (3, 3) or self.fixed_mask.shape != (n,):
            raise ValueError("StructureRecord 的 symbols、positions、cell、fixed_mask 尺寸不一致。")
        if abs(np.linalg.det(self.cell)) < 1e-10:
            raise ValueError("晶胞不可逆。")


def _group_order(symbols: list[str]) -> tuple[list[str], list[int], list[int]]:
    species: list[str] = []
    for symbol in symbols:
        if symbol not in species:
            species.append(symbol)
    indices = [i for symbol in species for i, item in enumerate(symbols) if item == symbol]
    counts = [symbols.count(symbol) for symbol in species]
    return species, counts, indices


def export_poscar(structure: StructureRecord, output_dir: str | Path) -> Path:
    """Export a DFT-ready POSCAR and a metadata sidecar JSON."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safe_id = f"{structure.molecule_id}_{structure.config_id}_{structure.facet}".replace("/", "_")
    path = output / f"{safe_id}.POSCAR"
    species, counts, order = _group_order(structure.symbols)
    lines = [
        f"{structure.molecule_id} {structure.config_id} {structure.facet}",
        "1.0",
        *("  " + "  ".join(f"{value:.12f}" for value in vector) for vector in structure.cell),
        "  " + "  ".join(species),
        "  " + "  ".join(map(str, counts)),
        "Selective dynamics",
        "Cartesian",
    ]
    for index in order:
        flags = "F F F" if structure.fixed_mask[index] else "T T T"
        lines.append("  " + "  ".join(f"{value:.12f}" for value in structure.positions[index]) + f"  {flags}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sidecar = {
        "molecule_id": structure.molecule_id,
        "config_id": structure.config_id,
        "facet": structure.facet,
        "format": "VASP POSCAR",
        "coordinate_mode": "Cartesian",
        "original_symbol_order": structure.symbols,
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_poscar(path: str | Path) -> StructureRecord:
    """Read POSCAR files written by :func:`export_poscar` for round-trip QA."""
    poscar = Path(path)
    lines = [line.strip() for line in poscar.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 9:
        raise ValueError("POSCAR 内容不完整。")
    title = lines[0].split()
    scale = float(lines[1])
    cell = np.array([[float(v) for v in lines[i].split()] for i in range(2, 5)]) * scale
    species = lines[5].split()
    counts = [int(value) for value in lines[6].split()]
    if len(species) != len(counts):
        raise ValueError("POSCAR 元素与数量不一致。")
    selective = lines[7].lower().startswith("selective")
    mode_index = 8 if selective else 7
    cartesian = lines[mode_index].lower().startswith(("c", "k"))
    n_atoms = sum(counts)
    atom_lines = lines[mode_index + 1: mode_index + 1 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError("POSCAR 原子坐标数量不足。")
    coords = np.array([[float(v) for v in line.split()[:3]] for line in atom_lines])
    positions = coords * scale if cartesian else coords @ cell
    fixed = np.array([all(flag.upper() == "F" for flag in line.split()[3:6]) for line in atom_lines], dtype=bool)
    symbols = [symbol for symbol, count in zip(species, counts) for _ in range(count)]
    sidecar_path = poscar.with_suffix(".json")
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
    return StructureRecord(
        symbols=symbols,
        positions=positions,
        cell=cell,
        fixed_mask=fixed,
        molecule_id=str(metadata.get("molecule_id", title[0] if title else "unknown")),
        config_id=str(metadata.get("config_id", title[1] if len(title) > 1 else "unknown")),
        facet=str(metadata.get("facet", title[2] if len(title) > 2 else "unknown")),
    )
