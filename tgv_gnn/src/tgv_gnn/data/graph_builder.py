from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

try:
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover - exercised only in minimal installations
    Data = None  # type: ignore[assignment]


# GAME-Net 官方集合。顺序被作为本项目模型 schema 的组成部分持久化；
# 加载外部预训练权重时仍须优先读取该权重随附的 vocab，禁止猜测索引。
GAME_NET_ELEMENT_VOCAB = (
    "Ag", "Au", "C", "Cd", "Co", "Cu", "Fe", "H", "Ir", "N",
    "Ni", "O", "Os", "Pd", "Pt", "Rh", "Ru", "S", "Zn",
)


def encode_elements(symbols: Sequence[str], vocab: Sequence[str] = GAME_NET_ELEMENT_VOCAB) -> torch.Tensor:
    """One-hot encode elements and reject out-of-domain species explicitly."""
    index = {symbol: i for i, symbol in enumerate(vocab)}
    unknown = sorted({str(symbol) for symbol in symbols if str(symbol) not in index})
    if unknown:
        raise ValueError(
            "GAME-Net 兼容模式遇到未知元素：" + ", ".join(unknown)
            + "。不得映射为全零向量；请使用扩展模型并重新训练。"
        )
    x = torch.zeros((len(symbols), len(vocab)), dtype=torch.float32)
    for row, symbol in enumerate(symbols):
        x[row, index[str(symbol)]] = 1.0
    return x


def _minimum_image(delta: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    """Apply the minimum-image convention for a general non-singular cell."""
    fractional = delta @ np.linalg.inv(cell)
    fractional[..., pbc] -= np.round(fractional[..., pbc])
    return fractional @ cell


def build_radius_graph(
    symbols: Sequence[str],
    positions: np.ndarray,
    cell: np.ndarray,
    *,
    cutoff: float = 3.5,
    pbc: Sequence[bool] = (True, True, False),
    atom_roles: Sequence[int] | None = None,
    molecule_id: str = "unknown",
    parent_molecule_id: str | None = None,
    config_id: str = "config_0000",
    facet: str = "Cu111",
    family: str = "unknown",
    y_ads: float | None = None,
    element_vocab: Sequence[str] = GAME_NET_ELEMENT_VOCAB,
) -> Any:
    """Build a bidirectional PBC-aware PyG radius graph.

    ``atom_roles`` uses 0=adsorbate, 1=surface, 2=counterion. Distances are
    retained as edge features even though the GAME-Net-compatible GraphSAGE
    baseline itself is topology-only.
    """
    if Data is None:
        raise ImportError("build_radius_graph 需要 torch-geometric。请安装 tgv_gnn/requirements.txt。")
    pos = np.asarray(positions, dtype=np.float64)
    lattice = np.asarray(cell, dtype=np.float64)
    periodic = np.asarray(pbc, dtype=bool)
    if pos.shape != (len(symbols), 3):
        raise ValueError(f"positions 应为 ({len(symbols)}, 3)，实际为 {pos.shape}。")
    if lattice.shape != (3, 3) or abs(np.linalg.det(lattice)) < 1e-10:
        raise ValueError("cell 必须为可逆的 3×3 晶胞矩阵。")
    if cutoff <= 0:
        raise ValueError("cutoff 必须大于 0。")
    roles = np.zeros(len(symbols), dtype=np.int64) if atom_roles is None else np.asarray(atom_roles, dtype=np.int64)
    if roles.shape != (len(symbols),) or not np.isin(roles, [0, 1, 2]).all():
        raise ValueError("atom_roles 必须与原子数一致，且只能使用 0/1/2。")

    senders: list[int] = []
    receivers: list[int] = []
    distances: list[float] = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            displacement = _minimum_image(pos[j] - pos[i], lattice, periodic)
            distance = float(np.linalg.norm(displacement))
            if 1e-8 < distance <= cutoff:
                senders.extend((i, j))
                receivers.extend((j, i))
                distances.extend((distance, distance))
    if not senders:
        raise ValueError("指定 cutoff 下没有任何边；请检查坐标、晶胞或截断半径。")

    edge_index = torch.tensor([senders, receivers], dtype=torch.long)
    data = Data(
        x=encode_elements(symbols, element_vocab),
        z=torch.tensor([_atomic_number(s) for s in symbols], dtype=torch.long),
        pos=torch.tensor(pos, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=torch.tensor(distances, dtype=torch.float32).unsqueeze(-1),
        cell=torch.tensor(lattice, dtype=torch.float32).unsqueeze(0),
        pbc=torch.tensor(periodic, dtype=torch.bool).unsqueeze(0),
        atom_role=torch.tensor(roles, dtype=torch.long),
    )
    data.molecule_id = str(molecule_id)
    data.parent_molecule_id = str(parent_molecule_id or molecule_id)
    data.config_id = str(config_id)
    data.facet = str(facet)
    data.family = str(family)
    data.element_vocab = list(element_vocab)
    if y_ads is not None:
        data.y_ads = torch.tensor([float(y_ads)], dtype=torch.float32)
    return data


def _atomic_number(symbol: str) -> int:
    # 只列出兼容 vocab；未知元素会在 encode_elements 中更早被拒绝。
    numbers = {
        "H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "Fe": 26,
        "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ru": 44,
        "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "Os": 76,
        "Ir": 77, "Pt": 78, "Au": 79,
    }
    return numbers[symbol]
