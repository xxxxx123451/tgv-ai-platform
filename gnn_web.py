"""Small, safe web adapters around the standalone P0 GNN package."""
from __future__ import annotations

from io import BytesIO, StringIO
import json
from pathlib import Path
import sys
import tempfile
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd


PACKAGE_SRC = Path(__file__).parent / "tgv_gnn" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

MODEL_FILES = {"model_config.json", "model_state.pt", "model_card.json", "target_scaler.json"}
MAX_MODEL_BUNDLE_BYTES = 100 * 1024 * 1024
MAX_GRAPH_ROWS = 50
MAX_ATOMS_PER_GRAPH = 500
MAX_TOTAL_ATOMS = 10_000
GRAPH_COLUMNS = ("molecule_id", "symbols", "positions_angstrom", "cell_angstrom")
GAME_NET_ELEMENT_VOCAB = [
    "Ag", "Au", "C", "Cd", "Co", "Cu", "Fe", "H", "Ir", "N",
    "Ni", "O", "Os", "Pd", "Pt", "Rh", "Ru", "S", "Zn",
]


def load_model_bundle(uploaded_bytes: bytes):
    """Load the P0 state-dict bundle; never unpickle arbitrary user objects."""
    if not uploaded_bytes or len(uploaded_bytes) > MAX_MODEL_BUNDLE_BYTES:
        raise ValueError("模型包为空或超过 100 MB 限制。")
    try:
        archive = ZipFile(BytesIO(uploaded_bytes))
    except BadZipFile as exc:
        raise ValueError("模型包不是有效 ZIP 文件。") from exc

    with archive:
        files = {}
        total_uncompressed = 0
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = Path(info.filename.replace("\\", "/"))
            if path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
                raise ValueError("模型 ZIP 只允许根目录中的模型文件，拒绝路径穿越条目。")
            if path.name not in MODEL_FILES:
                raise ValueError(f"模型 ZIP 含有不支持的文件：{path.name}")
            if path.name in files:
                raise ValueError(f"模型 ZIP 文件重复：{path.name}")
            total_uncompressed += info.file_size
            if info.file_size > MAX_MODEL_BUNDLE_BYTES or total_uncompressed > MAX_MODEL_BUNDLE_BYTES:
                raise ValueError("模型包解压后超过 100 MB 限制。")
            if info.compress_size and info.file_size / info.compress_size > 200:
                raise ValueError("模型包压缩率异常，已拒绝解压。")
            files[path.name] = info

        required = {"model_config.json", "model_state.pt", "model_card.json"}
        if not required.issubset(files):
            raise ValueError("模型 ZIP 必须包含 model_config.json、model_state.pt、model_card.json。")

        try:
            config = json.loads(archive.read(files["model_config.json"]).decode("utf-8"))
            card = json.loads(archive.read(files["model_card.json"]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("模型 config/card 必须是 UTF-8 JSON。") from exc
        if not isinstance(config, dict) or not isinstance(card, dict):
            raise ValueError("模型 config/card JSON 根节点必须是对象。")
        if card.get("model_type") != "GraphSAGEGMTRegressor":
            raise ValueError("只允许 GraphSAGEGMTRegressor P0 模型包。")
        limits = {"hidden_dim": (1, 512), "n_conv": (1, 8), "head_hidden_dim": (1, 512), "gmt_k": (1, 64)}
        if int(config.get("in_features", 0)) != 19 or config.get("pooling") not in {"gmt", "global_mean"}:
            raise ValueError("模型 config 的输入维数或 pooling 不符合 P0 schema。")
        if config.get("element_vocab") != GAME_NET_ELEMENT_VOCAB:
            raise ValueError("模型 element_vocab 的元素或顺序与 GAME-Net P0 schema 不一致。")
        dropout = config.get("dropout")
        if not isinstance(dropout, (int, float)) or not 0 <= float(dropout) < 1:
            raise ValueError("模型 dropout 必须在 [0, 1) 内。")
        for name, (lower, upper) in limits.items():
            value = config.get(name)
            if not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"模型 config 的 {name} 超出云端安全范围 {lower}–{upper}。")

        with tempfile.TemporaryDirectory(prefix="tgv_gnn_model_") as temp_dir:
            root = Path(temp_dir)
            for name, info in files.items():
                (root / name).write_bytes(archive.read(info))
            try:
                from tgv_gnn.models.graphsage_gmt import GraphSAGEGMTRegressor

                model = GraphSAGEGMTRegressor.load(root, map_location="cpu")
            except Exception as exc:
                raise ValueError(f"模型结构、权重或 schema 校验失败：{exc}") from exc
    return model, model.get_model_card()


def _json_cell(value, column: str):
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{column} 必须是合法 JSON 数组。") from exc


def parse_graph_csv(
    csv_bytes: bytes,
    *,
    cutoff_angstrom: float = 3.5,
    expected_pbc: list[bool] | tuple[bool, bool, bool] = (True, True, False),
):
    """Validate CSV graph records and build the model's PBC radius graphs."""
    try:
        frame = pd.read_csv(StringIO(csv_bytes.decode("utf-8-sig")))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError("结构 CSV 必须是 UTF-8 编码且格式正确。") from exc
    missing = [column for column in GRAPH_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("结构 CSV 缺少列：" + ", ".join(missing))
    if frame.empty or len(frame) > MAX_GRAPH_ROWS:
        raise ValueError(f"结构记录数必须为 1–{MAX_GRAPH_ROWS} 条。")

    from tgv_gnn.data.graph_builder import build_radius_graph

    graphs = []
    metadata = []
    total_atoms = 0
    for row_number, row in frame.iterrows():
        symbols = _json_cell(row["symbols"], "symbols")
        positions = np.asarray(_json_cell(row["positions_angstrom"], "positions_angstrom"), dtype=float)
        cell = np.asarray(_json_cell(row["cell_angstrom"], "cell_angstrom"), dtype=float)
        pbc = _json_cell(row["pbc"], "pbc") if "pbc" in frame and pd.notna(row["pbc"]) else [True, True, False]
        roles = _json_cell(row["atom_roles"], "atom_roles") if "atom_roles" in frame and pd.notna(row["atom_roles"]) else None
        if not isinstance(symbols, list) or not symbols or len(symbols) > MAX_ATOMS_PER_GRAPH:
            raise ValueError(f"第 {row_number + 2} 行 symbols 必须为 1–{MAX_ATOMS_PER_GRAPH} 项 JSON 数组。")
        total_atoms += len(symbols)
        if total_atoms > MAX_TOTAL_ATOMS:
            raise ValueError(f"全部结构原子总数不得超过 {MAX_TOTAL_ATOMS}。")
        if positions.shape != (len(symbols), 3) or not np.isfinite(positions).all():
            raise ValueError(f"第 {row_number + 2} 行 positions_angstrom 必须是有限的 N×3 坐标（单位 Å）。")
        if cell.shape != (3, 3) or not np.isfinite(cell).all() or abs(np.linalg.det(cell)) < 1e-10:
            raise ValueError(f"第 {row_number + 2} 行 cell_angstrom 必须是可逆的 3×3 晶胞（单位 Å）。")
        if not isinstance(pbc, list) or len(pbc) != 3 or not all(isinstance(v, bool) for v in pbc):
            raise ValueError(f"第 {row_number + 2} 行 pbc 必须为三个布尔值的 JSON 数组。")
        if list(pbc) != list(expected_pbc):
            raise ValueError(f"第 {row_number + 2} 行 pbc 与模型卡要求 {list(expected_pbc)} 不一致。")
        if roles is not None and (not isinstance(roles, list) or len(roles) != len(symbols)):
            raise ValueError(f"第 {row_number + 2} 行 atom_roles 长度必须与 symbols 一致。")
        molecule_id = str(row["molecule_id"]).strip()
        if not molecule_id or molecule_id.lower() == "nan":
            raise ValueError(f"第 {row_number + 2} 行 molecule_id 不能为空。")
        graph = build_radius_graph(
            symbols, positions, cell, cutoff=cutoff_angstrom, pbc=pbc,
            atom_roles=roles, molecule_id=molecule_id,
            parent_molecule_id=str(row.get("parent_molecule_id", molecule_id)),
            config_id=str(row.get("config_id", f"config_{row_number:04d}")),
            facet=str(row.get("facet", "Cu111")),
            family=str(row.get("family", "unknown")),
        )
        graphs.append(graph)
        metadata.append({
            "molecule_id": molecule_id,
            "config_id": graph.config_id,
            "facet": graph.facet,
            "family": graph.family,
            "n_atoms": len(symbols),
            "n_edges": int(graph.edge_index.shape[1]),
        })
    return graphs, pd.DataFrame(metadata)


def predict_graphs(model, graphs, *, batch_size: int = 16) -> np.ndarray:
    from torch_geometric.loader import DataLoader

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    return model.predict(loader)


def validate_candidate_csv(csv_bytes: bytes) -> pd.DataFrame:
    """Validate the P0 candidate data contract from a UTF-8 CSV upload."""
    try:
        frame = pd.read_csv(StringIO(csv_bytes.decode("utf-8-sig")))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError("候选 CSV 必须是 UTF-8 编码且格式正确。") from exc
    from tgv_gnn.data.validators import validate_candidates

    return validate_candidates(frame)


def validate_adsorption_csv(csv_bytes: bytes) -> pd.DataFrame:
    """Validate Cu adsorption labels without training or altering the upload."""
    try:
        frame = pd.read_csv(StringIO(csv_bytes.decode("utf-8-sig")))
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError("吸附标签 CSV 必须是 UTF-8 编码且格式正确。") from exc
    from tgv_gnn.data.validators import validate_adsorption_labels

    return validate_adsorption_labels(frame)


def export_graphs_as_poscar_zip(graphs) -> bytes:
    """Export validated graph coordinates as POSCAR files for external DFT."""
    from tgv_gnn.dft_interface.poscar import StructureRecord, export_poscar

    atomic_symbols = {
        1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 26: "Fe", 27: "Co",
        28: "Ni", 29: "Cu", 30: "Zn", 44: "Ru", 45: "Rh", 46: "Pd",
        47: "Ag", 48: "Cd", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au",
    }
    payload = BytesIO()
    with tempfile.TemporaryDirectory(prefix="tgv_gnn_poscar_") as temp_dir:
        output = Path(temp_dir)
        for graph in graphs:
            try:
                symbols = [atomic_symbols[int(number)] for number in graph.z.tolist()]
            except KeyError as exc:
                raise ValueError(f"无法导出未知原子序数：{exc.args[0]}") from exc
            structure = StructureRecord(
                symbols=symbols,
                positions=graph.pos.detach().cpu().numpy(),
                cell=graph.cell.detach().cpu().numpy().reshape(3, 3),
                fixed_mask=(graph.atom_role.detach().cpu().numpy() == 1),
                molecule_id=str(graph.molecule_id),
                config_id=str(graph.config_id),
                facet=str(graph.facet),
            )
            export_poscar(structure, output)
        with ZipFile(payload, "w", ZIP_DEFLATED) as archive:
            for path in sorted(output.iterdir()):
                archive.write(path, arcname=path.name)
    return payload.getvalue()
