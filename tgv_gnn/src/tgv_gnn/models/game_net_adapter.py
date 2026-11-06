from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tgv_gnn.data.graph_builder import GAME_NET_ELEMENT_VOCAB


@dataclass(frozen=True)
class GameNetMetadata:
    graph: dict[str, Any]
    train: dict[str, Any]
    architecture: dict[str, Any]
    element_vocab: tuple[str, ...] = GAME_NET_ELEMENT_VOCAB


def read_game_net_metadata(input_txt: str | Path) -> GameNetMetadata:
    """Safely inspect GAME-Net ``input.txt`` without unpickling legacy models."""
    payload = ast.literal_eval(Path(input_txt).read_text(encoding="utf-8"))
    for key in ("graph", "train", "architecture"):
        if key not in payload or not isinstance(payload[key], dict):
            raise ValueError(f"GAME-Net 元数据缺少 {key}。")
    architecture = payload["architecture"]
    if architecture.get("pool_layer") != "GMT":
        raise ValueError("兼容读取器只接受明确标记为 Graph Multiset Transformer 的 GMT 模型。")
    return GameNetMetadata(payload["graph"], payload["train"], architecture)


def legacy_checkpoint_warning() -> str:
    return (
        "官方旧版 model.pth 可能包含完整 Python pickle 对象；P0 不自动加载不受信任的 pickle。"
        "请在可信隔离环境中提取 state_dict，并连同 element_vocab 与 schema metadata 导入。"
    )
