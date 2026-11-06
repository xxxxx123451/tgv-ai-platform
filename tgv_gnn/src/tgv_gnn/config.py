from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str, config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load one of the versioned YAML configurations without mutating it."""
    root = Path(config_dir) if config_dir is not None else PACKAGE_ROOT / "configs"
    path = root / (name if name.endswith(".yaml") else f"{name}.yaml")
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置根节点必须是 mapping：{path}")
    return payload
