from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_model_card(
    *, model: Any, task: str, data_summary: dict[str, Any], metrics: dict[str, Any],
    split_protocol: str, limitations: list[str] | None = None,
) -> dict[str, Any]:
    base = model.get_model_card()
    base.update({
        "task": task,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_summary": data_summary,
        "metrics": metrics,
        "split_protocol": split_protocol,
        "limitations": list(dict.fromkeys(base.get("limitations", []) + (limitations or []))),
    })
    return base
