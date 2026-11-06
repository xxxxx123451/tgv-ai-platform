from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn


def _labels(loader: Any) -> np.ndarray:
    values = []
    for batch in loader:
        if not hasattr(batch, "y_ads"):
            raise ValueError("训练图缺少 y_ads。")
        values.extend(batch.y_ads.detach().cpu().view(-1).tolist())
    return np.asarray(values, dtype=float)


def train_regressor(model: nn.Module, train_loader: Any, val_loader: Any, config: dict[str, Any]) -> list[dict[str, float]]:
    """Train with checkpointed validation MAE and fold-local target scaling."""
    from tgv_gnn.training.scaling import TargetScaler

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    scaler = TargetScaler().fit(_labels(train_loader))
    model.target_scaler = scaler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=float(config.get("scheduler_factor", 0.7)),
        patience=int(config.get("scheduler_patience", 5)), min_lr=float(config.get("min_learning_rate", 1e-8)),
    )
    loss_fn = nn.L1Loss()
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(config.get("epochs", 200)) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            target = torch.as_tensor(scaler.transform(batch.y_ads.detach().cpu().numpy()), device=device, dtype=torch.float32).view(-1)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip_norm", 5.0)))
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        absolute_errors = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = scaler.inverse_transform(model(batch).cpu().numpy())
                truth = batch.y_ads.cpu().numpy().reshape(-1)
                absolute_errors.extend(np.abs(pred - truth))
        val_mae = float(np.mean(absolute_errors)) if absolute_errors else float("inf")
        scheduler.step(val_mae)
        history.append({
            "epoch": float(epoch), "train_scaled_mae": float(np.mean(train_losses)),
            "validation_mae_eV": val_mae, "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        if val_mae < best_val - 1e-12:
            best_val, best_state, stale = val_mae, deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= int(config.get("early_stopping_patience", 20)):
            break
    model.load_state_dict(best_state)
    model.eval()
    return history
