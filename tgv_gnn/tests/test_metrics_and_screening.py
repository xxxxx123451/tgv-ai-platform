from __future__ import annotations

import numpy as np
import pandas as pd

from tgv_gnn.screening.role_rules import tgv_screen
from tgv_gnn.training.metrics import regression_metrics


def test_regression_metrics_and_family_breakdown_are_correct():
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.0, 2.0, 2.0])
    report = regression_metrics(y_true, y_pred, groups=["a", "a", "b", "b"])
    assert np.isclose(report["mae"], 0.25)
    assert np.isclose(report["rmse"], 0.5)
    assert report["by_group"]["a"]["mae"] == 0.0
    assert report["by_group"]["b"]["mae"] == 0.5


def test_uncalibrated_screening_only_ranks_and_does_not_hard_filter():
    frame = pd.DataFrame({
        "molecule_id": ["a", "b"], "additive_role": ["accelerator", "leveler"],
        "facet": ["Cu111", "Cu111"], "E_ads_pred": [-1.0, -0.5],
        "E_ads_std": [0.1, 0.2], "E_ox_pred_vs_SHE": [0.4, 1.2],
        "E_red_pred_vs_SHE": [-0.2, -0.8], "ood_score": [0.1, 0.2],
    })
    result = tgv_screen(frame, role_rules={}, facet="Cu111")
    assert result["hard_filter_applied"].eq(False).all()
    assert result["screen_status"].eq("rank_only_un calibrated".replace(" ", "")).all()
    assert len(result) == 2


def test_role_specific_rules_do_not_apply_leveler_policy_to_accelerator():
    frame = pd.DataFrame({
        "molecule_id": ["acc", "lev"], "additive_role": ["accelerator", "leveler"],
        "facet": ["Cu111", "Cu111"], "E_ads_pred": [-0.8, -0.8],
        "E_ads_std": [0.05, 0.05], "E_ox_pred_vs_SHE": [0.5, 0.5],
        "E_red_pred_vs_SHE": [-0.2, -0.2], "ood_score": [0.1, 0.1],
    })
    rules = {
        "accelerator": {"Cu111": {"ads_window": [-1.2, -0.3], "redox_policy": "active_window", "active_window": [-0.3, 0.7]}},
        "leveler": {"Cu111": {"ads_window": [-1.2, -0.3], "redox_policy": "stable_outside", "ox_min": 1.0, "red_max": -0.5}},
    }
    result = tgv_screen(frame, rules, facet="Cu111")
    accepted = dict(zip(result.molecule_id, result.accepted))
    assert bool(accepted["acc"]) is True
    assert bool(accepted["lev"]) is False
