from __future__ import annotations

import numpy as np

from tgv_gnn.training.scaling import TargetScaler
from tgv_gnn.training.splits import make_nested_group_cv


def test_target_scaler_uses_only_values_passed_to_fit():
    scaler = TargetScaler().fit(np.array([1.0, 2.0, 3.0]))
    assert scaler.mean_ == 2.0
    assert np.isclose(scaler.std_, np.std([1.0, 2.0, 3.0]))
    transformed = scaler.transform(np.array([2.0]))
    assert np.isclose(transformed[0], 0.0)
    assert np.allclose(scaler.inverse_transform(transformed), [2.0])


def test_nested_cv_never_splits_parent_molecule_and_has_one_test_per_outer_fold():
    groups = np.repeat([f"m{i}" for i in range(12)], 3)
    families = np.repeat(["polyether", "sulfide", "aromatic"] * 4, 3)
    splits = list(make_nested_group_cv(groups, families, outer_splits=3, inner_splits=2, seed=42))
    assert len(splits) == 3
    all_outer_test_groups = []
    for split in splits:
        train_groups = set(groups[split.outer_train_idx])
        test_groups = set(groups[split.outer_test_idx])
        assert train_groups.isdisjoint(test_groups)
        all_outer_test_groups.extend(test_groups)
        for inner_train, inner_val in split.inner_splits:
            assert set(groups[inner_train]).isdisjoint(set(groups[inner_val]))
            assert set(inner_train).issubset(set(split.outer_train_idx))
            assert set(inner_val).issubset(set(split.outer_train_idx))
    assert sorted(all_outer_test_groups) == sorted(set(groups))
