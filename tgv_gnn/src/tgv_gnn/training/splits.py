from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


@dataclass(frozen=True)
class NestedGroupSplit:
    outer_fold: int
    outer_train_idx: np.ndarray
    outer_test_idx: np.ndarray
    inner_splits: tuple[tuple[np.ndarray, np.ndarray], ...]
    splitter_name: str


def _can_stratify(labels: np.ndarray, groups: np.ndarray, n_splits: int) -> bool:
    if len(np.unique(groups)) < n_splits:
        return False
    for label in np.unique(labels):
        if len(np.unique(groups[labels == label])) < n_splits:
            return False
    return True


def _splitter(labels: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    if _can_stratify(labels, groups, n_splits):
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed), "StratifiedGroupKFold"
    return GroupKFold(n_splits=n_splits), "GroupKFold(fallback)"


def make_nested_group_cv(
    groups,
    strata,
    *,
    outer_splits: int = 5,
    inner_splits: int = 4,
    seed: int = 42,
):
    """Yield correct nested-CV folds with molecule-level grouping.

    Inner folds select hyperparameters only. The caller must retrain once on
    the full outer training set before evaluating the outer test fold.
    """
    groups = np.asarray(groups, dtype=str)
    strata = np.asarray(strata, dtype=str)
    if groups.ndim != 1 or strata.shape != groups.shape:
        raise ValueError("groups 与 strata 必须是一维等长数组。")
    if len(np.unique(groups)) < outer_splits:
        raise ValueError("唯一 group 数不足以执行外层交叉验证。")
    outer, outer_name = _splitter(strata, groups, outer_splits, seed)
    all_indices = np.arange(len(groups))
    for fold, (outer_train_local, outer_test_local) in enumerate(
        outer.split(all_indices, strata, groups)
    ):
        outer_train = all_indices[outer_train_local]
        outer_test = all_indices[outer_test_local]
        if set(groups[outer_train]) & set(groups[outer_test]):
            raise AssertionError("外层分组泄漏。")
        if len(np.unique(groups[outer_train])) < inner_splits:
            raise ValueError("外层训练数据的唯一 group 数不足以执行内层交叉验证。")
        inner, inner_name = _splitter(strata[outer_train], groups[outer_train], inner_splits, seed + fold + 1)
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        local = np.arange(len(outer_train))
        for train_local, val_local in inner.split(local, strata[outer_train], groups[outer_train]):
            train_idx = outer_train[train_local]
            val_idx = outer_train[val_local]
            if set(groups[train_idx]) & set(groups[val_idx]):
                raise AssertionError("内层分组泄漏。")
            folds.append((train_idx, val_idx))
        yield NestedGroupSplit(fold, outer_train, outer_test, tuple(folds), f"{outer_name}; inner={inner_name}")
