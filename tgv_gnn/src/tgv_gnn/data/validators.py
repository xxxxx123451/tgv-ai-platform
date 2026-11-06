from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


CANDIDATE_COLUMNS = (
    "molecule_id", "canonical_smiles", "additive_role", "formal_charge",
    "protonation_state", "counterion_policy", "chemical_family", "source",
)
ADSORPTION_COLUMNS = (
    "molecule_id", "config_id", "facet", "E_slab_adsorbate", "E_clean_slab",
    "E_isolated_molecule", "E_ads", "energy_unit", "calculation_version",
)
REDOX_COLUMNS = (
    "molecule_id", "canonical_smiles", "charge", "protonation_state",
    "reference_electrode_raw", "solvent", "source",
)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} 必须是 pandas.DataFrame。")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{label} 缺少列：" + ", ".join(missing))
    if frame.empty:
        raise ValueError(f"{label} 不能为空。")
    return frame.copy()


def validate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    clean = _require_columns(frame, CANDIDATE_COLUMNS, "候选分子表")
    if clean["molecule_id"].isna().any() or clean["molecule_id"].astype(str).str.strip().eq("").any():
        raise ValueError("molecule_id 不能为空。")
    if clean["molecule_id"].duplicated().any():
        raise ValueError("molecule_id 必须唯一。")
    allowed = {"accelerator", "suppressor", "leveler", "unknown"}
    invalid_roles = sorted(set(clean["additive_role"].astype(str)) - allowed)
    if invalid_roles:
        raise ValueError("未知 additive_role：" + ", ".join(invalid_roles))
    charge = pd.to_numeric(clean["formal_charge"], errors="coerce")
    if charge.isna().any() or not np.allclose(charge, np.round(charge)):
        raise ValueError("formal_charge 必须为整数。")
    clean["formal_charge"] = charge.astype(int)
    return clean


def validate_adsorption_labels(frame: pd.DataFrame) -> pd.DataFrame:
    clean = _require_columns(frame, ADSORPTION_COLUMNS, "吸附标签表")
    if not clean["energy_unit"].astype(str).eq("eV").all():
        raise ValueError("吸附能单位必须统一为 eV。")
    numeric = ["E_slab_adsorbate", "E_clean_slab", "E_isolated_molecule", "E_ads"]
    for column in numeric:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    if clean[numeric].isna().any().any() or not np.isfinite(clean[numeric].to_numpy()).all():
        raise ValueError("吸附能列必须为有限数值。")
    keys = ["molecule_id", "config_id", "facet", "calculation_version"]
    if clean[keys].duplicated().any():
        raise ValueError("吸附标签键重复。")
    if not clean["facet"].isin(["Cu111", "Cu100"]).all():
        raise ValueError("P0 只接受 Cu111 或 Cu100 晶面标识。")
    return clean


def validate_redox_labels(frame: pd.DataFrame) -> pd.DataFrame:
    clean = _require_columns(frame, REDOX_COLUMNS, "氧化还原标签表")
    if clean["reference_electrode_raw"].isna().any() or clean["reference_electrode_raw"].astype(str).str.strip().eq("").any():
        raise ValueError("reference_electrode_raw 缺失；不得默认假定为 SHE。")
    if "E_ox_vs_SHE" not in clean and "E_red_vs_SHE" not in clean:
        raise ValueError("至少需要 E_ox_vs_SHE 或 E_red_vs_SHE。")
    for column in ("E_ox_vs_SHE", "E_red_vs_SHE"):
        if column in clean:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
            if clean[column].isna().any():
                raise ValueError(f"{column} 必须为有限数值。")
    return clean
