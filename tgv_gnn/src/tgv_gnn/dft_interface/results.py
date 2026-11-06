from __future__ import annotations

import numpy as np
import pandas as pd


RESULT_COLUMNS = (
    "molecule_id", "config_id", "facet", "dft_E_ads", "energy_unit", "calculation_version",
)


def append_dft_labels(dft_results: pd.DataFrame, graph_dataset: pd.DataFrame) -> pd.DataFrame:
    """Return a version-safe copy with matching DFT adsorption labels appended."""
    missing = [column for column in RESULT_COLUMNS if column not in dft_results]
    if missing:
        raise ValueError("DFT 结果缺少列：" + ", ".join(missing))
    if not dft_results["energy_unit"].astype(str).eq("eV").all():
        raise ValueError("DFT 回填能量单位必须为 eV。")
    keys = ["molecule_id", "config_id", "facet"]
    if dft_results[keys].duplicated().any():
        raise ValueError("DFT 结果存在重复键。")
    for column in [*keys, "calculation_version", "y_ads"]:
        if column not in graph_dataset:
            raise ValueError(f"图数据表缺少列：{column}")
    results = dft_results.copy()
    results["dft_E_ads"] = pd.to_numeric(results["dft_E_ads"], errors="coerce")
    if results["dft_E_ads"].isna().any() or not np.isfinite(results["dft_E_ads"]).all():
        raise ValueError("dft_E_ads 必须为有限数值。")
    base = graph_dataset.copy()
    merged = base.merge(
        results[keys + ["dft_E_ads", "calculation_version"]],
        on=keys, how="left", suffixes=("", "_result"), validate="one_to_one",
    )
    if merged["dft_E_ads"].isna().any():
        raise ValueError("部分图样本没有对应的 DFT 结果。")
    mismatch = merged["calculation_version"].astype(str) != merged["calculation_version_result"].astype(str)
    if mismatch.any():
        raise ValueError("DFT calculation_version 与图数据版本不一致。")
    merged["y_ads"] = merged["dft_E_ads"].astype(float)
    return merged.drop(columns=["dft_E_ads", "calculation_version_result"])
