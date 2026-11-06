from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


REQUIRED_PREDICTIONS = (
    "molecule_id", "additive_role", "facet", "E_ads_pred", "E_ads_std",
    "E_ox_pred_vs_SHE", "E_red_pred_vs_SHE", "ood_score",
)


def _active_window_ok(row: pd.Series, rule: Mapping) -> bool:
    lower, upper = map(float, rule["active_window"])
    return lower <= float(row["E_ox_pred_vs_SHE"]) <= upper or lower <= float(row["E_red_pred_vs_SHE"]) <= upper


def _rule_accepts(row: pd.Series, rule: Mapping) -> tuple[bool, str]:
    lower, upper = map(float, rule["ads_window"])
    if not lower <= float(row["E_ads_pred"]) <= upper:
        return False, "outside_adsorption_window"
    policy = str(rule.get("redox_policy", "none"))
    if policy == "stable_outside":
        ok = float(row["E_ox_pred_vs_SHE"]) >= float(rule["ox_min"]) and float(row["E_red_pred_vs_SHE"]) <= float(rule["red_max"])
        return ok, "accepted" if ok else "outside_stability_window"
    if policy == "active_window":
        ok = _active_window_ok(row, rule)
        return ok, "accepted" if ok else "outside_active_window"
    if policy in {"none", "rank_only"}:
        return True, "accepted"
    raise ValueError(f"未知 redox_policy：{policy}")


def tgv_screen(
    predictions: pd.DataFrame,
    role_rules: Mapping,
    facet: str,
    *,
    use_cda: bool = False,
    uncertainty_limit: float | None = None,
) -> pd.DataFrame:
    """Apply calibrated role/facet rules or return a transparent ranking only."""
    missing = [name for name in REQUIRED_PREDICTIONS if name not in predictions]
    if missing:
        raise ValueError("筛选输入缺少列：" + ", ".join(missing))
    frame = predictions.loc[predictions["facet"].astype(str) == str(facet)].copy()
    if frame.empty:
        raise ValueError(f"没有 facet={facet} 的预测。")
    numeric = ["E_ads_pred", "E_ads_std", "E_ox_pred_vs_SHE", "E_red_pred_vs_SHE", "ood_score"]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("筛选预测必须为有限数值。")
    if use_cda and "cda_probability" not in frame:
        raise ValueError("use_cda=True 时需要 cda_probability。")

    calibrated = all(
        role in role_rules and facet in role_rules[role]
        for role in frame["additive_role"].astype(str).unique()
    )
    # 排名惩罚高不确定度/OOD；未标定火山窗口时不臆造最佳吸附能方向。
    frame["priority_score"] = -(frame["E_ads_std"].astype(float) + frame["ood_score"].astype(float))
    frame["hard_filter_applied"] = bool(calibrated)
    if not calibrated:
        frame["accepted"] = True
        frame["screen_status"] = "rank_only_uncalibrated"
        return frame.sort_values("priority_score", ascending=False, ignore_index=True)

    accepted: list[bool] = []
    statuses: list[str] = []
    for _, row in frame.iterrows():
        rule = role_rules[str(row["additive_role"])][facet]
        ok, status = _rule_accepts(row, rule)
        if uncertainty_limit is not None and float(row["E_ads_std"]) > uncertainty_limit:
            ok, status = False, "high_uncertainty_requires_validation"
        if bool(rule.get("reject_ood", False)) and float(row["ood_score"]) > float(rule.get("ood_limit", 1.0)):
            ok, status = False, "out_of_domain"
        if use_cda and rule.get("cda_threshold") is not None:
            if float(row["cda_probability"]) < float(rule["cda_threshold"]):
                ok, status = False, "below_cda_threshold"
        accepted.append(bool(ok))
        statuses.append(status)
    frame["accepted"] = accepted
    frame["screen_status"] = statuses
    return frame.sort_values(["accepted", "priority_score"], ascending=[False, False], ignore_index=True)
