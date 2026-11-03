"""TGV (x-type) electroplating fill-area regression models.

The module keeps data preparation, six estimators, evaluation and Optuna
optimization independent from the Streamlit user interface so it can also be
used from notebooks or scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

FEATURES = ["Depth", "Wide", "i0", "c_ACC", "c_SUP", "c_LEVE"]
TARGET = "Area_ratio"


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce the six features plus target to numeric values."""
    missing = [c for c in FEATURES + [TARGET] if c not in frame.columns]
    if missing:
        raise ValueError("缺少列: " + ", ".join(missing))
    result = frame[FEATURES + [TARGET]].copy()
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).dropna()
    if len(result) < 10:
        raise ValueError("有效数据至少需要 10 行。")
    return result


def _scaled(estimator: Any) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def build_models(random_state: int = 42) -> Dict[str, Any]:
    """Build the six requested regressors with sensible preprocessing."""
    models: Dict[str, Any] = {
        "ANN": _scaled(MLPRegressor(
            hidden_layer_sizes=(100, 50), activation="relu",
            learning_rate_init=0.0005, batch_size=64, max_iter=2000,
            early_stopping=True, random_state=random_state,
        )),
        "RF": RandomForestRegressor(
            n_estimators=300, max_depth=15, random_state=random_state,
            n_jobs=-1,
        ),
        "GPR": _scaled(GaussianProcessRegressor(
            kernel=ConstantKernel(1.0, (1e-4, 10)) * RBF(1.0, (1e-4, 10)),
            normalize_y=True, n_restarts_optimizer=1, random_state=random_state,
        )),
        "SVR": _scaled(SVR(kernel="rbf", gamma="scale")),
        "PR": Pipeline([
            ("poly", PolynomialFeatures(degree=5, include_bias=False)),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBOOST"] = XGBRegressor(
            max_depth=10, n_estimators=300, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
            random_state=random_state, n_jobs=-1,
        )
    except ImportError:
        models["XGBOOST"] = None
    return models


def train_models(
    frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42,
    selected: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split, train and evaluate requested models.

    Returns (fitted_models, metrics, test_features, test_target).
    """
    clean = validate_frame(frame)
    x, y = clean[FEATURES], clean[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state,
    )
    available = build_models(random_state)
    names = list(selected) if selected else list(available)
    fitted, rows = {}, []
    for name in names:
        model = available.get(name)
        if model is None:
            continue
        # scikit-learn requires a sufficiently large validation split when
        # early_stopping=True; small demonstration files should still run.
        if len(x_train) < 30 and hasattr(model, "named_steps") and "model" in model.named_steps:
            estimator = model.named_steps["model"]
            if isinstance(estimator, MLPRegressor):
                estimator.set_params(early_stopping=False)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        fitted[name] = model
        rows.append({
            "Model": name, "R2": r2_score(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "MAE": mean_absolute_error(y_test, pred),
        })
    if not fitted:
        raise RuntimeError("没有可训练的模型；请安装 xgboost 或选择其他模型。")
    return fitted, pd.DataFrame(rows).sort_values("R2", ascending=False), x_test, y_test


def optimize_model(
    frame: pd.DataFrame, model_name: str, n_trials: int = 20,
    test_size: float = 0.2, random_state: int = 42,
) -> Tuple[Mapping[str, Any], float]:
    """Small Optuna study for XGBOOST or ANN; returns best params and RMSE."""
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna 未安装，请执行 pip install optuna。") from exc
    clean = validate_frame(frame)
    x_train, x_valid, y_train, y_valid = train_test_split(
        clean[FEATURES], clean[TARGET], test_size=test_size,
        random_state=random_state,
    )

    def objective(trial: Any) -> float:
        if model_name == "XGBOOST":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError("XGBoost 未安装，请执行 pip install xgboost。") from exc
            model = XGBRegressor(
                max_depth=trial.suggest_int("max_depth", 3, 10),
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                subsample=trial.suggest_float("subsample", 0.7, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
                objective="reg:squarederror", random_state=random_state, n_jobs=-1,
            )
        elif model_name == "ANN":
            model = _scaled(MLPRegressor(
                hidden_layer_sizes=(
                    trial.suggest_int("hidden_1", 32, 256),
                    trial.suggest_int("hidden_2", 16, 128),
                ),
                learning_rate_init=trial.suggest_float("learning_rate_init", 1e-5, 3e-3, log=True),
                alpha=trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
                batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
                max_iter=1500, early_stopping=True, random_state=random_state,
            ))
        else:
            raise ValueError("Optuna 目前支持 ANN 或 XGBOOST。")
        model.fit(x_train, y_train)
        return float(np.sqrt(mean_squared_error(y_valid, model.predict(x_valid))))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, float(study.best_value)


def save_bundle(models: Mapping[str, Any], path: str | Path, metadata: Optional[dict] = None) -> None:
    joblib.dump({"models": dict(models), "features": FEATURES, "target": TARGET,
                 "metadata": metadata or {}}, path)


def load_bundle(path) -> dict:
    """仅加载可信来源的 joblib；反序列化本身可执行代码，并非安全沙箱。"""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("文件不是本平台支持的模型包。")
    if bundle.get("features") != FEATURES or bundle.get("target") != TARGET:
        raise ValueError("模型特征或目标不匹配：需要 Depth、Wide 等六个特征及 Area_ratio；旧版 D1/D2 模型请重新训练。")
    models = bundle.get("models")
    allowed = {"ANN", "XGBOOST", "RF", "GPR", "SVR", "PR"}
    if not isinstance(models, dict) or not models:
        raise ValueError("模型包中没有可用模型。")
    for name, model in models.items():
        if name not in allowed or not callable(getattr(model, "predict", None)):
            raise ValueError(f"无效模型：{name}")
        if getattr(model, "n_features_in_", None) != len(FEATURES):
            raise ValueError(f"{name} 未训练或输入特征数量不匹配。")
        names = getattr(model, "feature_names_in_", None)
        if names is not None and list(names) != FEATURES:
            raise ValueError(f"{name} 的输入特征名称或顺序不匹配。")
    return bundle
