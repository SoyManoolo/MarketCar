"""Utilidades compartidas para entrenar, evaluar y persistir modelos."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    """Crea un preprocesador de números y categorías para scikit-learn."""
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def with_log_target(regressor: Any) -> TransformedTargetRegressor:
    """Entrena un regresor sobre log1p(precio) y predice en rupias."""
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def calculate_metrics(y_true, predictions) -> dict[str, float]:
    """Calcula métricas de regresión en la escala original de rupias."""
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(mean_squared_error(y_true, predictions) ** 0.5),
        "r2": float(r2_score(y_true, predictions)),
    }


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict[str, dict[str, float]]:
    """Evalúa un modelo entrenado en entrenamiento y test."""
    return {
        "train": calculate_metrics(y_train, model.predict(X_train)),
        "test": calculate_metrics(y_test, model.predict(X_test)),
    }


def print_metrics(model_name: str, metrics: dict[str, dict[str, float]]) -> None:
    """Muestra las métricas de entrenamiento y test de forma consistente."""
    print(f"\n{model_name}")
    for split, label in (("train", "Entrenamiento"), ("test", "Test")):
        values = metrics[split]
        print(f"\n{label}")
        print(f"MAE:  ₹{values['mae']:,.0f}")
        print(f"RMSE: ₹{values['rmse']:,.0f}")
        print(f"R²:   {values['r2']:.3f}")


def save_artifact(
    path: Path,
    *,
    model_name: str,
    model: Any,
    features: list[str],
    categorical_features: list[str],
    metrics: dict[str, dict[str, float]],
    metadata: dict[str, Any],
) -> None:
    """Guarda un modelo entrenado junto con el contexto necesario para usarlo."""
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "artifact_version": 1,
        "model_name": model_name,
        "model": model,
        "features": list(features),
        "categorical_features": list(categorical_features),
        "metrics": metrics,
        "metadata": metadata,
    }
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> dict[str, Any]:
    """Carga un artefacto generado por :func:`save_artifact`."""
    return joblib.load(path)
