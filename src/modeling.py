"""Utilidades compartidas para entrenar, evaluar y persistir modelos."""

import hashlib
import importlib.metadata
import inspect
import json
import platform
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data_cleaning

ARTIFACT_VERSION = 2
HISTORICAL_SUBDIR = "historical"
TRACKED_LIBRARIES = (
    "pandas",
    "numpy",
    "scikit-learn",
    "scipy",
    "joblib",
    "lightgbm",
    "xgboost",
    "catboost",
)
IDENTIFIER_COLUMNS = (
    "usedCarId",
    "usedCarSkuId",
    "used_carid",
    "ucid",
    "vlink",
)


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


def _hash_file(path: Path) -> str | None:
    """Calcula el SHA-256 de un archivo si existe en disco."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_pandas_data(data: pd.DataFrame | pd.Series) -> str:
    """Calcula un hash determinista del contenido, índice y esquema de pandas."""
    digest = hashlib.sha256()
    if isinstance(data, pd.DataFrame):
        schema = [
            (str(col), str(dtype))
            for col, dtype in zip(data.columns, data.dtypes, strict=False)
        ]
        digest.update(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    else:
        digest.update(f"{data.name}:{data.dtype}".encode("utf-8"))

    row_hashes = pd.util.hash_pandas_object(data, index=True).to_numpy(dtype=np.uint64)
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def get_cleaning_version(cleaning_module: Any = data_cleaning) -> dict[str, str]:
    """Devuelve la versión declarada y la huella del código de limpieza."""
    declared_version = str(getattr(cleaning_module, "CLEANING_VERSION", "1.0.0"))
    try:
        source_text = inspect.getsource(cleaning_module)
    except (OSError, TypeError):
        source_text = declared_version
    normalized_source = source_text.replace("\r\n", "\n")
    code_sha256 = hashlib.sha256(normalized_source.encode("utf-8")).hexdigest()
    return {
        "version": declared_version,
        "code_sha256": code_sha256,
    }


def compute_dataset_fingerprint(
    df: pd.DataFrame,
    *,
    target: pd.Series | None = None,
    data_path: Path | None = None,
) -> dict[str, Any]:
    """Construye la huella del dataset bruto y de la tabla preparada."""
    return {
        "file_path": str(data_path) if data_path is not None else None,
        "file_sha256": _hash_file(data_path) if data_path is not None else None,
        "dataframe_sha256": _hash_pandas_data(df),
        "target_sha256": _hash_pandas_data(target) if target is not None else None,
        "rows": int(len(df)),
        "columns": [str(col) for col in df.columns],
    }


def _normalize_indices(index: pd.Index) -> list[int | str]:
    result: list[int | str] = []
    for item in index.tolist():
        if isinstance(item, (int, np.integer)):
            result.append(int(item))
        else:
            result.append(str(item))
    return result


def compute_partition_metadata(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    random_state: int,
    test_size: float = 0.2,
    raw_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Registra índices, semilla, tamaños e identificadores de la partición."""
    train_indices = _normalize_indices(X_train.index)
    test_indices = _normalize_indices(X_test.index)
    payload = json.dumps(
        {"train": train_indices, "test": test_indices},
        separators=(",", ":"),
    ).encode("utf-8")
    partition_sha256 = hashlib.sha256(payload).hexdigest()

    identifier_hashes: dict[str, dict[str, str]] = {}
    if raw_df is not None:
        for col in IDENTIFIER_COLUMNS:
            if col in raw_df.columns:
                train_ids = raw_df.loc[X_train.index, col].astype("string").fillna("<NA>")
                test_ids = raw_df.loc[X_test.index, col].astype("string").fillna("<NA>")
                identifier_hashes[col] = {
                    "train_sha256": _hash_pandas_data(train_ids),
                    "test_sha256": _hash_pandas_data(test_ids),
                }

    return {
        "random_state": int(random_state),
        "test_size": float(test_size),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_indices": train_indices,
        "test_indices": test_indices,
        "partition_sha256": partition_sha256,
        "identifier_hashes": identifier_hashes,
    }


def get_library_versions() -> dict[str, str]:
    """Recopila las versiones de Python y de las librerías de modelado."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in TRACKED_LIBRARIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _sanitize_param_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_sanitize_param_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _sanitize_param_value(v) for k, v in value.items()}
    return repr(value)


def extract_hyperparameters(
    model: Any,
    training_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extrae los hiperparámetros efectivos del estimador y de la búsqueda."""
    training_metadata = training_metadata or {}
    recorded: dict[str, Any] = {}

    if "best_params" in training_metadata:
        recorded["selected_params"] = _sanitize_param_value(training_metadata["best_params"])
    elif "parameters" in training_metadata:
        recorded["selected_params"] = _sanitize_param_value(training_metadata["parameters"])

    estimator = model
    if isinstance(estimator, TransformedTargetRegressor):
        estimator = getattr(estimator, "regressor_", None) or estimator.regressor
    if isinstance(estimator, Pipeline) and "model" in estimator.named_steps:
        estimator = estimator.named_steps["model"]

    if hasattr(estimator, "get_params"):
        params = estimator.get_params(deep=False)
        recorded["estimator_class"] = type(estimator).__name__
        recorded["estimator_params"] = {
            str(key): _sanitize_param_value(val)
            for key, val in params.items()
        }
    return recorded


def build_reproducibility_context(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
    numeric_features: list[str],
    target: str,
    mainstream_price_limit: int | float,
    random_state: int,
    test_size: float = 0.2,
    data_path: Path | None = None,
    raw_df: pd.DataFrame | None = None,
    cleaning_module: Any = data_cleaning,
) -> dict[str, Any]:
    """Construye el contexto completo de reproducibilidad del experimento."""
    cleaning_info = get_cleaning_version(cleaning_module)
    return {
        "target": target,
        "mainstream_price_limit": mainstream_price_limit,
        "random_state": int(random_state),
        "cleaning_version": cleaning_info["version"],
        "cleaning_fingerprint": cleaning_info,
        "dataset_fingerprint": compute_dataset_fingerprint(
            X,
            target=y,
            data_path=data_path,
        ),
        "partition": compute_partition_metadata(
            X_train,
            X_test,
            random_state=random_state,
            test_size=test_size,
            raw_df=raw_df,
        ),
        "variables": {
            "features": list(features),
            "categorical_features": list(categorical_features),
            "numeric_features": list(numeric_features),
            "target": target,
        },
        "library_versions": get_library_versions(),
    }


def check_artifact_compatibility(
    artifact: dict[str, Any],
    *,
    features: list[str],
    categorical_features: list[str] | None = None,
    numeric_features: list[str] | None = None,
    reproducibility_context: dict[str, Any] | None = None,
    target: str = "dynx_totalvalue_x",
    mainstream_price_limit: int | float = 5_000_000,
) -> tuple[bool, list[str]]:
    """Valida si un artefacto coincide con los datos, limpieza y partición actuales."""
    reasons: list[str] = []
    if not isinstance(artifact, dict):
        return False, ["El artefacto no es un diccionario válido."]

    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        reasons.append(
            f"Versión de artefacto incompatible: {artifact.get('artifact_version')!r} "
            f"(esperada: {ARTIFACT_VERSION})."
        )

    if artifact.get("features") != list(features):
        reasons.append("Las variables del modelo no coinciden con las actuales.")

    if (
        categorical_features is not None
        and artifact.get("categorical_features") != list(categorical_features)
    ):
        reasons.append("Las variables categóricas no coinciden con las actuales.")

    if (
        numeric_features is not None
        and "numeric_features" in artifact
        and artifact.get("numeric_features") != list(numeric_features)
    ):
        reasons.append("Las variables numéricas no coinciden con las actuales.")

    metadata = artifact.get("metadata", {})
    if not isinstance(metadata, dict):
        return False, ["El artefacto no contiene metadatos válidos."]

    expected_target = (
        reproducibility_context.get("target", target)
        if reproducibility_context
        else target
    )
    expected_limit = (
        reproducibility_context.get("mainstream_price_limit", mainstream_price_limit)
        if reproducibility_context
        else mainstream_price_limit
    )

    if metadata.get("target") != expected_target:
        reasons.append("La variable objetivo no coincide.")
    if metadata.get("mainstream_price_limit") != expected_limit:
        reasons.append("El límite de precio generalista no coincide.")

    # Todos los artefactos vigentes deben incluir metadatos de reproducibilidad.
    for required_key in (
        "cleaning_version",
        "cleaning_fingerprint",
        "dataset_fingerprint",
        "partition",
        "random_state",
        "hyperparameters",
        "library_versions",
    ):
        if required_key not in metadata:
            reasons.append(f"Falta metadato de reproducibilidad requerido: {required_key}.")

    if reproducibility_context is not None:
        if metadata.get("cleaning_version") != reproducibility_context.get("cleaning_version"):
            reasons.append(
                f"La versión de limpieza no coincide: artefacto={metadata.get('cleaning_version')!r}, "
                f"actual={reproducibility_context.get('cleaning_version')!r}."
            )

        saved_cleaning_fp = metadata.get("cleaning_fingerprint", {})
        current_cleaning_fp = reproducibility_context.get("cleaning_fingerprint", {})
        if (
            isinstance(saved_cleaning_fp, dict)
            and isinstance(current_cleaning_fp, dict)
            and saved_cleaning_fp.get("code_sha256") != current_cleaning_fp.get("code_sha256")
        ):
            reasons.append("El código de limpieza de datos ha cambiado respecto al artefacto.")

        saved_ds = metadata.get("dataset_fingerprint", {})
        current_ds = reproducibility_context.get("dataset_fingerprint", {})
        if not isinstance(saved_ds, dict) or not isinstance(current_ds, dict):
            reasons.append("Huella del dataset inválida.")
        else:
            if (
                saved_ds.get("file_sha256") is not None
                and current_ds.get("file_sha256") is not None
                and saved_ds.get("file_sha256") != current_ds.get("file_sha256")
            ):
                reasons.append("El archivo CSV de origen ha cambiado (file_sha256 distinto).")
            if saved_ds.get("dataframe_sha256") != current_ds.get("dataframe_sha256"):
                reasons.append("Los datos de entrada limpios no coinciden (dataframe_sha256 distinto).")
            if saved_ds.get("target_sha256") != current_ds.get("target_sha256"):
                reasons.append("Los valores de la variable objetivo no coinciden (target_sha256 distinto).")
            if saved_ds.get("rows") != current_ds.get("rows"):
                reasons.append("El número de filas del dataset no coincide.")

        if metadata.get("random_state") != reproducibility_context.get("random_state"):
            reasons.append("La semilla aleatoria (random_state) no coincide.")

        saved_part = metadata.get("partition", {})
        current_part = reproducibility_context.get("partition", {})
        if not isinstance(saved_part, dict) or not isinstance(current_part, dict):
            reasons.append("Metadatos de partición inválidos.")
        else:
            if saved_part.get("partition_sha256") != current_part.get("partition_sha256"):
                reasons.append("Los índices de la partición train/test no coinciden.")
            if saved_part.get("train_indices") != current_part.get("train_indices"):
                reasons.append("Los índices de entrenamiento no coinciden.")
            if saved_part.get("test_indices") != current_part.get("test_indices"):
                reasons.append("Los índices de test no coinciden.")

    return len(reasons) == 0, reasons


def archive_historical_artifact(
    path: Path,
    *,
    historical_dir: Path | None = None,
) -> Path | None:
    """Conserva un artefacto previo en la subcarpeta histórica antes de reemplazarlo."""
    if not path.is_file():
        return None

    target_dir = historical_dir if historical_dir is not None else path.parent / HISTORICAL_SUBDIR
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        existing = joblib.load(path)
        version = existing.get("artifact_version", 1) if isinstance(existing, dict) else 1
        cleaning_ver = (
            existing.get("metadata", {}).get("cleaning_version", "legacy")
            if isinstance(existing, dict)
            else "legacy"
        )
    except Exception:
        version = 1
        cleaning_ver = "legacy"

    file_hash = (_hash_file(path) or "unknown")[:8]
    archive_name = f"v{version}_clean-{cleaning_ver}_{file_hash}_{path.name}"
    archive_path = target_dir / archive_name
    if not archive_path.exists():
        shutil.copy2(path, archive_path)
    return archive_path


def preserve_historical_artifacts(
    artifacts_dir: Path,
    *,
    historical_dir: Path | None = None,
) -> list[Path]:
    """Archiva todos los artefactos ``*.joblib`` existentes como referencia histórica."""
    if not artifacts_dir.is_dir():
        return []

    archived: list[Path] = []
    for path in sorted(artifacts_dir.glob("*.joblib")):
        dest = archive_historical_artifact(path, historical_dir=historical_dir)
        if dest is not None:
            archived.append(dest)
    return archived


def save_artifact(
    path: Path,
    *,
    model_name: str,
    model: Any,
    features: list[str],
    categorical_features: list[str],
    numeric_features: list[str] | None = None,
    metrics: dict[str, dict[str, float]],
    metadata: dict[str, Any],
    reproducibility_context: dict[str, Any] | None = None,
    preserve_existing: bool = True,
) -> None:
    """Guarda un modelo entrenado junto con su contexto completo de reproducibilidad."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if preserve_existing and path.exists():
        archive_historical_artifact(path)

    full_metadata = dict(metadata)
    if reproducibility_context is not None:
        for key, value in reproducibility_context.items():
            full_metadata.setdefault(key, value)

    training_meta = full_metadata.get("training", {})
    full_metadata.setdefault(
        "hyperparameters",
        extract_hyperparameters(model, training_meta if isinstance(training_meta, dict) else None),
    )
    full_metadata.setdefault("library_versions", get_library_versions())

    if numeric_features is None:
        categorical_set = set(categorical_features)
        numeric_features = [col for col in features if col not in categorical_set]

    full_metadata.setdefault(
        "variables",
        {
            "features": list(features),
            "categorical_features": list(categorical_features),
            "numeric_features": list(numeric_features),
            "target": full_metadata.get("target"),
        },
    )

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": model_name,
        "model": model,
        "features": list(features),
        "categorical_features": list(categorical_features),
        "numeric_features": list(numeric_features),
        "metrics": metrics,
        "metadata": full_metadata,
    }
    joblib.dump(artifact, path)


def load_artifact(
    path: Path,
    *,
    features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    numeric_features: list[str] | None = None,
    reproducibility_context: dict[str, Any] | None = None,
    allow_historical: bool = False,
) -> dict[str, Any]:
    """Carga un artefacto y opcionalmente verifica su compatibilidad con los datos actuales."""
    artifact = joblib.load(path)
    if not allow_historical and (features is not None or reproducibility_context is not None):
        expected_features = (
            features
            if features is not None
            else reproducibility_context.get("variables", {}).get("features", [])
        )
        compatible, reasons = check_artifact_compatibility(
            artifact,
            features=expected_features,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            reproducibility_context=reproducibility_context,
        )
        if not compatible:
            details = "; ".join(reasons)
            raise ValueError(
                f"{path} es incompatible con los datos o la limpieza actuales: {details}"
            )
    return artifact

