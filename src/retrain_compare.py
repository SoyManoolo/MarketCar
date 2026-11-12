"""Reentrenamiento y comparación reproducible de modelos mediante validación anidada.

Aplica el protocolo fijado en ``src/validation.py`` sobre ``X_train``:
- Validación anidada agrupada por vehículo (``build_vehicle_groups`` y
  ``build_nested_cv_splits``), sin consultar ``X_test`` para hiperparámetros,
  objetivo (``log1p`` frente a rupias directas) ni selección del modelo.
- Parada temprana (early stopping) en LightGBM, XGBoost y CatBoost usando
  exclusivamente una partición interna agrupada del conjunto de ajuste del fold.
- Selección pre-registrada por MAE en rupias y bootstrap pareado.
- Registro paso a paso y puntos de control por fold externo para reanudar tras
  interrupciones.

Ejemplos:
    uv run python src/retrain_compare.py --quick
    uv run python src/retrain_compare.py --device cuda --n-jobs 4 --output-dir models/comparison
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from catboost import CatBoostRegressor
import joblib
from lightgbm import LGBMRegressor, early_stopping
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from main import (
    DATA_PATH,
    MAINSTREAM_PRICE_LIMIT,
    RANDOM_STATE,
    TARGET,
    load_data_with_context,
)
from modeling import (
    TargetTransformedModel,
    build_preprocessor,
    evaluate_model,
    preserve_historical_artifacts,
    save_artifact,
)
from validation import (
    BOOTSTRAP_ITERATIONS,
    DEFAULT_NESTED_INNER_FOLDS,
    DEFAULT_NESTED_OUTER_FOLDS,
    PROTOCOL_VERSION,
    build_grouped_kfold_splits,
    build_nested_cv_splits,
    build_vehicle_groups,
    calculate_protocol_metrics,
    evaluate_by_price_band,
    hash_cv_splits,
    select_best_candidate,
)

ALL_MODEL_KEYS = (
    "linear",
    "ridge",
    "random_forest",
    "extra_trees",
    "catboost",
    "lightgbm",
    "xgboost",
)
DEFAULT_MODEL_KEYS = ("ridge", "linear", "lightgbm", "xgboost", "catboost")

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "linear": {"display_name": "Linear Regression", "complexity_rank": 0, "boosting": False},
    "ridge": {"display_name": "Ridge Regression", "complexity_rank": 1, "boosting": False},
    "lightgbm": {"display_name": "LightGBM", "complexity_rank": 2, "boosting": True},
    "xgboost": {"display_name": "XGBoost", "complexity_rank": 2, "boosting": True},
    "catboost": {"display_name": "CatBoost", "complexity_rank": 2, "boosting": True},
    "random_forest": {"display_name": "Random Forest", "complexity_rank": 3, "boosting": False},
    "extra_trees": {"display_name": "Extra Trees", "complexity_rank": 4, "boosting": False},
}

TARGET_TRANSFORMS = ("log1p", "raw")
EARLY_STOPPING_FOLDS = 6  # ~16.7 % interno agrupado por vehículo para early stopping.


def resolve_n_jobs(n_jobs: int) -> int:
    """Valida y resuelve ``--n-jobs`` a un número positivo de hilos."""
    if n_jobs == 0 or n_jobs < -1:
        raise ValueError("--n-jobs debe ser >= 1 o -1 (todos los hilos disponibles).")
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    return int(n_jobs)


def resolve_xgb_execution(device: str, n_jobs: int) -> tuple[int, int]:
    """Devuelve ``(search_jobs, model_n_jobs)`` para XGBoost según el dispositivo.

    Con ``cuda``, fuerza ejecución y búsqueda en serie (``n_jobs=1``).
    """
    if device == "cuda":
        return 1, 1
    resolved = resolve_n_jobs(n_jobs)
    return 1, resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define los argumentos de línea de comandos del flujo de reentrenamiento."""
    parser = argparse.ArgumentParser(
        description="Reentrena y compara modelos MarketCar con validación anidada agrupada.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", *ALL_MODEL_KEYS],
        default=list(DEFAULT_MODEL_KEYS),
        help=(
            "Modelos a reentrenar y comparar. Por defecto: "
            + " ".join(DEFAULT_MODEL_KEYS)
            + ". Usa 'all' para incluir también Random Forest y Extra Trees."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Dispositivo de cómputo solo para XGBoost (por defecto: cpu). Con cuda usa n_jobs=1.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Número máximo de hilos en CPU (-1 usa todos los núcleos; por defecto: 4).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Modo rápido de prueba con pocos candidatos, árboles reducidos y 2x2 folds.",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="Configuraciones de hiperparámetros por transformación de objetivo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/comparison"),
        help="Carpeta para artefactos, checkpoints y log de progreso (por defecto: models/comparison).",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/comparacion_modelos.md"),
        help="Ruta del informe comparativo Markdown (por defecto: reports/comparacion_modelos.md).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Ruta al CSV de entrada (por defecto: data/cars_details_merges.csv).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reanuda desde checkpoints compatibles en --output-dir (activado por defecto).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignora checkpoints previos y ejecuta todos los folds desde cero.",
    )
    return parser.parse_args(argv)


def requested_models(models: list[str]) -> list[str]:
    """Expande ``all`` y elimina duplicados conservando el orden."""
    names = list(ALL_MODEL_KEYS) if "all" in models else models
    return list(dict.fromkeys(names))


class StepLogger:
    """Registra eventos paso a paso en consola, ``progress.log`` y ``progress.jsonl``."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.text_log_path = self.output_dir / "progress.log"
        self.jsonl_log_path = self.output_dir / "progress.jsonl"

    def log(
        self,
        event: str,
        *,
        model: str | None = None,
        outer_fold: int | None = None,
        elapsed_sec: float | None = None,
        message: str = "",
        **extra: Any,
    ) -> None:
        """Escribe un paso en ambos formatos de log con volcado inmediato."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        parts = [f"[{timestamp}]", f"event={event}"]
        if model is not None:
            parts.append(f"model={model}")
        if outer_fold is not None:
            parts.append(f"fold={outer_fold}")
        if elapsed_sec is not None:
            parts.append(f"elapsed={elapsed_sec:.2f}s")
        for key, value in extra.items():
            if value is not None:
                parts.append(f"{key}={value}")
        if message:
            parts.append(f"| {message}")

        line = " ".join(parts)
        print(line, flush=True)
        with self.text_log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

        payload = {
            "timestamp": timestamp,
            "event": event,
            "model": model,
            "outer_fold": outer_fold,
            "elapsed_sec": round(float(elapsed_sec), 4) if elapsed_sec is not None else None,
            "message": message,
            **extra,
        }
        with self.jsonl_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()


def build_candidate_configs(
    model_key: str,
    *,
    quick: bool,
    n_candidates: int | None,
    random_state: int,
) -> list[dict[str, Any]]:
    """Construye la lista reproducible de candidatos ``(target_transform, params)``."""
    base_n = 1 if quick else (n_candidates if n_candidates is not None else 8)
    if base_n < 1:
        raise ValueError("--n-candidates debe ser >= 1.")

    param_sets: list[dict[str, Any]]
    if model_key == "linear":
        param_sets = [{}]
    elif model_key == "ridge":
        alphas = [1.0, 10.0] if quick else [0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0]
        param_sets = [{"alpha": float(a)} for a in alphas]
    elif model_key == "lightgbm":
        max_trees = 120 if quick else 2_500
        space = {
            "learning_rate": [0.03, 0.05, 0.08],
            "num_leaves": [31, 63, 127],
            "max_depth": [-1, 10, 15],
            "min_child_samples": [10, 20, 40],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_lambda": [0.0, 0.1, 1.0, 5.0],
        }
        sampled = list(ParameterSampler(space, n_iter=base_n, random_state=random_state))
        param_sets = [{"n_estimators": max_trees, **p} for p in sampled]
    elif model_key == "xgboost":
        max_trees = 120 if quick else 2_500
        space = {
            "learning_rate": [0.03, 0.05, 0.08],
            "max_depth": [4, 6, 8, 10],
            "min_child_weight": [1, 5, 10],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [1.0, 5.0, 10.0],
        }
        sampled = list(ParameterSampler(space, n_iter=base_n, random_state=random_state))
        param_sets = [{"n_estimators": max_trees, **p} for p in sampled]
    elif model_key == "catboost":
        max_trees = 60 if quick else 2_500
        depths = [4, 6] if quick else [6, 8, 10]
        space = {
            "depth": depths,
            "learning_rate": [0.03, 0.05, 0.08],
            "l2_leaf_reg": [1, 3, 7, 10],
            "random_strength": [0.5, 1.0, 2.0],
            "bagging_temperature": [0.5, 1.0, 2.0],
        }
        sampled = list(ParameterSampler(space, n_iter=base_n, random_state=random_state))
        param_sets = [{"iterations": max_trees, **p} for p in sampled]
    elif model_key == "random_forest":
        param_sets = [
            {
                "n_estimators": 40 if quick else 300,
                "max_depth": 15 if quick else 30,
                "min_samples_leaf": 2,
                "max_features": 0.5,
            }
        ]
    elif model_key == "extra_trees":
        param_sets = [
            {
                "n_estimators": 40 if quick else 300,
                "max_depth": 15 if quick else 30,
                "min_samples_leaf": 2,
                "max_features": 0.5,
            }
        ]
    else:
        raise ValueError(f"Modelo no soportado: {model_key!r}")

    candidates: list[dict[str, Any]] = []
    for idx, params in enumerate(param_sets):
        for target_transform in TARGET_TRANSFORMS:
            candidates.append({
                "candidate_id": f"{model_key}_c{idx}_{target_transform}",
                "target_transform": target_transform,
                "params": params,
            })
    return candidates


def _transform_target(y: pd.Series | np.ndarray, target_transform: str) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if target_transform == "log1p":
        return np.log1p(arr)
    if target_transform == "raw":
        return arr
    raise ValueError(f"Transformación no soportada: {target_transform!r}")


def _split_internal_early_stopping(
    X_fit: pd.DataFrame,
    groups_fit: pd.Series,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide ``X_fit`` en entrenamiento interno (~83 %) y validación de parada (~17 %).

    Agrupa por ``vehicle_group_id`` para evitar fugas entre reanuncios y nunca usa
    datos externos al conjunto de ajuste actual.
    """
    n_groups = int(groups_fit.loc[X_fit.index].nunique())
    n_splits = min(EARLY_STOPPING_FOLDS, max(2, n_groups))
    splits = build_grouped_kfold_splits(
        X_fit.index,
        groups_fit,
        n_splits=n_splits,
        random_state=random_state,
    )
    return splits[0]


def fit_candidate_model(
    model_key: str,
    *,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    groups_fit: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    target_transform: str,
    params: dict[str, Any],
    random_state: int,
    n_jobs: int,
    xgb_device: str,
    early_stopping_rounds: int,
) -> tuple[TargetTransformedModel, dict[str, Any]]:
    """Entrena un candidato usando únicamente ``X_fit`` (con ES interno en boosting)."""
    best_iteration: int | None = None
    y_fit_arr = _transform_target(y_fit, target_transform)

    if not MODEL_SPECS[model_key]["boosting"]:
        scale_numeric = model_key in {"linear", "ridge"}
        preprocessor = build_preprocessor(
            numeric_features,
            categorical_features,
            scale_numeric=scale_numeric,
        )
        if model_key == "linear":
            estimator = LinearRegression()
        elif model_key == "ridge":
            estimator = Ridge(**params)
        elif model_key == "random_forest":
            estimator = RandomForestRegressor(
                **params,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        elif model_key == "extra_trees":
            estimator = ExtraTreesRegressor(
                **params,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        else:
            raise ValueError(f"Modelo no soportado: {model_key!r}")

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_fit, y_fit_arr)
        wrapped = TargetTransformedModel(pipeline, target_transform=target_transform)
        return wrapped, {"best_iteration": None}

    # Para modelos de boosting: partición interna agrupada SOLO sobre X_fit.
    es_tr_pos, es_val_pos = _split_internal_early_stopping(
        X_fit,
        groups_fit,
        random_state=random_state,
    )
    X_es_tr = X_fit.iloc[es_tr_pos]
    X_es_val = X_fit.iloc[es_val_pos]
    y_es_tr = y_fit_arr[es_tr_pos]
    y_es_val = y_fit_arr[es_val_pos]

    if model_key == "catboost":
        model = CatBoostRegressor(
            **params,
            cat_features=categorical_features,
            random_seed=random_state,
            allow_writing_files=False,
            verbose=0,
            use_best_model=True,
            thread_count=n_jobs,
        )
        model.fit(
            X_es_tr,
            y_es_tr,
            eval_set=(X_es_val, y_es_val),
            early_stopping_rounds=early_stopping_rounds,
        )
        best_iteration = int(model.get_best_iteration())
        wrapped = TargetTransformedModel(model, target_transform=target_transform)
        return wrapped, {"best_iteration": best_iteration}

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=False,
    )
    X_es_tr_t = preprocessor.fit_transform(X_es_tr)
    X_es_val_t = preprocessor.transform(X_es_val)

    if model_key == "lightgbm":
        booster = LGBMRegressor(
            **params,
            random_state=random_state,
            n_jobs=n_jobs,
            subsample_freq=1,
            verbosity=-1,
        )
        booster.fit(
            X_es_tr_t,
            y_es_tr,
            eval_X=X_es_val_t,
            eval_y=y_es_val,
            callbacks=[early_stopping(early_stopping_rounds, verbose=False)],
        )
        best_iteration = int(getattr(booster, "best_iteration_", params.get("n_estimators", 0)) or 0)
    elif model_key == "xgboost":
        _, xgb_n_jobs = resolve_xgb_execution(xgb_device, n_jobs)
        booster = XGBRegressor(
            **params,
            objective="reg:squarederror",
            tree_method="hist",
            device=xgb_device,
            eval_metric="rmse",
            early_stopping_rounds=early_stopping_rounds,
            n_jobs=xgb_n_jobs,
            random_state=random_state,
        )
        booster.fit(
            X_es_tr_t,
            y_es_tr,
            eval_set=[(X_es_val_t, y_es_val)],
            verbose=False,
        )
        best_iteration = int(getattr(booster, "best_iteration", params.get("n_estimators", 0)) or 0)
    else:
        raise ValueError(f"Modelo no soportado: {model_key!r}")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", booster)])
    wrapped = TargetTransformedModel(pipeline, target_transform=target_transform)
    return wrapped, {"best_iteration": best_iteration}


def evaluate_inner_candidates(
    model_key: str,
    *,
    X_outer_train: pd.DataFrame,
    y_outer_train: pd.Series,
    groups_outer_train: pd.Series,
    inner_splits: list[tuple[np.ndarray, np.ndarray]],
    candidates: list[dict[str, Any]],
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
    n_jobs: int,
    xgb_device: str,
    early_stopping_rounds: int,
) -> dict[str, Any]:
    """Evalúa todos los candidatos en los folds internos y elige el de menor MAE INR."""
    candidate_records: list[dict[str, Any]] = []
    for cand_idx, cand in enumerate(candidates):
        fold_maes: list[float] = []
        fold_iters: list[int] = []
        for inner_fold_idx, (in_tr_pos, in_val_pos) in enumerate(inner_splits):
            X_in_tr = X_outer_train.iloc[in_tr_pos]
            y_in_tr = y_outer_train.iloc[in_tr_pos]
            X_in_val = X_outer_train.iloc[in_val_pos]
            y_in_val = y_outer_train.iloc[in_val_pos]

            fitted_model, fit_info = fit_candidate_model(
                model_key,
                X_fit=X_in_tr,
                y_fit=y_in_tr,
                groups_fit=groups_outer_train,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                target_transform=cand["target_transform"],
                params=cand["params"],
                random_state=random_state + inner_fold_idx + cand_idx * 13,
                n_jobs=n_jobs,
                xgb_device=xgb_device,
                early_stopping_rounds=early_stopping_rounds,
            )
            preds = fitted_model.predict(X_in_val)
            metrics = calculate_protocol_metrics(y_in_val, preds)
            fold_maes.append(metrics["mae"])
            if fit_info["best_iteration"] is not None:
                fold_iters.append(fit_info["best_iteration"])

        candidate_records.append({
            "candidate_id": cand["candidate_id"],
            "target_transform": cand["target_transform"],
            "params": cand["params"],
            "mean_inner_mae": float(np.mean(fold_maes)),
            "sd_inner_mae": float(np.std(fold_maes, ddof=1)) if len(fold_maes) > 1 else 0.0,
            "mean_best_iteration": float(np.mean(fold_iters)) if fold_iters else None,
        })

    candidate_records.sort(key=lambda r: (r["mean_inner_mae"], r["sd_inner_mae"]))
    best_by_transform: dict[str, float] = {}
    for transform in TARGET_TRANSFORMS:
        subset = [r for r in candidate_records if r["target_transform"] == transform]
        if subset:
            best_by_transform[transform] = float(subset[0]["mean_inner_mae"])

    return {
        "best_candidate": candidate_records[0],
        "best_inner_mae_by_transform": best_by_transform,
        "all_candidates": candidate_records,
    }


def _compute_run_signature(
    *,
    model_key: str,
    reproducibility_context: dict[str, Any],
    outer_splits_sha256: str,
    inner_splits_sha256: list[str],
    quick: bool,
    candidates: list[dict[str, Any]],
    xgb_device: str,
) -> str:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "dataset_sha256": reproducibility_context["dataset_fingerprint"]["dataframe_sha256"],
        "target_sha256": reproducibility_context["dataset_fingerprint"]["target_sha256"],
        "cleaning_sha256": reproducibility_context["cleaning_fingerprint"]["code_sha256"],
        "outer_splits_sha256": outer_splits_sha256,
        "inner_splits_sha256": inner_splits_sha256,
        "quick": quick,
        "candidates": candidates,
        "device": xgb_device if model_key == "xgboost" else "cpu",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_joblib_dump(payload: Any, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    joblib.dump(payload, tmp_path)
    os.replace(tmp_path, target_path)


def _atomic_json_dump(payload: Any, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, target_path)


def run_retrain_comparison(
    *,
    models: list[str],
    device: str = "cpu",
    n_jobs: int = 4,
    quick: bool = False,
    n_candidates: int | None = None,
    output_dir: Path = Path("models/comparison"),
    report_output: Path = Path("reports/comparacion_modelos.md"),
    data_path: Path = DATA_PATH,
    resume: bool = True,
    fresh: bool = False,
) -> dict[str, Any]:
    """Ejecuta el reentrenamiento y comparación completa siguiendo el protocolo."""
    run_start = time.perf_counter()
    resolved_jobs = resolve_n_jobs(n_jobs)
    selected_models = requested_models(models)
    use_resume = resume and not fresh

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logger = StepLogger(output_dir)

    raw_df = pd.read_csv(data_path, low_memory=False)
    (
        features,
        categorical_features,
        numeric_features,
        X_train,
        X_test,
        y_train,
        y_test,
        base_context,
    ) = load_data_with_context(data_path)

    # Reutilizamos el contexto base de src/main.py y añadimos los metadatos de ejecución.
    reproducibility_context = dict(base_context)
    reproducibility_context["execution"] = {
        "protocol_version": PROTOCOL_VERSION,
        "xgboost_device": device,
        "n_jobs": resolved_jobs,
        "quick": bool(quick),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    }

    train_groups = build_vehicle_groups(raw_df, X_train.index)
    outer_splits_n = 2 if quick else DEFAULT_NESTED_OUTER_FOLDS
    inner_splits_n = 2 if quick else DEFAULT_NESTED_INNER_FOLDS
    early_stopping_rounds = 15 if quick else 50

    nested_splits = build_nested_cv_splits(
        X_train.index,
        train_groups,
        outer_splits=outer_splits_n,
        inner_splits=inner_splits_n,
        random_state=RANDOM_STATE,
    )
    outer_splits_sha256 = hash_cv_splits(
        X_train.index,
        [(d["outer_train_positions"], d["outer_val_positions"]) for d in nested_splits],
    )
    inner_splits_sha256 = [d["inner_splits_sha256"] for d in nested_splits]

    logger.log(
        "start_run",
        message=(
            f"models={','.join(selected_models)} quick={quick} "
            f"outer={outer_splits_n} inner={inner_splits_n} "
            f"xgb_device={device} n_jobs={resolved_jobs} resume={use_resume}"
        ),
        outer_splits_sha256=outer_splits_sha256[:12],
    )

    preserve_historical_artifacts(output_dir)

    candidate_inputs: list[dict[str, Any]] = []
    model_reports: dict[str, Any] = {}
    frozen_test_rows: list[dict[str, Any]] = []

    for model_idx, model_key in enumerate(selected_models):
        model_start = time.perf_counter()
        display_name = MODEL_SPECS[model_key]["display_name"]
        model_n_jobs = 1 if (model_key == "xgboost" and device == "cuda") else resolved_jobs
        candidates = build_candidate_configs(
            model_key,
            quick=quick,
            n_candidates=n_candidates,
            random_state=RANDOM_STATE + model_idx * 101,
        )
        run_sig = _compute_run_signature(
            model_key=model_key,
            reproducibility_context=reproducibility_context,
            outer_splits_sha256=outer_splits_sha256,
            inner_splits_sha256=inner_splits_sha256,
            quick=quick,
            candidates=candidates,
            xgb_device=device,
        )

        logger.log(
            "start_model",
            model=model_key,
            n_candidates=len(candidates),
            n_jobs=model_n_jobs,
            device=device if model_key == "xgboost" else "cpu",
        )

        oof_predictions = np.full(len(X_train), np.nan, dtype=float)
        outer_fold_details: list[dict[str, Any]] = []

        for outer_item in nested_splits:
            outer_fold = int(outer_item["outer_fold"]) + 1
            ckpt_path = checkpoints_dir / f"{model_key}_fold_{outer_fold}.joblib"

            resumed_data = None
            if use_resume and ckpt_path.is_file():
                try:
                    loaded_ckpt = joblib.load(ckpt_path)
                    if loaded_ckpt.get("run_signature") == run_sig:
                        resumed_data = loaded_ckpt
                except Exception:
                    resumed_data = None

            outer_train_pos = outer_item["outer_train_positions"]
            outer_val_pos = outer_item["outer_val_positions"]

            if resumed_data is not None:
                oof_predictions[outer_val_pos] = resumed_data["val_predictions"]
                outer_fold_details.append(resumed_data["fold_summary"])
                logger.log(
                    "outer_fold_resumed",
                    model=model_key,
                    outer_fold=outer_fold,
                    elapsed_sec=resumed_data["fold_summary"]["elapsed_sec"],
                    mae_inr=round(resumed_data["fold_summary"]["outer_mae"], 2),
                    target_transform=resumed_data["fold_summary"]["selected_target_transform"],
                )
                continue

            fold_start = time.perf_counter()
            logger.log("outer_fold_start", model=model_key, outer_fold=outer_fold)

            X_outer_tr = X_train.iloc[outer_train_pos]
            y_outer_tr = y_train.iloc[outer_train_pos]
            X_outer_val = X_train.iloc[outer_val_pos]
            y_outer_val = y_train.iloc[outer_val_pos]

            inner_result = evaluate_inner_candidates(
                model_key,
                X_outer_train=X_outer_tr,
                y_outer_train=y_outer_tr,
                groups_outer_train=train_groups,
                inner_splits=outer_item["inner_splits"],
                candidates=candidates,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                random_state=RANDOM_STATE + outer_fold * 31,
                n_jobs=model_n_jobs,
                xgb_device=device,
                early_stopping_rounds=early_stopping_rounds,
            )

            best_cand = inner_result["best_candidate"]
            outer_model, outer_fit_info = fit_candidate_model(
                model_key,
                X_fit=X_outer_tr,
                y_fit=y_outer_tr,
                groups_fit=train_groups,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                target_transform=best_cand["target_transform"],
                params=best_cand["params"],
                random_state=RANDOM_STATE + outer_fold * 97,
                n_jobs=model_n_jobs,
                xgb_device=device,
                early_stopping_rounds=early_stopping_rounds,
            )
            val_preds = outer_model.predict(X_outer_val)
            oof_predictions[outer_val_pos] = val_preds
            val_metrics = calculate_protocol_metrics(y_outer_val, val_preds)
            fold_elapsed = time.perf_counter() - fold_start

            fold_summary = {
                "outer_fold": outer_fold,
                "train_rows": int(len(X_outer_tr)),
                "val_rows": int(len(X_outer_val)),
                "selected_target_transform": best_cand["target_transform"],
                "selected_params": best_cand["params"],
                "mean_inner_mae": best_cand["mean_inner_mae"],
                "best_inner_mae_by_transform": inner_result["best_inner_mae_by_transform"],
                "best_iteration": outer_fit_info["best_iteration"],
                "outer_mae": val_metrics["mae"],
                "outer_rmse": val_metrics["rmse"],
                "outer_r2": val_metrics["r2"],
                "outer_median_ape_pct": val_metrics["median_ape_pct"],
                "outer_within_10pct_pct": val_metrics["within_10pct_pct"],
                "outer_bias_inr": val_metrics["bias_inr"],
                "elapsed_sec": float(fold_elapsed),
            }
            _atomic_joblib_dump(
                {
                    "run_signature": run_sig,
                    "val_predictions": val_preds,
                    "fold_summary": fold_summary,
                },
                ckpt_path,
            )
            outer_fold_details.append(fold_summary)

            logger.log(
                "outer_fold_done",
                model=model_key,
                outer_fold=outer_fold,
                elapsed_sec=fold_elapsed,
                mae_inr=round(val_metrics["mae"], 2),
                target_transform=best_cand["target_transform"],
                best_iteration=outer_fit_info["best_iteration"],
            )

        if np.isnan(oof_predictions).any():
            raise RuntimeError(f"Predicciones OOF incompletas para {model_key}.")

        oof_metrics = calculate_protocol_metrics(y_train, oof_predictions)
        oof_bands = evaluate_by_price_band(y_train, oof_predictions)
        fold_maes = [float(item["outer_mae"]) for item in outer_fold_details]

        # Ajuste del artefacto final usando EXCLUSIVAMENTE X_train y sus folds agrupados internos.
        final_fit_start = time.perf_counter()
        final_ckpt_path = checkpoints_dir / f"{model_key}_final.joblib"
        final_artifact_path = output_dir / f"{model_key}_generalista.joblib"
        final_resumed = False

        if use_resume and final_ckpt_path.is_file() and final_artifact_path.is_file():
            try:
                loaded_final_ckpt = joblib.load(final_ckpt_path)
                if loaded_final_ckpt.get("run_signature") == run_sig:
                    final_summary = loaded_final_ckpt["final_summary"]
                    final_artifact = joblib.load(final_artifact_path)
                    final_model = final_artifact["model"]
                    final_resumed = True
            except Exception:
                final_resumed = False

        if not final_resumed:
            common_inner_splits = build_grouped_kfold_splits(
                X_train.index,
                train_groups,
                n_splits=inner_splits_n,
                random_state=RANDOM_STATE,
            )
            final_inner = evaluate_inner_candidates(
                model_key,
                X_outer_train=X_train,
                y_outer_train=y_train,
                groups_outer_train=train_groups,
                inner_splits=common_inner_splits,
                candidates=candidates,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                random_state=RANDOM_STATE + 777,
                n_jobs=model_n_jobs,
                xgb_device=device,
                early_stopping_rounds=early_stopping_rounds,
            )
            final_best = final_inner["best_candidate"]
            final_model, final_fit_info = fit_candidate_model(
                model_key,
                X_fit=X_train,
                y_fit=y_train,
                groups_fit=train_groups,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                target_transform=final_best["target_transform"],
                params=final_best["params"],
                random_state=RANDOM_STATE,
                n_jobs=model_n_jobs,
                xgb_device=device,
                early_stopping_rounds=early_stopping_rounds,
            )
            final_elapsed = time.perf_counter() - final_fit_start
            final_summary = {
                "target_transform": final_best["target_transform"],
                "selected_params": final_best["params"],
                "cv_mae": final_best["mean_inner_mae"],
                "best_inner_mae_by_transform": final_inner["best_inner_mae_by_transform"],
                "best_iteration": final_fit_info["best_iteration"],
                "elapsed_sec": float(final_elapsed),
                "device": device if model_key == "xgboost" else "cpu",
            }
            artifact_metrics = evaluate_model(final_model, X_train, y_train, X_test, y_test)
            save_artifact(
                final_artifact_path,
                model_name=display_name,
                model=final_model,
                features=features,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                metrics=artifact_metrics,
                metadata={
                    "target": TARGET,
                    "mainstream_price_limit": MAINSTREAM_PRICE_LIMIT,
                    "training_rows": len(X_train),
                    "test_rows": len(X_test),
                    "xgboost_device": device if model_key == "xgboost" else None,
                    "nested_cv_oof_metrics": oof_metrics,
                    "nested_cv_fold_maes": fold_maes,
                    "training": {
                        "parameters": final_best["params"],
                        "target_transform": final_best["target_transform"],
                        "best_iteration": final_fit_info["best_iteration"],
                        "cv_mae": final_best["mean_inner_mae"],
                        "device": final_summary["device"],
                        "strategy": (
                            f"NestedCV({outer_splits_n}x{inner_splits_n} grouped) + "
                            f"internal_ES({final_best['target_transform']})"
                        ),
                    },
                },
                reproducibility_context=reproducibility_context,
            )
            _atomic_joblib_dump(
                {"run_signature": run_sig, "final_summary": final_summary},
                final_ckpt_path,
            )

        logger.log(
            "final_fit_done",
            model=model_key,
            elapsed_sec=final_summary["elapsed_sec"],
            target_transform=final_summary["target_transform"],
            best_iteration=final_summary["best_iteration"],
            resumed=final_resumed,
        )

        # Referencia congelada en X_test (nunca usada para decidir).
        frozen_preds = final_model.predict(X_test)
        frozen_metrics = calculate_protocol_metrics(y_test, frozen_preds)
        frozen_test_rows.append({
            "model_key": model_key,
            "Modelo": display_name,
            "Objetivo final": final_summary["target_transform"],
            "MAE test INR": frozen_metrics["mae"],
            "RMSE test INR": frozen_metrics["rmse"],
            "R2 test": frozen_metrics["r2"],
            "Mediana APE %": frozen_metrics["median_ape_pct"],
            "Dentro ±10% %": frozen_metrics["within_10pct_pct"],
        })

        model_elapsed = time.perf_counter() - model_start
        model_reports[model_key] = {
            "model_key": model_key,
            "display_name": display_name,
            "complexity_rank": MODEL_SPECS[model_key]["complexity_rank"],
            "device": device if model_key == "xgboost" else "cpu",
            "n_jobs": model_n_jobs,
            "oof_predictions": oof_predictions,
            "oof_metrics": oof_metrics,
            "oof_price_bands": oof_bands.to_dict(orient="records"),
            "outer_folds": outer_fold_details,
            "fold_maes": fold_maes,
            "final_model_summary": final_summary,
            "artifact_path": str(final_artifact_path),
            "elapsed_sec": float(model_elapsed),
        }
        candidate_inputs.append({
            "name": model_key,
            "oof_predictions": oof_predictions,
            "fold_maes": fold_maes,
            "complexity_rank": MODEL_SPECS[model_key]["complexity_rank"],
        })

    selection = select_best_candidate(
        candidate_inputs,
        y_train,
        random_state=RANDOM_STATE,
    )
    total_elapsed = time.perf_counter() - run_start

    results_payload = {
        "protocol_version": PROTOCOL_VERSION,
        "quick": bool(quick),
        "outer_splits": outer_splits_n,
        "inner_splits": inner_splits_n,
        "xgboost_device": device,
        "n_jobs": resolved_jobs,
        "total_elapsed_sec": float(total_elapsed),
        "reproducibility_context": {
            key: value
            for key, value in reproducibility_context.items()
            if key != "partition"
        } | {
            "partition": {
                k: v
                for k, v in reproducibility_context["partition"].items()
                if k not in {"train_indices", "test_indices"}
            }
        },
        "outer_splits_sha256": outer_splits_sha256,
        "inner_splits_sha256": inner_splits_sha256,
        "selection": selection,
        "models": {
            k: {mk: mv for mk, mv in v.items() if mk != "oof_predictions"}
            for k, v in model_reports.items()
        },
        "frozen_test_reference": frozen_test_rows,
    }

    _atomic_json_dump(results_payload, output_dir / "comparison_results.json")

    markdown = render_comparison_markdown(
        results_payload=results_payload,
        model_reports=model_reports,
        X_train=X_train,
        y_train=y_train,
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(markdown, encoding="utf-8")

    logger.log(
        "comparison_done",
        elapsed_sec=total_elapsed,
        selected_model=selection["selected_model"],
        report=report_output.as_posix(),
    )
    return results_payload


def _markdown_table(frame: pd.DataFrame) -> str:
    frame = frame.copy().round(3).fillna("—")
    rows = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    rows.extend(
        "| " + " | ".join(map(str, row)) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def _segments_table(frame: pd.DataFrame, column: str, *, min_n: int = 1) -> pd.DataFrame:
    columns = ["Segmento", "n", "MAE INR", "Mediana APE %", "Cobertura ±10% %", "Sesgo INR"]
    rows: list[dict[str, Any]] = []
    for label, group in frame.groupby(column, observed=True):
        if len(group) < min_n:
            continue
        error = group["prediction"] - group["actual"]
        rows.append({
            "Segmento": str(label),
            "n": int(len(group)),
            "MAE INR": float(error.abs().mean()),
            "Mediana APE %": float((error.abs() / group["actual"]).median() * 100.0),
            "Cobertura ±10% %": float(((error.abs() / group["actual"]) <= 0.10).mean() * 100.0),
            "Sesgo INR": float(error.mean()),
        })
    return pd.DataFrame(rows, columns=columns)


def render_comparison_markdown(
    *,
    results_payload: dict[str, Any],
    model_reports: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> str:
    """Genera el informe Markdown ``reports/comparacion_modelos.md``."""
    selection = results_payload["selection"]
    winner_key = selection["selected_model"]
    winner_info = model_reports[winner_key]
    ctx = results_payload["reproducibility_context"]
    ds_fp = ctx["dataset_fingerprint"]
    clean_fp = ctx["cleaning_fingerprint"]

    mode_label = (
        f"Modo rápido (`--quick`, {results_payload['outer_splits']}x{results_payload['inner_splits']} folds)"
        if results_payload["quick"]
        else f"Protocolo completo ({results_payload['outer_splits']}x{results_payload['inner_splits']} folds)"
    )

    # 1. Tabla comparable principal (OOF sobre X_train).
    ranking_rows: list[dict[str, Any]] = []
    for row in selection["ranking"]:
        key = row["name"]
        rep = model_reports[key]
        transforms_used = sorted({f["selected_target_transform"] for f in rep["outer_folds"]})
        ranking_rows.append({
            "Modelo": rep["display_name"],
            "Objetivo validación": "/".join(transforms_used),
            "Objetivo final": rep["final_model_summary"]["target_transform"],
            "MAE OOF INR": row["mae"],
            "SD folds INR": row["fold_sd_mae"],
            "RMSE OOF INR": row["rmse"],
            "R²": row["r2"],
            "Mediana APE %": row["median_ape_pct"],
            "Dentro ±10% %": row["within_10pct_pct"],
            "Sesgo INR": row["bias_inr"],
            "MAE tramo ₹2–5M": row["high_band_mae"],
            "Tiempo (s)": rep["elapsed_sec"],
        })

    # 2. Dispersión entre folds externos.
    dispersion_rows: list[dict[str, Any]] = []
    for row in selection["ranking"]:
        key = row["name"]
        rep = model_reports[key]
        fold_maes = rep["fold_maes"]
        iters = [f["best_iteration"] for f in rep["outer_folds"] if f["best_iteration"] is not None]
        item: dict[str, Any] = {"Modelo": rep["display_name"]}
        for idx, mae_val in enumerate(fold_maes, start=1):
            item[f"Fold {idx} INR"] = mae_val
        item["Media INR"] = float(np.mean(fold_maes))
        item["SD INR"] = float(np.std(fold_maes, ddof=1)) if len(fold_maes) > 1 else 0.0
        item["Min INR"] = float(np.min(fold_maes))
        item["Max INR"] = float(np.max(fold_maes))
        item["Iter ES media"] = round(float(np.mean(iters)), 1) if iters else "—"
        dispersion_rows.append(item)

    # 3. Comparación de objetivos (log1p vs rupias directas) dentro de validación interna.
    transform_rows: list[dict[str, Any]] = []
    for row in selection["ranking"]:
        key = row["name"]
        rep = model_reports[key]
        log_maes = [
            f["best_inner_mae_by_transform"].get("log1p")
            for f in rep["outer_folds"]
            if "log1p" in f["best_inner_mae_by_transform"]
        ]
        raw_maes = [
            f["best_inner_mae_by_transform"].get("raw")
            for f in rep["outer_folds"]
            if "raw" in f["best_inner_mae_by_transform"]
        ]
        log_wins = sum(1 for f in rep["outer_folds"] if f["selected_target_transform"] == "log1p")
        raw_wins = sum(1 for f in rep["outer_folds"] if f["selected_target_transform"] == "raw")
        mean_log = float(np.mean(log_maes)) if log_maes else np.nan
        mean_raw = float(np.mean(raw_maes)) if raw_maes else np.nan
        transform_rows.append({
            "Modelo": rep["display_name"],
            "MAE CV medio log1p INR": mean_log,
            "MAE CV medio rupias INR": mean_raw,
            "Delta (log1p - rupias)": (mean_log - mean_raw) if (log_maes and raw_maes) else np.nan,
            "Folds log1p / rupias": f"{log_wins} / {raw_wins}",
            "Objetivo final": rep["final_model_summary"]["target_transform"],
        })

    # 4. Bootstrap pareado.
    bootstrap_df = pd.DataFrame(selection["paired_bootstrap"])
    if not bootstrap_df.empty:
        bootstrap_df = bootstrap_df.rename(columns={
            "leader": "Líder MAE",
            "challenger": "Retador",
            "delta_mae_inr": "Delta MAE INR",
            "ci95_low_inr": "IC95 inferior INR",
            "ci95_high_inr": "IC95 superior INR",
            "conclusive": "Concluyente",
        })

    # 5. Diagnóstico por segmentos sobre predicciones OOF del ganador provisional.
    winner_oof = winner_info["oof_predictions"]
    winner_bands_df = evaluate_by_price_band(y_train, winner_oof).rename(columns={
        "price_band": "Tramo de precio",
        "mae": "MAE INR",
        "rmse": "RMSE INR",
        "median_ape_pct": "Mediana APE %",
        "within_10pct_pct": "Dentro ±10% %",
        "bias_inr": "Sesgo INR",
    })

    seg_frame = pd.DataFrame({
        "brand_name": X_train["brand_name"].to_numpy(),
        "model_year": pd.to_numeric(X_train["model_year"], errors="coerce").to_numpy(),
        "actual": np.asarray(y_train, dtype=float),
        "prediction": np.asarray(winner_oof, dtype=float),
    })
    seg_frame["year_band"] = pd.cut(seg_frame["model_year"], [0, 2005, 2010, 2015, 2020, 2100])
    year_df = _segments_table(seg_frame, "year_band")
    brand_df = _segments_table(seg_frame, "brand_name", min_n=50)
    if not brand_df.empty:
        brand_df = brand_df.sort_values("MAE INR", ascending=False)

    frozen_df = pd.DataFrame(results_payload["frozen_test_reference"]).drop(
        columns=["model_key"], errors="ignore"
    )

    lines = [
        "# Comparación de modelos tras la corrección de datos (MarketCar)",
        "",
        f"- **Modo de ejecución**: {mode_label}",
        f"- **Dispositivo XGBoost**: `{results_payload['xgboost_device']}` "
        f"| **Hilos (`--n-jobs`)**: `{results_payload['n_jobs']}` "
        f"| **Duración total**: `{results_payload['total_elapsed_sec']:.2f} s`",
        f"- **Elección provisional (sobre validación anidada en `X_train`)**: "
        f"**{winner_info['display_name']}** (`{winner_key}`)",
        f"- **Criterio aplicado**: {selection['selection_reason']}",
        "",
        "## 1. Reproducibilidad y protocolo aplicado",
        "",
        f"- **Protocolo**: `v{results_payload['protocol_version']}` "
        "(validación anidada agrupada por `vehicle_group_id` sobre `X_train`; `X_test` no interviene en ninguna decisión).",
        f"- **Versión de limpieza**: `{ctx['cleaning_version']}` (`{clean_fp['code_sha256']}`)",
        f"- **SHA-256 CSV (`file_sha256`)**: `{ds_fp['file_sha256']}`",
        f"- **SHA-256 `X` limpio (`dataframe_sha256`)**: `{ds_fp['dataframe_sha256']}`",
        f"- **SHA-256 folds externos**: `{results_payload['outer_splits_sha256']}`",
        "",
        "## 2. Comparación en validación anidada (OOF sobre `X_train`)",
        "",
        _markdown_table(pd.DataFrame(ranking_rows)),
        "",
        "## 3. Dispersión entre folds externos",
        "",
        _markdown_table(pd.DataFrame(dispersion_rows)),
        "",
        "## 4. Comparación de objetivos (`log1p` frente a rupias directas)",
        "",
        _markdown_table(pd.DataFrame(transform_rows)),
        "",
        "## 5. Incertidumbre de la ventaja (Bootstrap pareado OOF)",
        "",
        _markdown_table(bootstrap_df) if not bootstrap_df.empty else "Solo se ejecutó un modelo.",
        "",
        f"## 6. Diagnóstico por segmentos OOF — {winner_info['display_name']}",
        "",
        "### Por tramo de precio real",
        "",
        _markdown_table(winner_bands_df),
        "",
        "### Por año de fabricación",
        "",
        _markdown_table(year_df) if not year_df.empty else "Sin datos suficientes.",
        "",
        "### Por marca (al menos 50 ejemplos en entrenamiento)",
        "",
        _markdown_table(brand_df) if not brand_df.empty else "Ninguna marca supera 50 ejemplos en esta muestra.",
        "",
        "## 7. Referencia histórica congelada en `X_test`",
        "",
        "> **Nota**: `X_test` permaneció congelado y **no intervino** en la búsqueda de "
        "hiperparámetros, la parada temprana, la elección entre `log1p` y rupias ni la "
        "selección del modelo ganador.",
        "",
        _markdown_table(frozen_df),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_retrain_comparison(
        models=args.models,
        device=args.device,
        n_jobs=args.n_jobs,
        quick=args.quick,
        n_candidates=args.n_candidates,
        output_dir=args.output_dir,
        report_output=args.report_output,
        data_path=args.data_path,
        resume=args.resume,
        fresh=args.fresh,
    )


if __name__ == "__main__":
    main()
