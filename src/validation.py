"""Protocolo reproducible de validación y selección de modelos para MarketCar.

Define:
1. Revisión de duplicados por vehículo y agrupación por identificadores reales
   (sin deduplicar vehículos distintos que coincidan en características).
2. Partición de referencia (80/20, semilla 42) y folds comunes agrupados por
   vehículo sobre entrenamiento, integrados con los metadatos de reproducibilidad.
3. Esquema de validación anidada agrupada para estimar el error de generalización
   sin reutilizar el conjunto de test histórico (ya consultado en 7 modelos).
4. Métrica principal (MAE en INR) y métricas complementarias (RMSE, R², mediana
   de error relativo APE %, cobertura ±10 %, sesgo y desglose por tramos de precio).
5. Criterio pre-registrado de selección de modelos con bootstrap pareado sobre
   predicciones out-of-fold.

Ejecución desde la raíz:
    uv run python src/validation.py
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from main import DATA_PATH, MAINSTREAM_PRICE_LIMIT, RANDOM_STATE, TARGET, load_data_with_context
from modeling import IDENTIFIER_COLUMNS, _hash_pandas_data

PROTOCOL_VERSION = "1.0.0"
DEFAULT_COMMON_FOLDS = 5
DEFAULT_DIAGNOSTIC_FOLDS = 3
DEFAULT_NESTED_OUTER_FOLDS = 5
DEFAULT_NESTED_INNER_FOLDS = 3
BOOTSTRAP_ITERATIONS = 3_000
PRACTICAL_EQUIVALENCE_MAE_INR = 1_000.0

PRICE_BAND_EDGES = (0, 300_000, 600_000, 1_000_000, 2_000_000, 5_000_000)
VEHICLE_ID_COLUMNS = (
    "usedCarId",
    "usedCarSkuId",
    "used_carid",
    "ucid",
    "sid",
    "vlink",
)


def build_vehicle_groups(
    raw_df: pd.DataFrame,
    index: pd.Index | None = None,
    *,
    id_columns: tuple[str, ...] = VEHICLE_ID_COLUMNS,
) -> pd.Series:
    """Asigna un identificador de grupo por vehículo físico usando componentes conexas.

    Une anuncios que compartan cualquiera de los identificadores de vehículo
    (``usedCarId``, ``usedCarSkuId``, ``used_carid``, ``ucid``, ``sid``, ``vlink``).
    No agrupa ni deduplica vehículos distintos solo porque sus variables de entrada
    coincidan.
    """
    subset = raw_df.loc[index] if index is not None else raw_df
    n_rows = len(subset)
    if n_rows == 0:
        return pd.Series(dtype="string", index=subset.index, name="vehicle_group_id")

    parent = list(range(n_rows))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            if ri < rj:
                parent[rj] = ri
            else:
                parent[ri] = rj

    available_cols = [col for col in id_columns if col in subset.columns]
    for col in available_cols:
        values = subset[col]
        first_seen: dict[str, int] = {}
        for pos, val in enumerate(values):
            if pd.isna(val):
                continue
            key = str(val).strip()
            if not key or key.lower() == "nan":
                continue
            if key in first_seen:
                union(pos, first_seen[key])
            else:
                first_seen[key] = pos

    roots = [find(i) for i in range(n_rows)]
    group_labels = [f"veh_group_{subset.index[r]}" for r in roots]
    return pd.Series(group_labels, index=subset.index, dtype="string", name="vehicle_group_id")


def audit_vehicle_duplicates(
    raw_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict[str, Any]:
    """Audita duplicados por identificador de vehículo y por características."""
    combined_index = X_train.index.union(X_test.index)
    subset = raw_df.loc[combined_index]
    groups = build_vehicle_groups(raw_df, combined_index)

    train_groups = set(groups.loc[X_train.index])
    test_groups = set(groups.loc[X_test.index])
    shared_groups = train_groups & test_groups

    id_summary: list[dict[str, Any]] = []
    for col in VEHICLE_ID_COLUMNS:
        if col not in subset.columns:
            continue
        series = subset[col].dropna().astype(str).str.strip()
        train_vals = set(raw_df.loc[X_train.index, col].dropna().astype(str).str.strip())
        test_vals = raw_df.loc[X_test.index, col].dropna().astype(str).str.strip()
        id_summary.append({
            "identifier": col,
            "non_null": int(len(series)),
            "unique_values": int(series.nunique()),
            "duplicate_rows": int(series.duplicated().sum()),
            "test_in_train": int(test_vals.isin(train_vals).sum()),
        })

    group_counts = groups.value_counts()
    multi_row_groups = group_counts[group_counts > 1]
    multi_row_indices = groups[groups.isin(multi_row_groups.index)].index.tolist()

    X_all = pd.concat([X_train, X_test])
    feature_hashes = pd.util.hash_pandas_object(X_all, index=False)
    train_feat_hashes = set(pd.util.hash_pandas_object(X_train, index=False))
    test_feat_hashes = pd.util.hash_pandas_object(X_test, index=False)

    return {
        "total_rows": int(len(combined_index)),
        "unique_vehicle_groups": int(groups.nunique()),
        "multi_listing_groups": int(len(multi_row_groups)),
        "multi_listing_rows": int(len(multi_row_indices)),
        "multi_listing_indices": [int(i) for i in multi_row_indices],
        "cross_split_vehicle_groups": int(len(shared_groups)),
        "exact_duplicate_raw_rows": int(subset.duplicated().sum()),
        "duplicate_feature_rows_total": int(feature_hashes.duplicated().sum()),
        "test_rows_matching_train_features": int(test_feat_hashes.isin(train_feat_hashes).sum()),
        "identifier_summary": id_summary,
    }


def build_grouped_kfold_splits(
    index: pd.Index,
    groups: pd.Series,
    *,
    n_splits: int = DEFAULT_COMMON_FOLDS,
    random_state: int = RANDOM_STATE,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Genera folds barajados y balanceados manteniendo juntos los anuncios del mismo vehículo.

    Devuelve una lista de tuplas ``(train_positions, val_positions)`` con posiciones
    enteras relativas a ``index`` (compatible con ``cross_val_score`` y ``GridSearchCV``).
    """
    if n_splits < 2:
        raise ValueError("n_splits debe ser al menos 2.")
    if len(index) < n_splits:
        raise ValueError("El número de filas no puede ser menor que n_splits.")

    aligned_groups = groups.loc[index].astype(str)
    group_to_positions: dict[str, list[int]] = {}
    for pos, grp in enumerate(aligned_groups):
        group_to_positions.setdefault(grp, []).append(pos)

    unique_groups = list(group_to_positions.keys())
    if len(unique_groups) < n_splits:
        raise ValueError("El número de grupos únicos debe ser al menos n_splits.")

    rng = np.random.default_rng(random_state)
    shuffled_order = rng.permutation(len(unique_groups))

    # Primero asignamos grupos con múltiples anuncios al fold con menos filas,
    # y después repartimos los grupos unitarios de forma equilibrada.
    multi_groups = [
        unique_groups[i] for i in shuffled_order if len(group_to_positions[unique_groups[i]]) > 1
    ]
    single_groups = [
        unique_groups[i] for i in shuffled_order if len(group_to_positions[unique_groups[i]]) == 1
    ]

    fold_positions: list[list[int]] = [[] for _ in range(n_splits)]
    fold_sizes = np.zeros(n_splits, dtype=int)

    for grp in multi_groups:
        target_fold = int(np.argmin(fold_sizes))
        positions = group_to_positions[grp]
        fold_positions[target_fold].extend(positions)
        fold_sizes[target_fold] += len(positions)

    for grp in single_groups:
        target_fold = int(np.argmin(fold_sizes))
        positions = group_to_positions[grp]
        fold_positions[target_fold].extend(positions)
        fold_sizes[target_fold] += len(positions)

    all_positions = np.arange(len(index), dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_splits):
        val_pos = np.array(sorted(fold_positions[fold_idx]), dtype=int)
        train_mask = np.ones(len(index), dtype=bool)
        train_mask[val_pos] = False
        train_pos = all_positions[train_mask]
        splits.append((train_pos, val_pos))

    return splits


def hash_cv_splits(
    index: pd.Index,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> str:
    """Calcula una huella SHA-256 reproducible de los folds sobre los índices reales."""
    serializable = [
        {
            "fold": fold_idx,
            "train_indices": [int(index[i]) for i in train_pos],
            "val_indices": [int(index[i]) for i in val_pos],
        }
        for fold_idx, (train_pos, val_pos) in enumerate(splits)
    ]
    payload = json.dumps(serializable, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_nested_cv_splits(
    index: pd.Index,
    groups: pd.Series,
    *,
    outer_splits: int = DEFAULT_NESTED_OUTER_FOLDS,
    inner_splits: int = DEFAULT_NESTED_INNER_FOLDS,
    random_state: int = RANDOM_STATE,
) -> list[dict[str, Any]]:
    """Construye particiones de validación anidada agrupadas por vehículo."""
    outer = build_grouped_kfold_splits(
        index,
        groups,
        n_splits=outer_splits,
        random_state=random_state,
    )
    nested: list[dict[str, Any]] = []
    for outer_fold, (outer_train_pos, outer_val_pos) in enumerate(outer):
        outer_train_index = index[outer_train_pos]
        inner = build_grouped_kfold_splits(
            outer_train_index,
            groups.loc[outer_train_index],
            n_splits=inner_splits,
            random_state=random_state + outer_fold + 1,
        )
        nested.append({
            "outer_fold": outer_fold,
            "outer_train_positions": outer_train_pos,
            "outer_val_positions": outer_val_pos,
            "inner_splits": inner,
            "inner_splits_sha256": hash_cv_splits(outer_train_index, inner),
        })
    return nested


def calculate_protocol_metrics(y_true, predictions) -> dict[str, float]:
    """Calcula todas las métricas oficiales del protocolo en rupias (INR)."""
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    if actual.shape != pred.shape:
        raise ValueError("y_true y predictions deben tener la misma dimensión.")
    if len(actual) == 0:
        raise ValueError("No se pueden calcular métricas sobre un conjunto vacío.")
    if np.any(actual <= 0):
        raise ValueError("Los precios reales deben ser estrictamente positivos.")

    error = pred - actual
    abs_error = np.abs(error)
    ape = abs_error / actual

    ss_res = float(np.sum(np.square(error)))
    ss_tot = float(np.sum(np.square(actual - np.mean(actual))))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "r2": float(r2),
        "median_ape_pct": float(np.median(ape) * 100.0),
        "within_10pct_pct": float(np.mean(ape <= 0.10) * 100.0),
        "bias_inr": float(np.mean(error)),
    }


def evaluate_by_price_band(
    y_true,
    predictions,
    *,
    band_edges: tuple[int, ...] = PRICE_BAND_EDGES,
) -> pd.DataFrame:
    """Desglosa las métricas del protocolo por tramos de precio real."""
    actual = pd.Series(np.asarray(y_true, dtype=float))
    pred = pd.Series(np.asarray(predictions, dtype=float))
    bands = pd.cut(actual, list(band_edges))

    rows: list[dict[str, Any]] = []
    for label, group_actual in actual.groupby(bands, observed=False):
        if len(group_actual) == 0:
            rows.append({
                "price_band": str(label),
                "n": 0,
                "mae": np.nan,
                "rmse": np.nan,
                "median_ape_pct": np.nan,
                "within_10pct_pct": np.nan,
                "bias_inr": np.nan,
            })
            continue
        group_pred = pred.loc[group_actual.index]
        metrics = calculate_protocol_metrics(group_actual, group_pred)
        rows.append({
            "price_band": str(label),
            "n": int(len(group_actual)),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "median_ape_pct": metrics["median_ape_pct"],
            "within_10pct_pct": metrics["within_10pct_pct"],
            "bias_inr": metrics["bias_inr"],
        })

    return pd.DataFrame(rows)


def paired_bootstrap_mae(
    y_true,
    first_predictions,
    second_predictions,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Bootstrap pareado de la diferencia de MAE: MAE(first) - MAE(second)."""
    actual = np.asarray(y_true, dtype=float)
    first_err = np.abs(np.asarray(first_predictions, dtype=float) - actual)
    second_err = np.abs(np.asarray(second_predictions, dtype=float) - actual)
    diff = first_err - second_err

    rng = np.random.default_rng(random_state)
    n = len(diff)
    boot_means = np.empty(iterations, dtype=float)
    for idx in range(iterations):
        sample_indices = rng.integers(0, n, size=n)
        boot_means[idx] = diff[sample_indices].mean()

    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
    return {
        "delta_mae_inr": float(diff.mean()),
        "ci95_low_inr": float(ci_low),
        "ci95_high_inr": float(ci_high),
        "conclusive": bool(ci_high < 0.0 or ci_low > 0.0),
    }


def select_best_candidate(
    candidates: list[dict[str, Any]],
    y_true,
    *,
    practical_margin_inr: float = PRACTICAL_EQUIVALENCE_MAE_INR,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Aplica el criterio pre-registrado de selección sobre predicciones out-of-fold.

    Cada elemento de ``candidates`` debe incluir:
    - ``name``: nombre del modelo/configuración.
    - ``oof_predictions``: predicciones out-of-fold sobre el conjunto de validación.
    - ``fold_maes``: secuencia con el MAE de cada fold.
    - ``complexity_rank`` (opcional): menor valor indica modelo más simple/rápido.

    Regla de decisión:
    1. Ordenar candidatos por menor MAE medio de validación (métrica principal).
    2. Comparar el mejor candidato en MAE frente a los demás mediante bootstrap
       pareado (IC95 % con 3.000 remuestreos).
    3. Si la ventaja frente a otro candidato no es estadísticamente concluyente
       (el IC95 % incluye 0) y la diferencia de MAE es menor que
       ``practical_margin_inr`` (₹1.000), desempatar por:
       a) Menor dispersión entre folds (SD de MAE) y menor RMSE global.
       b) Menor mediana de APE % y mayor cobertura ±10 %.
       c) Menor MAE en el tramo alto (₹2.000.000–₹5.000.000).
       d) Menor ``complexity_rank``.
    """
    if not candidates:
        raise ValueError("Debe proporcionarse al menos un candidato.")

    actual = np.asarray(y_true, dtype=float)
    evaluated: list[dict[str, Any]] = []
    for item in candidates:
        oof = np.asarray(item["oof_predictions"], dtype=float)
        metrics = calculate_protocol_metrics(actual, oof)
        bands = evaluate_by_price_band(actual, oof).set_index("price_band")
        high_band_mae = (
            float(bands.loc["(2000000, 5000000]", "mae"])
            if "(2000000, 5000000]" in bands.index and pd.notna(bands.loc["(2000000, 5000000]", "mae"])
            else metrics["mae"]
        )
        fold_maes = np.asarray(item.get("fold_maes", [metrics["mae"]]), dtype=float)
        fold_sd = float(fold_maes.std(ddof=1)) if len(fold_maes) > 1 else 0.0

        evaluated.append({
            "name": str(item["name"]),
            "oof_predictions": oof,
            "metrics": metrics,
            "fold_mean_mae": float(fold_maes.mean()),
            "fold_sd_mae": fold_sd,
            "high_band_mae": high_band_mae,
            "complexity_rank": int(item.get("complexity_rank", 100)),
        })

    evaluated.sort(key=lambda c: c["metrics"]["mae"])
    leader = evaluated[0]

    contenders: list[dict[str, Any]] = [leader]
    comparisons: list[dict[str, Any]] = []
    for challenger in evaluated[1:]:
        boot = paired_bootstrap_mae(
            actual,
            leader["oof_predictions"],
            challenger["oof_predictions"],
            random_state=random_state,
        )
        comparisons.append({
            "leader": leader["name"],
            "challenger": challenger["name"],
            **boot,
        })
        within_margin = abs(boot["delta_mae_inr"]) <= practical_margin_inr
        if (not boot["conclusive"]) and within_margin:
            contenders.append(challenger)

    if len(contenders) == 1:
        winner = leader
        reason = "Menor MAE en validación sin empate técnico dentro del margen práctico."
    else:
        contenders.sort(
            key=lambda c: (
                round(c["fold_sd_mae"] + c["metrics"]["rmse"], 3),
                round(c["metrics"]["median_ape_pct"] - c["metrics"]["within_10pct_pct"], 4),
                round(c["high_band_mae"], 3),
                c["complexity_rank"],
                c["metrics"]["mae"],
            )
        )
        winner = contenders[0]
        reason = (
            "Empate técnico en MAE dentro del intervalo bootstrap / margen práctico; "
            "seleccionado por criterios secundarios pre-registrados (estabilidad/RMSE, "
            "error relativo/cobertura ±10 %, tramo alto y simplicidad)."
        )

    return {
        "selected_model": winner["name"],
        "selection_reason": reason,
        "ranking": [
            {
                "name": c["name"],
                "mae": c["metrics"]["mae"],
                "rmse": c["metrics"]["rmse"],
                "r2": c["metrics"]["r2"],
                "median_ape_pct": c["metrics"]["median_ape_pct"],
                "within_10pct_pct": c["metrics"]["within_10pct_pct"],
                "bias_inr": c["metrics"]["bias_inr"],
                "fold_sd_mae": c["fold_sd_mae"],
                "high_band_mae": c["high_band_mae"],
                "complexity_rank": c["complexity_rank"],
            }
            for c in evaluated
        ],
        "paired_bootstrap": comparisons,
    }


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


def build_validation_protocol_summary(data_path: Path = DATA_PATH) -> dict[str, Any]:
    """Ejecuta el protocolo sobre los datos actuales y devuelve sus metadatos y huellas."""
    raw_df = pd.read_csv(data_path, low_memory=False)
    (
        features,
        categorical_features,
        numeric_features,
        X_train,
        X_test,
        y_train,
        y_test,
        reproducibility_context,
    ) = load_data_with_context(data_path)

    duplicate_audit = audit_vehicle_duplicates(raw_df, X_train, X_test)
    train_groups = build_vehicle_groups(raw_df, X_train.index)

    common_5_splits = build_grouped_kfold_splits(
        X_train.index,
        train_groups,
        n_splits=DEFAULT_COMMON_FOLDS,
        random_state=RANDOM_STATE,
    )
    common_3_splits = build_grouped_kfold_splits(
        X_train.index,
        train_groups,
        n_splits=DEFAULT_DIAGNOSTIC_FOLDS,
        random_state=RANDOM_STATE,
    )
    nested_splits = build_nested_cv_splits(
        X_train.index,
        train_groups,
        outer_splits=DEFAULT_NESTED_OUTER_FOLDS,
        inner_splits=DEFAULT_NESTED_INNER_FOLDS,
        random_state=RANDOM_STATE,
    )

    fold_5_rows: list[dict[str, Any]] = []
    for fold_idx, (tr_pos, val_pos) in enumerate(common_5_splits, start=1):
        val_index = X_train.index[val_pos]
        val_y = y_train.loc[val_index]
        fold_5_rows.append({
            "Fold": fold_idx,
            "Filas train": int(len(tr_pos)),
            "Filas val": int(len(val_pos)),
            "Grupos vehículo val": int(train_groups.loc[val_index].nunique()),
            "Media precio val INR": round(float(val_y.mean()), 2),
            "Mediana precio val INR": round(float(val_y.median()), 2),
        })

    return {
        "protocol_version": PROTOCOL_VERSION,
        "reproducibility_context": reproducibility_context,
        "duplicate_audit": duplicate_audit,
        "common_folds_5": {
            "n_splits": DEFAULT_COMMON_FOLDS,
            "random_state": RANDOM_STATE,
            "splits_sha256": hash_cv_splits(X_train.index, common_5_splits),
            "fold_summary": fold_5_rows,
        },
        "common_folds_3": {
            "n_splits": DEFAULT_DIAGNOSTIC_FOLDS,
            "random_state": RANDOM_STATE,
            "splits_sha256": hash_cv_splits(X_train.index, common_3_splits),
        },
        "nested_cv": {
            "outer_splits": DEFAULT_NESTED_OUTER_FOLDS,
            "inner_splits": DEFAULT_NESTED_INNER_FOLDS,
            "random_state": RANDOM_STATE,
            "outer_splits_sha256": hash_cv_splits(
                X_train.index,
                [(d["outer_train_positions"], d["outer_val_positions"]) for d in nested_splits],
            ),
            "inner_splits_sha256": [d["inner_splits_sha256"] for d in nested_splits],
        },
        "feature_counts": {
            "total": len(features),
            "numeric": len(numeric_features),
            "categorical": len(categorical_features),
        },
    }


def render_protocol_markdown(summary: dict[str, Any]) -> str:
    """Genera el documento markdown ``reports/protocolo_validacion.md``."""
    ctx = summary["reproducibility_context"]
    ds_fp = ctx["dataset_fingerprint"]
    clean_fp = ctx["cleaning_fingerprint"]
    part = ctx["partition"]
    dup = summary["duplicate_audit"]
    folds_5 = summary["common_folds_5"]
    folds_3 = summary["common_folds_3"]
    nested = summary["nested_cv"]

    id_table = _markdown_table(
        pd.DataFrame(dup["identifier_summary"]).rename(columns={
            "identifier": "Identificador",
            "non_null": "No nulos",
            "unique_values": "Valores únicos",
            "duplicate_rows": "Filas repetidas",
            "test_in_train": "Filas test presentes en train",
        })
    )
    folds_table = _markdown_table(pd.DataFrame(folds_5["fold_summary"]))

    price_bands_df = pd.DataFrame([
        {"Tramo de precio real (INR)": "(0, 300000]", "Segmento": "Económico / entrada (≤ ₹3 Lakh)"},
        {"Tramo de precio real (INR)": "(300000, 600000]", "Segmento": "Compacto generalista (₹3–6 Lakh)"},
        {"Tramo de precio real (INR)": "(600000, 1000000]", "Segmento": "Medio (₹6–10 Lakh)"},
        {"Tramo de precio real (INR)": "(1000000, 2000000]", "Segmento": "Medio-alto / SUV (₹10–20 Lakh)"},
        {"Tramo de precio real (INR)": "(2000000, 5000000]", "Segmento": "Alto dentro del alcance (₹20–50 Lakh)"},
    ])

    lines = [
        "# Protocolo de validación y selección de modelos de MarketCar",
        "",
        f"Versión del protocolo: `{summary['protocol_version']}`. "
        "Este documento fija de antemano las particiones, el tratamiento de duplicados "
        "por vehículo, las métricas de evaluación, el esquema de validación anidada y la "
        "regla de decisión antes de reentrenar y comparar modelos (tarea #4).",
        "",
        "## 1. Huella de reproducibilidad de datos y partición base",
        "",
        "El protocolo reutiliza los metadatos de reproducibilidad introducidos en `src/modeling.py`:",
        "",
        f"- **Archivo de origen**: `{Path(ds_fp['file_path']).as_posix() if ds_fp['file_path'] else '—'}`",
        f"- **SHA-256 del CSV bruto (`file_sha256`)**: `{ds_fp['file_sha256']}`",
        f"- **Versión de limpieza (`cleaning_version`)**: `{ctx['cleaning_version']}` "
        f"(código SHA-256: `{clean_fp['code_sha256']}`)",
        f"- **SHA-256 de la matriz limpia `X` (`dataframe_sha256`)**: `{ds_fp['dataframe_sha256']}`",
        f"- **SHA-256 del objetivo `y` (`target_sha256`)**: `{ds_fp['target_sha256']}`",
        f"- **Población generalista (`{ctx['target']} <= {ctx['mainstream_price_limit']:,}`)**: "
        f"`{ds_fp['rows']:,}` anuncios (`{part['train_rows']:,}` entrenamiento / `{part['test_rows']:,}` test)",
        f"- **Semilla base (`random_state`)**: `{part['random_state']}`",
        f"- **SHA-256 de la partición 80/20 (`partition_sha256`)**: `{part['partition_sha256']}`",
        f"- **Variables de entrada**: `{summary['feature_counts']['total']}` "
        f"(`{summary['feature_counts']['numeric']}` numéricas y "
        f"`{summary['feature_counts']['categorical']}` categóricas)",
        "",
        "## 2. Auditoría de duplicados por vehículo y agrupación",
        "",
        _markdown_table(pd.DataFrame([
            {"Concepto": "Filas generalistas evaluadas", "Recuento": dup["total_rows"]},
            {"Concepto": "Filas exactamente duplicadas en el CSV bruto", "Recuento": dup["exact_duplicate_raw_rows"]},
            {"Concepto": "Grupos de vehículo únicos (componentes conexas de IDs)", "Recuento": dup["unique_vehicle_groups"]},
            {"Concepto": "Grupos con múltiples anuncios del mismo vehículo (`ucid`/`sid`)", "Recuento": dup["multi_listing_groups"]},
            {"Concepto": "Filas pertenecientes a grupos con múltiples anuncios", "Recuento": dup["multi_listing_rows"]},
            {"Concepto": "Grupos de vehículo compartidos entre train y test", "Recuento": dup["cross_split_vehicle_groups"]},
            {"Concepto": "Filas test con las mismas características de entrada `X` que train", "Recuento": dup["test_rows_matching_train_features"]},
        ])),
        "",
        "### Detalle por identificador de vehículo",
        "",
        id_table,
        "",
        "### Regla sobre duplicados",
        "",
        "1. **Agrupación por vehículo (`build_vehicle_groups`)**: existen 3 pares de anuncios "
        f"(6 filas: índices `{dup['multi_listing_indices']}`) que comparten `ucid` y `sid` "
        "(reanuncios del mismo vehículo en el mismo vendedor con ligeras variaciones de precio "
        "o +1 km). En la partición 80/20 (`random_state=42`) las 6 filas caen en entrenamiento "
        "(`0` fugas hacia test), pero un `KFold` estándar sin agrupar las separa en folds distintos. "
        "Por ello, todas las particiones de validación cruzada agrupan por `vehicle_group_id`.",
        "2. **No deduplicación por coincidencia de características**: las 27 filas de test que "
        "coinciden en variables de entrada `X` con alguna fila de entrenamiento tienen `usedCarId`, "
        "`usedCarSkuId`, `ucid` y `vlink` distintos. Representan coches diferentes del mismo "
        "modelo, año, ciudad y kilometraje redondeado, por lo que **no se eliminan**.",
        "",
        "## 3. Folds comunes de entrenamiento y validación anidada",
        "",
        "### 3.1. Alcance del conjunto de test actual y necesidad de validación anidada",
        "",
        "El conjunto `X_test` (`7.496` anuncios) ya se consultó para comparar siete familias de "
        "modelos en el diagnóstico inicial. Cambiar la semilla aleatoria sobre el mismo CSV no "
        "crea un conjunto de test virgen, porque las decisiones de preprocesamiento y familias "
        "candidatas ya utilizaron esa muestra.",
        "",
        "- **Mientras no se incorporen anuncios nuevos independientes**:",
        "  - La **selección de hiperparámetros, comparación de objetivos (`log1p` vs rupias)** y "
        "**elección de modelo** se realizan **exclusivamente sobre `X_train` (`29.980` filas)** "
        "usando los folds comunes agrupados y validación anidada.",
        "  - `X_test` se mantiene congelado únicamente como **referencia histórica de regresión** "
        "frente a los artefactos `v1`, declarando explícitamente que no constituye un test "
        "independiente virgen.",
        "",
        "### 3.2. Folds comunes de entrenamiento (comparación directa de candidatos)",
        "",
        "- **Folds principales de selección (`5` folds agrupados por `vehicle_group_id`)**:",
        f"  - Semilla: `{folds_5['random_state']}`",
        f"  - Huella SHA-256 (`splits_sha256`): `{folds_5['splits_sha256']}`",
        "- **Folds diagnósticos rápidos (`3` folds agrupados por `vehicle_group_id`)**:",
        f"  - Semilla: `{folds_3['random_state']}`",
        f"  - Huella SHA-256 (`splits_sha256`): `{folds_3['splits_sha256']}`",
        "",
        folds_table,
        "",
        "### 3.3. Protocolo de validación anidada (`build_nested_cv_splits`)",
        "",
        "Cuando no se dispone de datos externos nuevos, la estimación no sesgada del rendimiento "
        "de un procedimiento de ajuste se obtiene mediante validación anidada sobre `X_train`:",
        "",
        f"- **Bucle externo (`{nested['outer_splits']}` folds agrupados por `vehicle_group_id`)**: "
        f"semilla `{nested['random_state']}`, huella SHA-256 `{nested['outer_splits_sha256']}`. "
        "Mide el error fuera de muestra de cada familia/procedimiento sin intervenir en el ajuste.",
        f"- **Bucle interno (`{nested['inner_splits']}` folds agrupados por `vehicle_group_id`)**: "
        "dentro de cada fold externo, selecciona hiperparámetros y compara transformación del "
        "objetivo (`log1p` frente a rupias directas) minimizando MAE en rupias.",
        "- **Huellas SHA-256 de los bucles internos por fold externo**:",
        *[
            f"  - Fold externo {idx + 1}: `{inner_hash}`"
            for idx, inner_hash in enumerate(nested["inner_splits_sha256"])
        ],
        "",
        "## 4. Métricas de evaluación",
        "",
        "Todas las predicciones se evalúan en la escala monetaria real (**rupias indias, INR**), "
        "aplicando `expm1` cuando el modelo entrene sobre `log1p(precio)`:",
        "",
        "1. **Métrica principal**: **MAE (INR)** — error absoluto medio en rupias.",
        "2. **Métricas complementarias globales**:",
        "   - **RMSE (INR)** — raíz del error cuadrático medio (sensible a errores extremos).",
        "   - **R²** — proporción de varianza explicada en rupias.",
        "   - **Mediana APE (%)** — mediana de `|predicción - real| / real * 100`.",
        "   - **Cobertura ±10 % (%)** — porcentaje de anuncios con error relativo `<= 10 %`.",
        "   - **Sesgo medio (INR)** — media de `predicción - real` (negativo indica infravaloración).",
        "3. **Desglose obligatorio por tramos de precio real (`evaluate_by_price_band`)**:",
        "",
        _markdown_table(price_bands_df),
        "",
        "## 5. Criterio de selección del modelo fijado de antemano",
        "",
        "La función `select_best_candidate` en `src/validation.py` codifica la regla de decisión "
        "que se aplicará en la tarea #4 sobre las predicciones *out-of-fold* de validación:",
        "",
        "1. **Ordenación primaria por MAE (INR)**: el candidato con menor MAE fuera de fold en "
        "validación es el líder inicial.",
        "2. **Contraste de incertidumbre por bootstrap pareado (`paired_bootstrap_mae`)**: se "
        f"calcula el intervalo de confianza al 95 % (`{BOOTSTRAP_ITERATIONS:,}` remuestreos "
        f"pareados por anuncio, semilla `{RANDOM_STATE}`) de `MAE(líder) - MAE(candidato)`.",
        "3. **Regla de empate técnico y parsimonia**: si el IC95 % incluye `0` y la diferencia "
        f"absoluta de MAE es `<= ₹{PRACTICAL_EQUIVALENCE_MAE_INR:,.0f}`, los candidatos se "
        "consideran en empate técnico y se desempata en este orden estricto:",
        "   1. Menor suma de dispersión entre folds (`SD` del MAE por fold) y **RMSE (INR)**.",
        "   2. Menor **Mediana APE (%)** y mayor **cobertura ±10 % (%)**.",
        "   3. Menor **MAE (INR)** en el tramo alto `(2000000, 5000000]`.",
        "   4. Menor complejidad operativa (`complexity_rank`: Ridge < LightGBM/XGBoost/CatBoost < bosques masivos).",
        "4. **Prohibición de selección sobre `X_test`**: ningún hiperparámetro ni elección entre "
        "familias u objetivos (`log1p` vs rupias) puede decidirse por su resultado en `X_test`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_PATH,
        help="Ruta al CSV de entrada (por defecto: data/cars_details_merges.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/protocolo_validacion.md"),
        help="Ruta del informe markdown del protocolo.",
    )
    args = parser.parse_args()

    summary = build_validation_protocol_summary(args.data_path)
    markdown = render_protocol_markdown(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Protocolo generado en: {args.output}")


if __name__ == "__main__":
    main()
