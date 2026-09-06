"""Funciones para limpiar y preparar los datos de vehículos."""

import ast
import re

import pandas as pd

from features import (
    boolean_text_cols,
    categorical_cols,
    convert_to_number,
    features,
    kg_cols,
    list_cols,
    mileage_cols,
    mm_cols,
)


def parse_weight(value):
    """Extrae el peso medio cuando el campo contiene uno o varios valores."""
    if pd.isna(value):
        return None

    numbers = re.findall(r"\d+(?:\.\d+)?", str(value))
    if not numbers:
        return None

    return sum(map(float, numbers)) / len(numbers)


def count_features(value):
    """Cuenta los elementos de una columna que almacena una lista serializada."""
    try:
        return len(ast.literal_eval(value))
    except (ValueError, SyntaxError, TypeError):
        return 0


def extract_first_number(value):
    """Convierte el primer número presente en un texto a ``float``."""
    if pd.isna(value):
        return None

    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Limpia el conjunto de datos y devuelve columnas de modelo actualizadas.

    Las listas de ``features.py`` no se modifican: las copias devueltas incluyen
    las columnas derivadas durante la limpieza.
    """
    df = df.copy()
    model_features = features.copy()
    model_categorical_cols = categorical_cols.copy()

    df = df[df["km_driven"] < 1_000_000].copy()
    df = df[df["dynx_totalvalue_x"] < 100_000_000].copy()

    is_electric = (
        df["fuel_type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("electric")
        .fillna(False)
    )

    # Cero no es válido para vehículos con motor térmico.
    df.loc[~is_electric & df["Displacement"].eq(0), "Displacement"] = pd.NA
    # Un eléctrico no tiene cilindros ni válvulas por cilindro.
    df.loc[is_electric, "Values per Cylinder"] = pd.NA
    # Valores mecánicamente imposibles en coches no eléctricos.
    df.loc[
        ~is_electric & df["Values per Cylinder"].gt(12),
        "Values per Cylinder",
    ] = pd.NA
    # Eliminar coches con precio mayor a 5 millones de rupias, que son casos atípicos.
    df = df[df["dynx_totalvalue_x"] <= 5_000_000].copy()

    for col in model_categorical_cols:
        cleaned = df[col].astype("string").str.lower().str.strip()
        df[col] = cleaned.mask(cleaned == "").fillna("unknown")

    for col in mm_cols:
        df[col] = pd.to_numeric(
            df[col].str.replace("mm", "", regex=False).str.strip(),
            errors="coerce",
        )

    for col in kg_cols:
        df[col] = df[col].apply(parse_weight)

    boolean_missing_cols = []
    for col in boolean_text_cols:
        df[col] = (
            df[col]
            .astype("string")
            .str.lower()
            .str.strip()
            .map({"yes": True, "no": False, "twin": True, "turbo": True})
            .astype("boolean")
        )

        missing_col = f"{col}_missing"
        df[missing_col] = df[col].isna().astype(int)
        df[col] = df[col].fillna(False).astype(int)
        boolean_missing_cols.append(missing_col)

    model_features.extend(boolean_missing_cols)

    for col in list_cols:
        df[f"{col}_count"] = df[col].apply(count_features)
    df = df.drop(columns=list_cols)

    for col in convert_to_number:
        df[col] = df[col].apply(extract_first_number)

    df["mileage_unit"] = (
        df["mileage_new"]
        .astype("string")
        .str.lower()
        .str.extract(r"(kmpl|km/kg)", expand=False)
        .fillna("unknown")
    )
    for col in mileage_cols:
        df[col] = df[col].apply(extract_first_number)

    model_categorical_cols.append("mileage_unit")
    model_features.append("mileage_unit")

    return df, model_features, model_categorical_cols
