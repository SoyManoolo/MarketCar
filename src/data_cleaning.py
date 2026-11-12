"""Funciones para limpiar y preparar los datos de vehículos."""

import ast
import re

import numpy as np
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

CLEANING_VERSION = "2.0.0"

# Factores físicos estándar de conversión de unidades.
KGM_TO_NM = 9.80665
KW_TO_BHP = 1.34102209

# Rangos físicamente plausibles para turismos y SUVs del mercado evaluado.
# Los valores fuera de rango se marcan como implausibles y pasan a nulo (NA)
# para ser imputados con indicador de ausencia, sin borrar el anuncio.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "model_year": (1980.0, 2030.0),
    "km_driven": (0.0, 1_000_000.0),
    "Displacement": (50.0, 10_000.0),
    "Max Power": (10.0, 1_500.0),
    "Max Torque": (20.0, 2_500.0),
    "No of Cylinder": (1.0, 16.0),
    "Values per Cylinder": (1.0, 12.0),
    "Length": (1_500.0, 7_000.0),
    "Width": (1_000.0, 3_000.0),
    "Height": (900.0, 3_000.0),
    "Wheel Base": (1_200.0, 5_000.0),
    "Kerb Weight": (300.0, 6_000.0),
    "seating_capacity_new": (2.0, 15.0),
    "Top Speed": (40.0, 450.0),
    "Acceleration": (1.5, 60.0),
    "No Door Numbers": (2.0, 6.0),
    "Cargo Volumn": (10.0, 3_000.0),
    "mileage_new": (1.0, 100.0),
    "Alloy Wheel Size": (10.0, 26.0),
}

_THOUSANDS_COMMA_RE = re.compile(r"(?<![.\d])(\d{1,3}(?:,\d{3})+)(?!\d)")
_LEADING_NEGATIVE_RE = re.compile(r"^\s*-\s*(\d+(?:\.\d+)?)")


def normalize_thousands_separators(value) -> str:
    """Elimina comas de miles (p. ej. ``3,690`` -> ``3690``) conservando decimales."""
    if pd.isna(value):
        return ""
    return _THOUSANDS_COMMA_RE.sub(
        lambda match: match.group(1).replace(",", ""),
        str(value),
    )


def parse_dimension_mm(value) -> float | None:
    """Extrae una dimensión en milímetros corrigiendo separadores de miles."""
    if pd.isna(value):
        return None

    text = normalize_thousands_separators(value).strip()
    if not text:
        return None

    neg_match = _LEADING_NEGATIVE_RE.match(text)
    if neg_match:
        return -float(neg_match.group(1))

    cleaned = re.sub(r"(?i)mm\b", "", text).strip()
    try:
        return float(cleaned)
    except ValueError:
        numbers = re.findall(r"\d+(?:\.\d+)?", cleaned)
        if not numbers:
            return None
        return sum(map(float, numbers)) / len(numbers)


def parse_weight(value) -> float | None:
    """Extrae el peso medio en kg corrigiendo separadores de miles."""
    if pd.isna(value):
        return None

    text = normalize_thousands_separators(value).strip()
    if not text:
        return None

    neg_match = _LEADING_NEGATIVE_RE.match(text)
    if neg_match:
        return -float(neg_match.group(1))

    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if not numbers:
        return None

    return sum(map(float, numbers)) / len(numbers)


def count_features(value):
    """Cuenta los elementos de una columna que almacena una lista serializada."""
    try:
        return len(ast.literal_eval(value))
    except (ValueError, SyntaxError, TypeError):
        return 0


def extract_first_number(value) -> float | None:
    """Convierte el primer número presente en un texto a ``float``."""
    if pd.isna(value):
        return None

    text = normalize_thousands_separators(value).strip()
    if not text:
        return None

    neg_match = _LEADING_NEGATIVE_RE.match(text)
    if neg_match:
        return -float(neg_match.group(1))

    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else None


def parse_torque(value) -> float | None:
    """Normaliza el par máximo a Newton-metro (Nm).

    Soporta formatos con ``Nm`` directo, ``kgm`` junto al valor (``19.4 kgm @ 2000 rpm``),
    textos mixtos con ambas unidades y el formato de CarDekho ``11.5@ 4,500(kgm@ rpm)``.
    """
    if pd.isna(value):
        return None

    text = normalize_thousands_separators(value).strip()
    if not text:
        return None

    neg_match = _LEADING_NEGATIVE_RE.match(text)
    if neg_match:
        return -float(neg_match.group(1))

    nm_match = re.search(
        r"(\d+(?:\.\d+)?)\s*n[\s.-]*m(?![a-zA-Z])",
        text,
        flags=re.IGNORECASE,
    )
    if nm_match:
        return float(nm_match.group(1))

    kgm_match = re.search(
        r"(\d+(?:\.\d+)?)\s*kg(?:f)?(?:[\s.-]*m)?(?![a-zA-Z])",
        text,
        flags=re.IGNORECASE,
    )
    if kgm_match:
        return round(float(kgm_match.group(1)) * KGM_TO_NM, 4)

    first_number = extract_first_number(text)
    if first_number is None:
        return None

    if re.search(r"kg(?:f)?(?:[\s.-]*m)?(?![a-zA-Z])", text, flags=re.IGNORECASE):
        return round(first_number * KGM_TO_NM, 4)

    return first_number


def parse_power(value) -> float | None:
    """Normaliza la potencia máxima a brake horsepower (bhp).

    Si el texto contiene un valor explícito en ``bhp`` (incluso junto a ``kW``,
    como ``88.50bhp (66kW)@ 6000rpm`` o ``66kW (88.50bhp)``), conserva el valor
    en ``bhp``. Si solo expresa la potencia en ``kW``, la convierte a ``bhp``.
    """
    if pd.isna(value):
        return None

    text = normalize_thousands_separators(value).strip()
    if not text:
        return None

    neg_match = _LEADING_NEGATIVE_RE.match(text)
    if neg_match:
        return -float(neg_match.group(1))

    bhp_match = re.search(
        r"(\d+(?:\.\d+)?)\s*bhp(?![a-zA-Z])",
        text,
        flags=re.IGNORECASE,
    )
    if bhp_match:
        return float(bhp_match.group(1))

    kw_match = re.search(
        r"(\d+(?:\.\d+)?)\s*kw(?![a-zA-Z])",
        text,
        flags=re.IGNORECASE,
    )
    if kw_match:
        return round(float(kw_match.group(1)) * KW_TO_BHP, 4)

    first_number = extract_first_number(text)
    if first_number is None:
        return None

    if re.search(r"(?<![a-zA-Z])kw(?![a-zA-Z])", text, flags=re.IGNORECASE):
        return round(first_number * KW_TO_BHP, 4)

    return first_number


def mark_implausible_values(
    df: pd.DataFrame,
    is_electric: pd.Series | None = None,
) -> pd.DataFrame:
    """Marca valores físicamente implausibles y los reemplaza por nulo sin borrar filas.

    Para cada columna de ``PLAUSIBLE_RANGES`` presente en ``df``, crea una columna
    booleana ``<col>_implausible`` y resume en ``has_implausible_values`` e
    ``implausible_columns`` qué anuncios tenían datos fuera de rango físico.
    """
    df = df.copy()
    if is_electric is None:
        if "fuel_type" in df.columns:
            is_electric = (
                df["fuel_type"]
                .astype("string")
                .str.strip()
                .str.lower()
                .eq("electric")
                .fillna(False)
            )
        else:
            is_electric = pd.Series(False, index=df.index, dtype=bool)

    implausible_flags: dict[str, pd.Series] = {}

    for col, (min_val, max_val) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue

        numeric_series = pd.to_numeric(df[col], errors="coerce")
        df[col] = numeric_series

        if col == "Displacement":
            # En eléctricos 0 cc es válido; valores negativos o > 0 no aplican.
            electric_invalid = is_electric & numeric_series.notna() & ~numeric_series.eq(0)
            thermal_invalid = (
                ~is_electric
                & numeric_series.notna()
                & ((numeric_series < min_val) | (numeric_series > max_val))
            )
            mask = electric_invalid | thermal_invalid
        elif col in {"Values per Cylinder", "No of Cylinder"}:
            # Un eléctrico no tiene cilindros ni válvulas por cilindro.
            if col == "Values per Cylinder":
                df.loc[is_electric, col] = np.nan
                numeric_series = pd.to_numeric(df[col], errors="coerce")
            mask = (
                ~is_electric
                & numeric_series.notna()
                & ((numeric_series < min_val) | (numeric_series > max_val))
            )
        else:
            mask = numeric_series.notna() & (
                (numeric_series < min_val) | (numeric_series > max_val)
            )

        mask = mask.fillna(False).astype(bool)
        df[f"{col}_implausible"] = mask
        implausible_flags[col] = mask
        if mask.any():
            df.loc[mask, col] = np.nan

    if implausible_flags:
        flag_Matrix = pd.DataFrame(implausible_flags, index=df.index)
        df["has_implausible_values"] = flag_Matrix.any(axis=1)
        df["implausible_columns"] = flag_Matrix.apply(
            lambda row: ",".join(col for col, val in row.items() if val),
            axis=1,
        )
    else:
        df["has_implausible_values"] = False
        df["implausible_columns"] = ""

    return df


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

    for col in model_categorical_cols:
        cleaned = df[col].astype("string").str.lower().str.strip()
        df[col] = cleaned.mask(cleaned == "").fillna("unknown")

    for col in mm_cols:
        df[col] = df[col].apply(parse_dimension_mm).astype(float)

    for col in kg_cols:
        df[col] = df[col].apply(parse_weight).astype(float)

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
        if col == "Max Torque":
            df[col] = df[col].apply(parse_torque).astype(float)
        elif col == "Max Power":
            df[col] = df[col].apply(parse_power).astype(float)
        else:
            df[col] = df[col].apply(extract_first_number).astype(float)

    df["mileage_unit"] = (
        df["mileage_new"]
        .astype("string")
        .str.lower()
        .str.extract(r"(kmpl|km/kg)", expand=False)
        .fillna("unknown")
    )
    for col in mileage_cols:
        df[col] = df[col].apply(extract_first_number).astype(float)

    df = mark_implausible_values(df, is_electric=is_electric)

    model_categorical_cols.append("mileage_unit")
    model_features.append("mileage_unit")

    return df, model_features, model_categorical_cols


def _legacy_parse_weight(value) -> float | None:
    """Reproduce el comportamiento previo de ``parse_weight`` sin quitar comas."""
    if pd.isna(value):
        return None
    numbers = re.findall(r"\d+(?:\.\d+)?", str(value))
    if not numbers:
        return None
    return sum(map(float, numbers)) / len(numbers)


def _legacy_extract_first_number(value) -> float | None:
    """Reproduce el comportamiento previo de ``extract_first_number``."""
    if pd.isna(value):
        return None
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else None


def summarize_cleaning_changes(
    raw_df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Genera un resumen antes/después de filas afectadas por columna.

    Compara el resultado de la limpieza anterior frente a la limpieza corregida
    sobre las mismas filas filtradas por alcance inicial.
    """
    base = raw_df.copy()
    if "km_driven" in base.columns:
        base = base[base["km_driven"] < 1_000_000].copy()
    if "dynx_totalvalue_x" in base.columns:
        base = base[base["dynx_totalvalue_x"] < 100_000_000].copy()

    is_electric = (
        base["fuel_type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("electric")
        .fillna(False)
        if "fuel_type" in base.columns
        else pd.Series(False, index=base.index, dtype=bool)
    )

    before_df = pd.DataFrame(index=base.index)
    after_df = pd.DataFrame(index=base.index)

    for col in mm_cols:
        if col in base.columns:
            before_df[col] = pd.to_numeric(
                base[col].astype("string").str.replace("mm", "", regex=False).str.strip(),
                errors="coerce",
            ).astype(float)
            after_df[col] = base[col].apply(parse_dimension_mm).astype(float)

    for col in kg_cols:
        if col in base.columns:
            before_df[col] = base[col].apply(_legacy_parse_weight).astype(float)
            after_df[col] = base[col].apply(parse_weight).astype(float)

    for col in convert_to_number:
        if col not in base.columns:
            continue
        before_df[col] = base[col].apply(_legacy_extract_first_number).astype(float)
        if col == "Max Torque":
            after_df[col] = base[col].apply(parse_torque).astype(float)
        elif col == "Max Power":
            after_df[col] = base[col].apply(parse_power).astype(float)
        else:
            after_df[col] = base[col].apply(extract_first_number).astype(float)

    for col in mileage_cols:
        if col in base.columns:
            before_df[col] = base[col].apply(_legacy_extract_first_number).astype(float)
            after_df[col] = base[col].apply(extract_first_number).astype(float)

    for col in ("Displacement", "Values per Cylinder", "No of Cylinder", "model_year", "km_driven", "seating_capacity_new", "No Door Numbers"):
        if col not in base.columns:
            continue
        s = pd.to_numeric(base[col], errors="coerce").astype(float)
        before_col = s.copy()
        if col == "Displacement":
            before_col.loc[~is_electric & before_col.eq(0)] = np.nan
        elif col == "Values per Cylinder":
            before_col.loc[is_electric] = np.nan
            before_col.loc[~is_electric & before_col.gt(12)] = np.nan
        before_df[col] = before_col
        after_df[col] = s.copy()

    after_checked = mark_implausible_values(after_df, is_electric=is_electric)

    target_columns = columns or [
        col
        for col in [
            *mm_cols,
            *kg_cols,
            "Max Torque",
            "Max Power",
            "Displacement",
            "Values per Cylinder",
            "No of Cylinder",
            "Top Speed",
            "Acceleration",
            "Cargo Volumn",
            "Alloy Wheel Size",
            "mileage_new",
            "seating_capacity_new",
            "No Door Numbers",
            "model_year",
            "km_driven",
        ]
        if col in before_df.columns
    ]

    rows: list[dict[str, object]] = []
    for col in target_columns:
        b_series = before_df[col]
        a_series = after_checked[col]
        implausible_col = f"{col}_implausible"
        implausible_mask = (
            after_checked[implausible_col]
            if implausible_col in after_checked.columns
            else pd.Series(False, index=base.index, dtype=bool)
        )

        recovered_mask = b_series.isna() & a_series.notna()
        both_valid = b_series.notna() & a_series.notna()
        value_changed_mask = both_valid & ~np.isclose(
            b_series.fillna(0.0),
            a_series.fillna(0.0),
            rtol=1e-6,
            atol=1e-6,
        )
        became_null_mask = b_series.notna() & a_series.isna()
        affected_mask = recovered_mask | value_changed_mask | became_null_mask | implausible_mask

        rows.append({
            "column": col,
            "nulls_before": int(b_series.isna().sum()),
            "nulls_after": int(a_series.isna().sum()),
            "recovered_from_null": int(recovered_mask.sum()),
            "values_recalculated": int(value_changed_mask.sum()),
            "implausible_marked": int(implausible_mask.sum()),
            "under_100_before": int(b_series.lt(100).sum()) if col == "Kerb Weight" else 0,
            "under_100_after": int(a_series.lt(100).sum()) if col == "Kerb Weight" else 0,
            "total_rows_affected": int(affected_mask.sum()),
        })

    return pd.DataFrame(rows)

