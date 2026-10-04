"""Audita los artefactos locales sin reentrenarlos ni modificar los datos.

Ejecutar desde la raíz: python src/diagnose.py [--cv]
"""

import argparse
import gc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score

from main import MODEL_TRAINERS, artifact_is_compatible, load_data
from modeling import calculate_metrics


def table(frame):
    """Markdown sin dependencia de tabulate."""
    frame = frame.copy().round(3).fillna("—")
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(map(str, row)) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def segments(frame, column):
    rows = []
    for label, group in frame.groupby(column, observed=True):
        error = group.prediction - group.actual
        rows.append({"Segmento": str(label), "n": len(group),
                     "MAE INR": error.abs().mean(),
                     "Mediana APE %": (error.abs() / group.actual).median() * 100,
                     "Sesgo INR": error.mean()})
    return pd.DataFrame(rows)


def bootstrap_difference(first, second, iterations=3000):
    """Remuestreo pareado por fila: MAE(first) - MAE(second)."""
    difference = first - second
    rng = np.random.default_rng(42)
    means = np.empty(iterations)
    for index in range(iterations):
        means[index] = difference[rng.integers(0, len(difference), len(difference))].mean()
    return difference.mean(), *np.quantile(means, [0.025, 0.975])


def cleaning_diagnostics(raw, cleaned):
    """Cuenta problemas de formato sobre la población generalista evaluada."""
    original = raw.loc[cleaned.index]
    rows = []
    for column in ["Length", "Width", "Height", "Wheel Base"]:
        comma = original[column].astype(str).str.contains(",", regex=False)
        rows.append({"Problema": f"{column}: comas convertidas a nulo",
                     "Filas": int((comma & cleaned[column].isna()).sum())})
    rows.extend([
        {"Problema": "Max Torque: texto con unidad kg", "Filas": int(
            original["Max Torque"].astype(str).str.contains("kg", case=False).sum())},
        {"Problema": "Max Power: texto con kW (a veces también bhp)", "Filas": int(
            original["Max Power"].astype(str).str.contains("kw", case=False).sum())},
        {"Problema": "Kerb Weight: texto con comas", "Filas": int(
            original["Kerb Weight"].astype(str).str.contains(",", regex=False).sum())},
        {"Problema": "Kerb Weight: resultado inferior a 100 kg", "Filas": int(
            cleaned["Kerb Weight"].lt(100).sum())},
    ])
    return ("## Problemas comprobados en la limpieza\n\n" + table(pd.DataFrame(rows)) +
            "\n\nRecuentos sobre los 37.476 coches generalistas; las filas pueden aparecer "
            "en varias categorías. extract_first_number no normaliza unidades: el par en "
            "kgm queda mezclado con Nm. La potencia incluye algunos textos solo en kW "
            "y otros con bhp primero; el contador no significa que todos los 19 casos "
            "estén mal convertidos. Las dimensiones como `3,690 mm` se convierten en "
            "nulos; parse_weight divide `1,060 kg` en 1 y 60 y devuelve 30,5. "
            "Estas son anomalías del procesamiento, no evidencia de su impacto causal "
            "en el MAE. Corregir separadores y normalizar unidades con reglas verificadas, "
            "revisar valores físicamente improbables y volver a entrenar antes de comparar "
            "la mejora con validación de entrenamiento.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", action="store_true",
                        help="Evalúa LightGBM y XGBoost con parámetros fijos en folds comunes.")
    parser.add_argument("--output", type=Path, default=Path("reports/diagnostico.md"))
    args = parser.parse_args()
    features, _, _, train, test, y_train, y_test = load_data()
    raw = pd.read_csv("data/cars_details_merges.csv", low_memory=False)
    actual = y_test.to_numpy(dtype=float)
    predictions, rows, metadata = {}, [], {}
    cv_rows = []
    for key, (name, _) in MODEL_TRAINERS.items():
        artifact = joblib.load(Path("models") / f"{key}_generalista.joblib")
        if not artifact_is_compatible(artifact, features):
            raise ValueError(f"Artefacto incompatible: {key}")
        model = artifact["model"]
        pred = np.asarray(model.predict(test), dtype=float)
        if not np.isfinite(pred).all():
            raise ValueError(f"Predicciones no finitas: {key}")
        predictions[key] = pred
        metrics = calculate_metrics(actual, pred)
        saved = artifact["metrics"]["test"]
        metadata[key] = artifact["metadata"].get("training", {})
        rows.append({"Modelo": name, "MAE INR": metrics["mae"],
                     "RMSE INR": metrics["rmse"], "R2": metrics["r2"],
                     "MAE train guardado INR": artifact["metrics"]["train"]["mae"],
                     "Delta MAE vs guardado INR": metrics["mae"] - saved["mae"],
                     "Mediana APE %": np.median(np.abs(pred - actual) / actual) * 100,
                     "Dentro ±10% %": np.mean(np.abs(pred - actual) / actual <= .1) * 100})
        if args.cv and key in {"lightgbm", "xgboost"}:
            print(f"CV común con parámetros fijos: {key}", flush=True)
            scores = -cross_val_score(
                clone(model), train, y_train,
                cv=KFold(n_splits=3, shuffle=True, random_state=42),
                scoring="neg_mean_absolute_error", n_jobs=2,
            )
            cv_rows.append({"Modelo": key, "Fold 1 INR": scores[0],
                            "Fold 2 INR": scores[1], "Fold 3 INR": scores[2],
                            "Media INR": scores.mean(), "SD folds INR": scores.std(ddof=1)})
        del artifact, model
        gc.collect()
        print(f"Evaluado: {key}, MAE={metrics['mae']:.2f}", flush=True)

    report = ["# Diagnóstico de MarketCar", "",
              "Evaluación local de los siete artefactos existentes sobre la partición original "
              "(semilla 42, 80/20). Precios y errores monetarios en INR. No se modifican "
              "los modelos ni se realiza una nueva búsqueda de hiperparámetros.",
              "## Reproducción de métricas", table(pd.DataFrame(rows).sort_values("MAE INR")),
              "El MAE de entrenamiento de esta tabla procede del artefacto guardado; "
              "las métricas de test se recalculan. Coincidir con las métricas guardadas "
              "respalda la reproducción, pero no certifica la identidad histórica de los datos."]

    checks = []
    for column in ["usedCarId", "usedCarSkuId", "used_carid", "ucid", "vlink"]:
        if column in raw:
            known = set(raw.loc[train.index, column].dropna().astype(str))
            shared = raw.loc[test.index, column].dropna().astype(str).isin(known).sum()
            checks.append({"Identificador": column, "Filas test presentes en train": int(shared)})
    train_hash = pd.util.hash_pandas_object(train, index=False)
    test_hash = pd.util.hash_pandas_object(test, index=False)
    shared_features = test_hash.isin(set(train_hash))
    report.extend(["## Independencia y calidad de datos", table(pd.DataFrame(checks)),
                   f"Filas totalmente duplicadas en el CSV original: {raw.duplicated().sum()}. "
                   f"Filas test con las mismas características de entrada que una fila train: "
                   f"{shared_features.sum()} de {len(test)}. Coincidencia de características "
                   "no demuestra que sean el mismo vehículo; los identificadores no se usan como entradas.",
                   f"Precio mínimo generalista: {min(y_train.min(), y_test.min()):,.0f}; "
                   f"precios no positivos: {int((pd.concat([y_train, y_test]) <= 0).sum())}.",
                   "La limpieza común aplica reglas por fila. Imputación, escalado y one-hot se "
                   "ajustan dentro de los pipelines y dentro de cada fold en los modelos sklearn. "
                   "No se encontraron columnas de precio ni identificadores entre las entradas.",
                   "El corte de 5 millones se aplica al precio real antes de dividir. Define una "
                   "población condicionada al objetivo; no proporciona un detector que decida "
                   "si un coche nuevo está dentro del alcance. No equivale a excluir marcas premium."])
    # Las columnas necesarias ya están limpias en la partición del modelo.
    report.append(cleaning_diagnostics(raw, pd.concat([train, test])))

    pairs = []
    base_error = np.abs(predictions["lightgbm"] - actual)
    for key in ["xgboost", "catboost", "ridge"]:
        delta, low, high = bootstrap_difference(base_error, np.abs(predictions[key] - actual))
        pairs.append({"Comparación (LightGBM − otro)": key, "Delta MAE INR": delta,
                      "IC95 inferior INR": low, "IC95 superior INR": high})
    report.extend(["## Incertidumbre de la ventaja", table(pd.DataFrame(pairs)),
                   "Bootstrap pareado, 3.000 remuestreos de filas, semilla 42. Negativo favorece "
                   "a LightGBM; si el intervalo incluye cero no hay una ventaja concluyente "
                   "bajo este procedimiento. Supone filas independientes y no incorpora la "
                   "variación del entrenamiento, la búsqueda ni la selección repetida sobre test."])
    if cv_rows:
        report.extend(["## Validación cruzada común", table(pd.DataFrame(cv_rows)),
                       "Tres folds barajados idénticos, únicamente sobre las 29.980 filas de "
                       "entrenamiento; parámetros ganadores fijos. Estos datos ya intervinieron "
                       "en la selección original: esta comparación es diagnóstica, no una "
                       "evaluación anidada ni un test independiente. SD describe dispersión "
                       "entre folds, no un intervalo de confianza."])
    report.extend(["## Metadatos de búsqueda", table(pd.DataFrame([
        {"Modelo": key, "MAE CV original INR": value.get("cv_mae", np.nan),
         "Estrategia": value.get("strategy", "sin información")}
        for key, value in metadata.items()])),
        "El módulo actual de CatBoost y su artefacto utilizan un CatBoostRegressor directo: "
        "sus predicciones están en rupias. El código de entrenamiento ajusta y_train sin "
        "log1p; los otros seis usan TransformedTargetRegressor con log1p/expm1. "
        "La comparación de métricas en rupias es válida, pero compara también objetivos "
        "de entrenamiento distintos. El historial de selección de CatBoost y Random Forest "
        "no es reproducible solo con los módulos actuales."])

    frame = test[["brand_name", "model_name", "model_year", "km_driven"]].copy()
    frame["actual"], frame["prediction"] = actual, predictions["lightgbm"]
    frame["price_band"] = pd.cut(frame.actual, [0, 300000, 600000, 1000000, 2000000, 5000000])
    report.extend(["## LightGBM: errores por precio real", table(segments(frame, "price_band")),
                   "Sesgo = predicción − precio real: negativo indica infravaloración."])
    brand = segments(frame, "brand_name")
    report.extend(["## LightGBM: marcas con al menos 50 ejemplos test",
                   table(brand[brand.n >= 50].sort_values("MAE INR", ascending=False))])
    frame["year_band"] = pd.cut(frame.model_year, [0, 2005, 2010, 2015, 2020, 2100])
    report.extend(["## LightGBM: errores por año de fabricación", table(segments(frame, "year_band"))])
    comparison = []
    for band, group in frame.groupby("price_band", observed=True):
        positions = test.index.get_indexer(group.index)
        row = {"Precio real": str(band), "n": len(group)}
        for key in ["lightgbm", "xgboost", "catboost"]:
            row[f"MAE {key} INR"] = np.abs(predictions[key][positions] - actual[positions]).mean()
        comparison.append(row)
    report.extend(["## Comparación por precio", table(pd.DataFrame(comparison))])
    error = frame.prediction - frame.actual
    square = np.square(error.to_numpy())
    worst = np.argsort(square)[-max(1, int(np.ceil(len(square) * .01))):]
    report.extend(["## Cola de errores y cobertura",
                   f"El 1% de filas con mayor error absoluto concentra el "
                   f"{square[worst].sum() / square.sum() * 100:.2f}% del error cuadrático total. "
                   f"Mediana del error absoluto: {error.abs().median():,.0f}; "
                   f"percentil 95: {error.abs().quantile(.95):,.0f}; "
                   f"percentil 99: {error.abs().quantile(.99):,.0f}. "
                   f"Sesgo medio: {error.mean():,.0f}."])
    for key in ["lightgbm", "xgboost", "catboost"]:
        e = np.abs(predictions[key] - actual)
        report.append(f"{key}: MAE sin coincidencias de características train/test "
                      f"({int((~shared_features).sum())} filas) = {e[~shared_features.to_numpy()].mean():,.0f}.")
    unseen = np.zeros(len(test), dtype=bool)
    for column in ["brand_name", "model_name", "variant_name"]:
        mask = ~test[column].isin(set(train[column]))
        unseen |= mask.to_numpy()
        report.append(f"{column}: {mask.sum()} filas test con categoría no vista en train.")
    for label, mask in [("Categorías conocidas", ~unseen), ("Alguna categoría nueva", unseen)]:
        if mask.any():
            report.append(f"{label}: n={mask.sum()}, MAE LightGBM={base_error[mask].mean():,.0f}.")
    worst_frame = frame.assign(absolute_error=error.abs()).nlargest(10, "absolute_error")
    report.extend(["## Diez mayores errores de LightGBM",
                   table(worst_frame[["brand_name", "model_name", "model_year", "actual",
                                      "prediction", "absolute_error"]]),
                   "## Riesgos de reproducibilidad y próximos pasos",
                   "1. Los artefactos no guardan huella del dataset, índices de partición, versión "
                   "del preprocesamiento ni versiones de librerías. La comprobación de carga solo "
                   "valida nombres de variables, objetivo y límite; puede aceptar un modelo "
                   "entrenado con otra revisión de datos. Guardar estas huellas en futuros entrenamientos.",
                   "2. El test se consulta para comparar siete modelos y ya ha guiado decisiones. "
                   "Reservar datos nuevos independientes o utilizar validación anidada antes de "
                   "afirmar el rendimiento final; cambiar la semilla sobre datos ya usados no crea "
                   "un test virgen para los modelos existentes.",
                   "3. Revisar los anuncios con mayores errores antes de excluirlos; distinguir "
                   "fallos de datos, poca representación y variables relevantes ausentes.",
                   "4. Comparar un objetivo en rupias con log1p usando solo validación de entrenamiento, "
                   "y calibrar intervalos en datos reservados. El logaritmo puede favorecer errores "
                   "relativos, pero el sesgo observado no permite atribuirle causalidad por sí solo.",
                   "5. Validar temporalmente cuando exista fecha de anuncio y definir una política "
                   "de alcance observable antes de conocer el precio real."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n\n".join(report) + "\n", encoding="utf-8")
    print(f"Informe: {args.output}", flush=True)


if __name__ == "__main__":
    main()
