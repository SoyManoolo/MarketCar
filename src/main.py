"""Orquesta el entrenamiento, carga y guardado de modelos de MarketCar.

Ejemplos:
    python src/main.py --models lightgbm
    python src/main.py --models ridge catboost
    python src/main.py --models all --retrain
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data_cleaning import clean_data
from modeling import evaluate_model, load_artifact, print_metrics, save_artifact
from models.catboost import train_model as train_catboost
from models.extra_trees import train_model as train_extra_trees
from models.lightgbm import train_model as train_lightgbm
from models.linear import train_model as train_linear
from models.random_forest import train_model as train_random_forest
from models.ridge import train_model as train_ridge
from models.xgboost import train_model as train_xgboost


DATA_PATH = Path("data/cars_details_merges.csv")
TARGET = "dynx_totalvalue_x"
RANDOM_STATE = 42
MAINSTREAM_PRICE_LIMIT = 5_000_000

MODEL_TRAINERS = {
    "linear": ("Linear Regression", train_linear),
    "ridge": ("Ridge Regression", train_ridge),
    "random_forest": ("Random Forest ajustado", train_random_forest),
    "extra_trees": ("Extra Trees ajustado", train_extra_trees),
    "catboost": ("CatBoost ajustado", train_catboost),
    "lightgbm": ("LightGBM ajustado", train_lightgbm),
    "xgboost": ("XGBoost ajustado", train_xgboost),
}


def parse_args() -> argparse.Namespace:
    """Define los modelos a ejecutar y si deben reentrenarse."""
    parser = argparse.ArgumentParser(
        description="Entrena o carga modelos de tasación MarketCar.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["all", *MODEL_TRAINERS],
        default=["all"],
        help="Modelos a ejecutar. Por defecto: todos.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Ignora artefactos existentes y vuelve a entrenar los modelos elegidos.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("models"),
        help="Carpeta para los modelos guardados (por defecto: models).",
    )
    return parser.parse_args()


def requested_models(models: list[str]) -> list[str]:
    """Expande ``all`` y elimina selecciones duplicadas conservando el orden."""
    names = list(MODEL_TRAINERS) if "all" in models else models
    return list(dict.fromkeys(names))


def load_data():
    """Carga, limpia y limita los datos al alcance del tasador generalista."""
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df, features, categorical_cols = clean_data(df)

    premium_count = int((df[TARGET] > MAINSTREAM_PRICE_LIMIT).sum())
    df = df[df[TARGET] <= MAINSTREAM_PRICE_LIMIT].copy()

    X = df[features].copy()
    y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    categorical_features = [
        column for column in categorical_cols if column in X_train.columns
    ]
    numeric_features = [
        column for column in X_train.columns if column not in categorical_features
    ]

    print(
        f"Datos generalistas: {len(df):,} coches. "
        f"Vehículos premium fuera de alcance: {premium_count:,}."
    )
    return (
        features,
        categorical_features,
        numeric_features,
        X_train,
        X_test,
        y_train,
        y_test,
    )


def artifact_is_compatible(artifact, features: list[str]) -> bool:
    """Comprueba que un modelo guardado coincide con el alcance actual."""
    metadata = artifact.get("metadata", {})
    return (
        artifact.get("features") == features
        and metadata.get("target") == TARGET
        and metadata.get("mainstream_price_limit") == MAINSTREAM_PRICE_LIMIT
    )


def print_summary(results: list[dict[str, object]]) -> None:
    """Muestra una comparación compacta de los modelos ejecutados."""
    if not results:
        return

    summary = pd.DataFrame(results).set_index("Modelo")
    print("\nResumen de resultados en test")
    print(summary.to_string(formatters={
        "MAE": "₹{:,.0f}".format,
        "RMSE": "₹{:,.0f}".format,
        "R²": "{:.3f}".format,
    }))


def main() -> None:
    """Entrena o carga los modelos solicitados y guarda sus artefactos."""
    args = parse_args()
    selected_models = requested_models(args.models)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    (
        features,
        categorical_features,
        numeric_features,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_data()

    summary_rows: list[dict[str, object]] = []
    for key in selected_models:
        display_name, trainer = MODEL_TRAINERS[key]
        artifact_path = args.artifacts_dir / f"{key}_generalista.joblib"
        loaded_from_artifact = artifact_path.exists() and not args.retrain

        if loaded_from_artifact:
            artifact = load_artifact(artifact_path)
            if not artifact_is_compatible(artifact, features):
                raise ValueError(
                    f"{artifact_path} no coincide con las variables o el alcance actual. "
                    "Ejecuta de nuevo con --retrain."
                )
            model = artifact["model"]
            print(f"\nCargado: {artifact_path}")
        else:
            print(f"\nEntrenando {display_name}...")
            model, training_metadata = trainer(
                X_train,
                y_train,
                numeric_features,
                categorical_features,
                RANDOM_STATE,
            )
            if "best_params" in training_metadata:
                print("Mejores parámetros:")
                print(training_metadata["best_params"])
            if "cv_mae" in training_metadata:
                print(f"MAE medio CV: ₹{training_metadata['cv_mae']:,.0f}")

        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        if not loaded_from_artifact:
            save_artifact(
                artifact_path,
                model_name=display_name,
                model=model,
                features=features,
                categorical_features=categorical_features,
                metrics=metrics,
                metadata={
                    "target": TARGET,
                    "mainstream_price_limit": MAINSTREAM_PRICE_LIMIT,
                    "training_rows": len(X_train),
                    "test_rows": len(X_test),
                    "training": training_metadata,
                },
            )
            print(f"Guardado: {artifact_path}")
        print_metrics(display_name, metrics)
        summary_rows.append({
            "Modelo": display_name,
            "MAE": metrics["test"]["mae"],
            "RMSE": metrics["test"]["rmse"],
            "R²": metrics["test"]["r2"],
        })

    print_summary(summary_rows)


if __name__ == "__main__":
    main()
