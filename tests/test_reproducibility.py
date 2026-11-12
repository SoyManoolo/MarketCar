"""Pruebas unitarias de reproducibilidad, validación y archivo histórico de artefactos."""

import sys
import tempfile
import types
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from main import MAINSTREAM_PRICE_LIMIT, TARGET, artifact_is_compatible
from modeling import (
    ARTIFACT_VERSION,
    HISTORICAL_SUBDIR,
    build_preprocessor,
    build_reproducibility_context,
    check_artifact_compatibility,
    evaluate_model,
    load_artifact,
    preserve_historical_artifacts,
    save_artifact,
    with_log_target,
)


def _sample_experiment_data(n_rows: int = 30, seed: int = 42):
    rng = np.random.default_rng(seed)
    raw_df = pd.DataFrame({
        "usedCarId": [f"car-{i}" for i in range(n_rows)],
        "usedCarSkuId": [f"sku-{i}" for i in range(n_rows)],
        "model_year": rng.integers(2012, 2024, size=n_rows),
        "km_driven": rng.integers(5_000, 120_000, size=n_rows),
        "Max Power": rng.uniform(60.0, 150.0, size=n_rows),
        "brand_name": rng.choice(["maruti", "hyundai", "honda"], size=n_rows),
        "fuel_type": rng.choice(["petrol", "diesel"], size=n_rows),
        TARGET: rng.uniform(250_000.0, 1_800_000.0, size=n_rows),
    })
    features = ["model_year", "km_driven", "Max Power", "brand_name", "fuel_type"]
    categorical_features = ["brand_name", "fuel_type"]
    numeric_features = ["model_year", "km_driven", "Max Power"]

    X = raw_df[features].copy()
    y = raw_df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
    )
    return (
        raw_df,
        X,
        y,
        X_train,
        X_test,
        y_train,
        y_test,
        features,
        categorical_features,
        numeric_features,
    )


def _fit_dummy_ridge(X_train, y_train, numeric_features, categorical_features):
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=True,
    )
    model = with_log_target(Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1.0)),
    ]))
    model.fit(X_train, y_train)
    return model, {
        "best_params": {"regressor__model__alpha": 1.0},
        "cv_mae": 45_000.0,
        "strategy": "GridSearchCV(5 folds) + log1p(target)",
    }


class TestReproducibility(unittest.TestCase):
    """Verifica el registro de metadatos, rechazo de incompatibles y archivo histórico."""

    def test_saved_artifact_contains_all_reproducibility_fields(self):
        (
            raw_df,
            X,
            y,
            X_train,
            X_test,
            y_train,
            y_test,
            features,
            categorical_features,
            numeric_features,
        ) = _sample_experiment_data()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "cars.csv"
            raw_df.to_csv(csv_path, index=False)

            context = build_reproducibility_context(
                X=X,
                y=y,
                X_train=X_train,
                X_test=X_test,
                features=features,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                target=TARGET,
                mainstream_price_limit=MAINSTREAM_PRICE_LIMIT,
                random_state=42,
                data_path=csv_path,
                raw_df=raw_df,
            )
            model, training_meta = _fit_dummy_ridge(
                X_train,
                y_train,
                numeric_features,
                categorical_features,
            )
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
            artifact_path = tmp_path / "ridge_generalista.joblib"

            save_artifact(
                artifact_path,
                model_name="Ridge Regression",
                model=model,
                features=features,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                metrics=metrics,
                metadata={
                    "target": TARGET,
                    "mainstream_price_limit": MAINSTREAM_PRICE_LIMIT,
                    "training": training_meta,
                },
                reproducibility_context=context,
            )

            loaded = load_artifact(
                artifact_path,
                features=features,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                reproducibility_context=context,
            )
            self.assertEqual(loaded["artifact_version"], ARTIFACT_VERSION)
            meta = loaded["metadata"]

            # 1. Huella del dataset.
            self.assertIsNotNone(meta["dataset_fingerprint"]["file_sha256"])
            self.assertIsNotNone(meta["dataset_fingerprint"]["dataframe_sha256"])
            self.assertEqual(meta["dataset_fingerprint"]["rows"], len(X))

            # 2. Versión y huella de limpieza.
            self.assertIn("cleaning_version", meta)
            self.assertIn("code_sha256", meta["cleaning_fingerprint"])

            # 3. Índices e identificadores de partición y semilla.
            self.assertEqual(meta["random_state"], 42)
            self.assertEqual(meta["partition"]["train_indices"], list(X_train.index))
            self.assertEqual(meta["partition"]["test_indices"], list(X_test.index))
            self.assertIn("usedCarId", meta["partition"]["identifier_hashes"])

            # 4. Variables, hiperparámetros y versiones de librerías.
            self.assertEqual(meta["variables"]["features"], features)
            self.assertEqual(meta["variables"]["numeric_features"], numeric_features)
            self.assertEqual(meta["hyperparameters"]["estimator_class"], "Ridge")
            self.assertEqual(
                meta["hyperparameters"]["selected_params"],
                {"regressor__model__alpha": 1.0},
            )
            for pkg in ("python", "pandas", "numpy", "scikit-learn", "lightgbm", "xgboost", "catboost"):
                self.assertIn(pkg, meta["library_versions"])

    def test_rejects_artifact_when_data_cleaning_or_split_changes(self):
        (
            raw_df,
            X,
            y,
            X_train,
            X_test,
            y_train,
            y_test,
            features,
            categorical_features,
            numeric_features,
        ) = _sample_experiment_data()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            context = build_reproducibility_context(
                X=X,
                y=y,
                X_train=X_train,
                X_test=X_test,
                features=features,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                target=TARGET,
                mainstream_price_limit=MAINSTREAM_PRICE_LIMIT,
                random_state=42,
                raw_df=raw_df,
            )
            model, training_meta = _fit_dummy_ridge(
                X_train,
                y_train,
                numeric_features,
                categorical_features,
            )
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
            artifact_path = tmp_path / "ridge_generalista.joblib"
            save_artifact(
                artifact_path,
                model_name="Ridge Regression",
                model=model,
                features=features,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                metrics=metrics,
                metadata={
                    "target": TARGET,
                    "mainstream_price_limit": MAINSTREAM_PRICE_LIMIT,
                    "training": training_meta,
                },
                reproducibility_context=context,
            )

            # Caso A: cambia un dato en X.
            modified_X = X.copy()
            modified_X.iloc[0, 0] += 1
            modified_data_ctx = build_reproducibility_context(
                X=modified_X,
                y=y,
                X_train=X_train,
                X_test=X_test,
                features=features,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                target=TARGET,
                mainstream_price_limit=MAINSTREAM_PRICE_LIMIT,
                random_state=42,
            )
            with self.assertRaisesRegex(ValueError, "dataframe_sha256"):
                load_artifact(
                    artifact_path,
                    features=features,
                    reproducibility_context=modified_data_ctx,
                )

            # Caso B: cambia la versión o el código de limpieza.
            fake_cleaning = types.ModuleType("fake_cleaning")
            fake_cleaning.CLEANING_VERSION = "9.9.9"
            modified_cleaning_ctx = build_reproducibility_context(
                X=X,
                y=y,
                X_train=X_train,
                X_test=X_test,
                features=features,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                target=TARGET,
                mainstream_price_limit=MAINSTREAM_PRICE_LIMIT,
                random_state=42,
                cleaning_module=fake_cleaning,
            )
            with self.assertRaisesRegex(ValueError, "limpieza"):
                load_artifact(
                    artifact_path,
                    features=features,
                    reproducibility_context=modified_cleaning_ctx,
                )

            # Caso C: cambia la semilla / partición.
            X_train_99, X_test_99, _, _ = train_test_split(X, y, test_size=0.2, random_state=99)
            modified_split_ctx = build_reproducibility_context(
                X=X,
                y=y,
                X_train=X_train_99,
                X_test=X_test_99,
                features=features,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                target=TARGET,
                mainstream_price_limit=MAINSTREAM_PRICE_LIMIT,
                random_state=99,
            )
            with self.assertRaisesRegex(ValueError, "random_state|partición"):
                load_artifact(
                    artifact_path,
                    features=features,
                    reproducibility_context=modified_split_ctx,
                )

    def test_legacy_v1_artifact_is_rejected_and_preserved_as_historical(self):
        (
            raw_df,
            X,
            y,
            X_train,
            X_test,
            y_train,
            y_test,
            features,
            categorical_features,
            numeric_features,
        ) = _sample_experiment_data()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifact_path = tmp_path / "ridge_generalista.joblib"
            model, training_meta = _fit_dummy_ridge(
                X_train,
                y_train,
                numeric_features,
                categorical_features,
            )
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

            # Simulamos un artefacto histórico v1 tal como se guardaba antes de #2.
            legacy_artifact = {
                "artifact_version": 1,
                "model_name": "Ridge Regression",
                "model": model,
                "features": list(features),
                "categorical_features": list(categorical_features),
                "metrics": metrics,
                "metadata": {
                    "target": TARGET,
                    "mainstream_price_limit": MAINSTREAM_PRICE_LIMIT,
                    "training_rows": len(X_train),
                    "test_rows": len(X_test),
                    "training": training_meta,
                },
            }
            joblib.dump(legacy_artifact, artifact_path)

            # 1. Al comprobar compatibilidad o cargarlo para uso actual, se rechaza.
            self.assertFalse(artifact_is_compatible(legacy_artifact, features))
            compatible, reasons = check_artifact_compatibility(
                legacy_artifact,
                features=features,
            )
            self.assertFalse(compatible)
            self.assertTrue(any("Versión de artefacto incompatible" in r for r in reasons))

            with self.assertRaises(ValueError):
                load_artifact(artifact_path, features=features)

            # 2. Puede consultarse explícitamente como referencia histórica.
            historical_loaded = load_artifact(
                artifact_path,
                features=features,
                allow_historical=True,
            )
            self.assertEqual(historical_loaded["artifact_version"], 1)

            # 3. Al preservar o reentrenar, el artefacto v1 se conserva en models/historical/.
            archived_paths = preserve_historical_artifacts(tmp_path)
            self.assertEqual(len(archived_paths), 1)
            self.assertTrue(archived_paths[0].exists())
            self.assertEqual(archived_paths[0].parent, tmp_path / HISTORICAL_SUBDIR)

            context = build_reproducibility_context(
                X=X,
                y=y,
                X_train=X_train,
                X_test=X_test,
                features=features,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                target=TARGET,
                mainstream_price_limit=MAINSTREAM_PRICE_LIMIT,
                random_state=42,
                raw_df=raw_df,
            )
            save_artifact(
                artifact_path,
                model_name="Ridge Regression",
                model=model,
                features=features,
                categorical_features=categorical_features,
                numeric_features=numeric_features,
                metrics=metrics,
                metadata={
                    "target": TARGET,
                    "mainstream_price_limit": MAINSTREAM_PRICE_LIMIT,
                    "training": training_meta,
                },
                reproducibility_context=context,
            )

            # El archivo histórico v1 sigue intacto tras guardar el nuevo v2.
            self.assertTrue(archived_paths[0].exists())
            archived_v1 = joblib.load(archived_paths[0])
            self.assertEqual(archived_v1["artifact_version"], 1)


if __name__ == "__main__":
    unittest.main()
