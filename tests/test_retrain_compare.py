"""Pruebas unitarias e integración rápida para el flujo de reentrenamiento y comparación."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import categorical_cols, list_cols
from main import TARGET
from modeling import ARTIFACT_VERSION
from retrain_compare import (
    parse_args,
    resolve_n_jobs,
    resolve_xgb_execution,
    run_retrain_comparison,
)


class _FakeXGBRegressor(BaseEstimator, RegressorMixin):
    """Estimador ligero para verificar parámetros CUDA sin requerir GPU local."""

    captured_kwargs: list[dict[str, object]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.best_iteration = 5

    def get_params(self, deep=False):
        return dict(self.kwargs)

    def fit(self, X, y, eval_set=None, verbose=False):
        type(self).captured_kwargs.append(dict(self.kwargs))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean_, dtype=float)


def _create_synthetic_csv(path: Path, n_rows: int = 80, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    brands = np.array(["Maruti", "Hyundai", "Honda", "Tata"])
    models = np.array(["Swift", "i20", "City", "Nexon"])
    fuels = np.array(["Petrol", "Diesel"])

    rows: list[dict[str, object]] = []
    for idx in range(n_rows):
        b_idx = idx % len(brands)
        year = int(rng.integers(2013, 2023))
        km = int(rng.integers(8_000, 120_000))
        power_val = float(rng.uniform(68.0, 140.0))
        price = float(
            250_000
            + (year - 2012) * 55_000
            + power_val * 2_200
            - km * 1.1
            + rng.normal(0.0, 20_000.0)
        )
        price = max(120_000.0, min(4_500_000.0, price))

        row: dict[str, object] = {
            "usedCarId": 100_000 + idx,
            "usedCarSkuId": f"sku-{idx}",
            "used_carid": 100_000 + idx,
            # Creamos un reanuncio intencionado con mismo ucid/sid para probar agrupación.
            "ucid": 50_000 + (0 if idx == 1 else idx),
            "sid": f"seller-{0 if idx == 1 else idx}",
            "vlink": f"/car-{idx}",
            "model_year": year,
            "brand_name": brands[b_idx],
            "model_name": f"{brands[b_idx]} {models[b_idx]}",
            "variant_name": "VX",
            "km_driven": km,
            "fuel_type": fuels[idx % len(fuels)],
            "transmission_type": "Manual",
            "owner_type": "First Owner",
            "seller_type_new": "Dealer",
            "state": "Maharashtra",
            "city_name_new": "Mumbai",
            "car_segment": "Hatchback",
            "Engine Type": "K-Series",
            "Displacement": 1197.0,
            "Max Power": f"{power_val:.1f}bhp@ 6000rpm",
            "Max Torque": f"{power_val * 1.4:.1f}Nm@ 4000rpm",
            "No of Cylinder": 4.0,
            "Values per Cylinder": 4.0,
            "Value Configuration": "DOHC",
            "Turbo Charger": "No",
            "Super Charger": "No",
            "Length": "3,840 mm",
            "Width": "1,735 mm",
            "Height": "1,530 mm",
            "Wheel Base": "2,450 mm",
            "Kerb Weight": "1,060 kg",
            "Gear Box": "5 Speed",
            "Drive Type": "FWD",
            "seating_capacity_new": 5.0,
            "Steering Type": "Power",
            "Front Brake Type": "Disc",
            "Rear Brake Type": "Drum",
            "Top Speed": "165 kmph",
            "Acceleration": "12.5 Seconds",
            "Tyre Type": "Tubeless",
            "No Door Numbers": 5.0,
            "Cargo Volumn": "268-litres",
            "mileage_new": "20.5 kmpl",
            "exterior_color": "White",
            "Fuel Suppy System": "MPFI",
            "Alloy Wheel Size": "15",
            "imgCount": 10,
            "threesixty": False,
            TARGET: price,
        }
        for col in categorical_cols:
            row.setdefault(col, "unknown")
        for col in list_cols:
            row.setdefault(col, "['Air Conditioner']")
        rows.append(row)

    pd.DataFrame(rows).to_csv(path, index=False)


class TestRetrainCompare(unittest.TestCase):
    """Verifica CLI, CUDA serial, ejecución --quick, logging, checkpoint y reanudación."""

    def test_cli_and_device_n_jobs_resolution(self):
        args = parse_args(["--device", "cuda", "--n-jobs", "8", "--quick", "--output-dir", "tmp_out"])
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.n_jobs, 8)
        self.assertTrue(args.quick)
        self.assertEqual(args.output_dir, Path("tmp_out"))

        # En CUDA la búsqueda y modelo XGBoost se fuerzan a serie (n_jobs=1).
        self.assertEqual(resolve_xgb_execution("cuda", 8), (1, 1))
        self.assertEqual(resolve_xgb_execution("cpu", 3), (1, 3))
        self.assertEqual(resolve_n_jobs(2), 2)
        with self.assertRaises(ValueError):
            resolve_n_jobs(0)

    def test_quick_pipeline_saves_artifacts_report_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "synthetic_cars.csv"
            out_dir = tmp_path / "artifacts"
            report_path = tmp_path / "reports" / "comparacion_modelos.md"
            _create_synthetic_csv(csv_path, n_rows=90)

            results = run_retrain_comparison(
                models=["ridge", "lightgbm", "xgboost", "catboost"],
                device="cpu",
                n_jobs=2,
                quick=True,
                output_dir=out_dir,
                report_output=report_path,
                data_path=csv_path,
                resume=False,
            )

            self.assertTrue(results["quick"])
            self.assertEqual(results["outer_splits"], 2)
            self.assertEqual(results["inner_splits"], 2)
            self.assertEqual(results["xgboost_device"], "cpu")
            self.assertIn(results["selection"]["selected_model"], {"ridge", "lightgbm", "xgboost", "catboost"})

            # Verifica artefactos v2, reporte Markdown, log de pasos y comparación log1p vs raw.
            self.assertTrue(report_path.is_file())
            report_text = report_path.read_text(encoding="utf-8")
            self.assertIn("log1p", report_text)
            self.assertIn("rupias", report_text)
            self.assertIn("Referencia histórica congelada", report_text)

            for key in ("ridge", "lightgbm", "xgboost", "catboost"):
                art_path = out_dir / f"{key}_generalista.joblib"
                self.assertTrue(art_path.is_file())
                art = joblib.load(art_path)
                self.assertEqual(art["artifact_version"], ARTIFACT_VERSION)
                self.assertEqual(art["metadata"]["execution"]["xgboost_device"], "cpu")
                if key in {"lightgbm", "xgboost", "catboost"}:
                    self.assertIsNotNone(art["metadata"]["training"]["best_iteration"])

            # Verifica log estructurado por pasos y reanudación tras borrar un checkpoint.
            progress_jsonl = out_dir / "progress.jsonl"
            self.assertTrue(progress_jsonl.is_file())
            events_first = [
                json.loads(line)
                for line in progress_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(any(e["event"] == "outer_fold_done" and e["model"] == "xgboost" for e in events_first))

            # Simulamos corte eliminando el fold 2 de lightgbm y el artefacto final de lightgbm.
            (out_dir / "checkpoints" / "lightgbm_fold_2.joblib").unlink()
            (out_dir / "checkpoints" / "lightgbm_final.joblib").unlink()
            progress_jsonl.unlink()

            resumed_results = run_retrain_comparison(
                models=["ridge", "lightgbm"],
                device="cpu",
                n_jobs=2,
                quick=True,
                output_dir=out_dir,
                report_output=report_path,
                data_path=csv_path,
                resume=True,
            )
            self.assertIn("lightgbm", resumed_results["models"])
            events_resume = [
                json.loads(line)
                for line in progress_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            resumed_events = [e for e in events_resume if e["event"] == "outer_fold_resumed"]
            done_events = [e for e in events_resume if e["event"] == "outer_fold_done"]
            # Ridge (folds 1 y 2) + LightGBM (fold 1) deben haberse reanudado; solo LightGBM fold 2 se re-entrena.
            self.assertEqual(len(resumed_events), 3)
            self.assertEqual(len(done_events), 1)
            self.assertEqual(done_events[0]["model"], "lightgbm")
            self.assertEqual(done_events[0]["outer_fold"], 2)

    def test_cuda_option_forces_serial_and_records_device_in_metadata(self):
        _FakeXGBRegressor.captured_kwargs.clear()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = tmp_path / "synthetic_cars.csv"
            out_dir = tmp_path / "artifacts_cuda"
            report_path = tmp_path / "comparacion_cuda.md"
            _create_synthetic_csv(csv_path, n_rows=60)

            with patch("retrain_compare.XGBRegressor", _FakeXGBRegressor):
                results = run_retrain_comparison(
                    models=["xgboost"],
                    device="cuda",
                    n_jobs=8,
                    quick=True,
                    output_dir=out_dir,
                    report_output=report_path,
                    data_path=csv_path,
                    resume=False,
                )

            self.assertGreater(len(_FakeXGBRegressor.captured_kwargs), 0)
            for kwargs in _FakeXGBRegressor.captured_kwargs:
                self.assertEqual(kwargs["device"], "cuda")
                self.assertEqual(kwargs["n_jobs"], 1)

            self.assertEqual(results["xgboost_device"], "cuda")
            art = joblib.load(out_dir / "xgboost_generalista.joblib")
            self.assertEqual(art["metadata"]["xgboost_device"], "cuda")
            self.assertEqual(art["metadata"]["execution"]["xgboost_device"], "cuda")


if __name__ == "__main__":
    unittest.main()
