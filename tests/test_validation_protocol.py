"""Pruebas unitarias para el protocolo de validación y selección de modelos."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from validation import (
     audit_vehicle_duplicates,
    build_grouped_kfold_splits,
    build_nested_cv_splits,
    build_vehicle_groups,
    calculate_protocol_metrics,
    evaluate_by_price_band,
    hash_cv_splits,
    paired_bootstrap_mae,
    select_best_candidate,
)


class TestValidationProtocol(unittest.TestCase):
    """Verifica agrupación por vehículo, folds comunes/anidados, métricas y selección."""

    def test_vehicle_groups_connect_shared_ids_without_merging_identical_features(self):
        raw_df = pd.DataFrame(
            {
                "usedCarId": [101, 102, 103, 104],
                "usedCarSkuId": ["sku-1", "sku-2", "sku-3", "sku-4"],
                "ucid": [9001, 9001, 9002, 9003],
                "sid": ["s-1", "s-1", "s-2", "s-3"],
                "vlink": ["/car-1", "/car-2", "/car-3", "/car-4"],
                "brand_name": ["maruti", "maruti", "hyundai", "hyundai"],
                "km_driven": [25_000, 25_000, 40_000, 40_000],
            },
            index=[10, 20, 30, 40],
        )
        X_train = raw_df.loc[[10, 20, 30], ["brand_name", "km_driven"]]
        X_test = raw_df.loc[[40], ["brand_name", "km_driven"]]

        groups = build_vehicle_groups(raw_df)
        # Filas 10 y 20 comparten ucid=9001 y sid='s-1' -> mismo grupo de vehículo.
        self.assertEqual(groups.loc[10], groups.loc[20])
        # Filas 30 y 40 tienen las mismas características (hyundai, 40_000 km),
        # pero identificadores de vehículo distintos -> grupos distintos y no se deduplican.
        self.assertNotEqual(groups.loc[30], groups.loc[40])

        audit = audit_vehicle_duplicates(raw_df, X_train, X_test)
        self.assertEqual(audit["total_rows"], 4)
        self.assertEqual(audit["unique_vehicle_groups"], 3)
        self.assertEqual(audit["multi_listing_groups"], 1)
        self.assertEqual(audit["multi_listing_rows"], 2)
        self.assertEqual(audit["cross_split_vehicle_groups"], 0)
        self.assertEqual(audit["test_rows_matching_train_features"], 1)

    def test_grouped_kfold_and_nested_cv_keep_vehicle_groups_together(self):
        index = pd.Index(range(100, 130))
        # Creamos 3 pares de reanuncios del mismo vehículo y 24 vehículos únicos.
        group_values = [f"g_{i}" for i in range(30)]
        group_values[1] = group_values[0]
        group_values[15] = group_values[2]
        group_values[28] = group_values[5]
        groups = pd.Series(group_values, index=index)

        splits = build_grouped_kfold_splits(index, groups, n_splits=5, random_state=42)
        self.assertEqual(len(splits), 5)

        seen_val: list[int] = []
        for train_pos, val_pos in splits:
            train_groups = set(groups.iloc[train_pos])
            val_groups = set(groups.iloc[val_pos])
            # Ningún vehículo puede estar a la vez en train y val de un fold.
            self.assertEqual(len(train_groups & val_groups), 0)
            seen_val.extend(val_pos.tolist())

        # Cada fila aparece exactamente una vez en validación.
        self.assertEqual(sorted(seen_val), list(range(len(index))))

        # La huella SHA-256 es determinista.
        hash_a = hash_cv_splits(index, splits)
        hash_b = hash_cv_splits(
            index,
            build_grouped_kfold_splits(index, groups, n_splits=5, random_state=42),
        )
        self.assertEqual(hash_a, hash_b)

        # Validación anidada: tampoco hay fuga de grupo en los folds internos.
        nested = build_nested_cv_splits(
            index,
            groups,
            outer_splits=5,
            inner_splits=3,
            random_state=42,
        )
        self.assertEqual(len(nested), 5)
        for outer_item in nested:
            outer_train_idx = index[outer_item["outer_train_positions"]]
            outer_train_groups = groups.loc[outer_train_idx]
            for inner_tr, inner_val in outer_item["inner_splits"]:
                self.assertEqual(
                    len(
                        set(outer_train_groups.iloc[inner_tr])
                        & set(outer_train_groups.iloc[inner_val])
                    ),
                    0,
                )

    def test_protocol_metrics_and_price_bands(self):
        y_true = np.array([200_000.0, 500_000.0, 800_000.0, 1_500_000.0, 3_000_000.0])
        # Errores: +10_000 (5%), -25_000 (5%), +80_000 (10%), -300_000 (20%), -300_000 (10%)
        preds = np.array([210_000.0, 475_000.0, 880_000.0, 1_200_000.0, 2_700_000.0])

        metrics = calculate_protocol_metrics(y_true, preds)
        self.assertAlmostEqual(metrics["mae"], 143_000.0)
        self.assertAlmostEqual(metrics["median_ape_pct"], 10.0)
        # 4 de 5 anuncios están dentro de ±10% (5%, 5%, 10%, 20%, 10%).
        self.assertAlmostEqual(metrics["within_10pct_pct"], 80.0)
        self.assertAlmostEqual(metrics["bias_inr"], -107_000.0)

        bands_df = evaluate_by_price_band(y_true, preds)
        self.assertEqual(len(bands_df), 5)
        self.assertEqual(bands_df["n"].tolist(), [1, 1, 1, 1, 1])

    def test_selection_criterion_and_tie_breaker(self):
        rng = np.random.default_rng(42)
        y_true = rng.uniform(250_000.0, 3_500_000.0, size=200)

        # Caso 1: Candidato A supera claramente a B en MAE.
        pred_a = y_true + rng.normal(0.0, 30_000.0, size=200)
        pred_b = y_true + rng.normal(0.0, 90_000.0, size=200)
        res_clear = select_best_candidate(
            [
                {"name": "Model_A", "oof_predictions": pred_a, "fold_maes": [24_000, 25_000, 26_000]},
                {"name": "Model_B", "oof_predictions": pred_b, "fold_maes": [70_000, 72_000, 74_000]},
            ],
            y_true,
        )
        self.assertEqual(res_clear["selected_model"], "Model_A")
        self.assertTrue(res_clear["paired_bootstrap"][0]["conclusive"])

        # Caso 2: Empate técnico en MAE (< 1.000 INR y IC95 incluye 0),
        # desempata por menor dispersión entre folds / menor RMSE / simplicidad.
        base_noise = rng.normal(0.0, 40_000.0, size=200)
        pred_unstable = y_true + base_noise
        # pred_stable tiene prácticamente el mismo MAE (+50 INR), pero mucho menor SD entre folds.
        pred_stable = y_true + base_noise * 0.999 + 100.0
        boot = paired_bootstrap_mae(y_true, pred_unstable, pred_stable)
        self.assertLess(abs(boot["delta_mae_inr"]), 1_000.0)

        res_tie = select_best_candidate(
            [
                {
                    "name": "Unstable_Complex",
                    "oof_predictions": pred_unstable,
                    "fold_maes": [28_000.0, 32_000.0, 36_000.0],
                    "complexity_rank": 3,
                },
                {
                    "name": "Stable_Simple",
                    "oof_predictions": pred_stable,
                    "fold_maes": [31_900.0, 32_050.0, 32_100.0],
                    "complexity_rank": 1,
                },
            ],
            y_true,
        )
        self.assertEqual(res_tie["selected_model"], "Stable_Simple")


if __name__ == "__main__":
    unittest.main()
