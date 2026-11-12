"""Pruebas unitarias de limpieza de datos con formatos reales del CSV de CarDekho."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_cleaning import (
    KGM_TO_NM,
    KW_TO_BHP,
    clean_data,
    extract_first_number,
    parse_dimension_mm,
    parse_power,
    parse_torque,
    parse_weight,
    summarize_cleaning_changes,
)
from features import categorical_cols, list_cols


def _make_raw_car_row(**overrides) -> dict[str, object]:
    """Construye un anuncio con valores realistas del CSV de CarDekho."""
    base: dict[str, object] = {
        "model_year": 2019,
        "brand_name": "Maruti",
        "model_name": "Maruti Swift",
        "variant_name": "VXI",
        "km_driven": 35_000,
        "fuel_type": "Petrol",
        "transmission_type": "Manual",
        "owner_type": "First Owner",
        "seller_type_new": "Dealer",
        "state": "Maharashtra",
        "city_name_new": "Mumbai",
        "car_segment": "Hatchback",
        "Engine Type": "1.2L K-Series",
        "Displacement": 1197,
        "Max Power": "88.50bhp@ 6000rpm",
        "Max Torque": "113Nm@ 4200rpm",
        "No of Cylinder": 4,
        "Values per Cylinder": 4,
        "Value Configuration": "DOHC",
        "Turbo Charger": "No",
        "Super Charger": "No",
        "Length": "3840 mm",
        "Width": "1735 mm",
        "Height": "1530 mm",
        "Wheel Base": "2450 mm",
        "Kerb Weight": "960 kg",
        "Gear Box": "5 Speed",
        "Drive Type": "FWD",
        "seating_capacity_new": 5,
        "Steering Type": "Power",
        "Front Brake Type": "Disc",
        "Rear Brake Type": "Drum",
        "Top Speed": "165 kmph",
        "Acceleration": "12.5 Seconds",
        "Tyre Type": "Tubeless,Radial",
        "No Door Numbers": 5,
        "Cargo Volumn": "268-litres",
        "mileage_new": "21.21 kmpl",
        "exterior_color": "White",
        "top_features": "['Air Conditioner', 'Power Steering']",
        "comfort_features": "['Power Windows Front']",
        "interior_features": "['Tachometer']",
        "exterior_features": "['Adjustable Head Lights']",
        "safety_features": "['Anti Lock Braking System', 'Driver AirBag']",
        "Fuel Suppy System": "MPFI",
        "Alloy Wheel Size": "15",
        "imgCount": 12,
        "threesixty": False,
        "dynx_totalvalue_x": 550_000,
    }
    for col in categorical_cols:
        base.setdefault(col, "unknown")
    for col in list_cols:
        base.setdefault(col, "[]")
    base.update(overrides)
    return base


class TestDataCleaning(unittest.TestCase):
    """Verifica corrección de separadores, unidades, implausibles y resumen."""

    def test_dimensions_with_thousands_commas_are_parsed(self):
        self.assertEqual(parse_dimension_mm("3,690 mm"), 3690.0)
        self.assertEqual(parse_dimension_mm("3,995mm"), 3995.0)
        self.assertEqual(parse_dimension_mm("1,680 mm"), 1680.0)
        self.assertEqual(parse_dimension_mm("1,505 mm"), 1505.0)
        self.assertEqual(parse_dimension_mm("2,450 mm"), 2450.0)
        self.assertIsNone(parse_dimension_mm(None))

    def test_kerb_weight_thousands_commas_and_ranges(self):
        # Antes: "1,060 kg" se dividía en 1 y 60 devolviendo 30.5 kg (< 100 kg).
        self.assertEqual(parse_weight("1,060 kg"), 1060.0)
        self.assertEqual(parse_weight("1,190kg"), 1190.0)
        self.assertEqual(parse_weight("860-1,060 kg"), 960.0)
        self.assertEqual(parse_weight("1,060 - 1,120 kg"), 1090.0)
        self.assertEqual(extract_first_number("1,250 Litres"), 1250.0)

    def test_max_torque_normalized_to_nm(self):
        self.assertEqual(parse_torque("190Nm@ 2000rpm"), 190.0)
        self.assertEqual(parse_torque("250 Nm @ 1,500-2,500 rpm"), 250.0)
        self.assertAlmostEqual(
            parse_torque("11.5@ 4,500(kgm@ rpm)"),
            round(11.5 * KGM_TO_NM, 4),
        )
        self.assertAlmostEqual(
            parse_torque("19.4 kgm @ 2,000 rpm"),
            round(19.4 * KGM_TO_NM, 4),
        )
        self.assertEqual(parse_torque("190Nm (19.37kgm)@ 2000rpm"), 190.0)
        self.assertEqual(parse_torque("19.37kgm (190Nm)@ 2000rpm"), 190.0)

    def test_max_power_normalized_to_bhp(self):
        self.assertEqual(parse_power("88.50bhp@ 6000rpm"), 88.5)
        self.assertEqual(parse_power("88.50bhp (66kW)@ 6000rpm"), 88.5)
        self.assertEqual(parse_power("66kW (88.50bhp)@ 6000rpm"), 88.5)
        self.assertAlmostEqual(
            parse_power("66kW@ 6000rpm"),
            round(66.0 * KW_TO_BHP, 4),
        )
        self.assertAlmostEqual(
            parse_power("41.7 kW @ 5,000 rpm"),
            round(41.7 * KW_TO_BHP, 4),
        )
        self.assertAlmostEqual(
            parse_power("66@ 6,000(kW@ rpm)"),
            round(66.0 * KW_TO_BHP, 4),
        )

    def test_implausible_values_marked_without_dropping_rows(self):
        df = pd.DataFrame([
            _make_raw_car_row(
                Length="3,690 mm",
                Width="1,680 mm",
                Height="1,505 mm",
                **{"Wheel Base": "2,450 mm", "Kerb Weight": "1,060 kg"},
            ),
            _make_raw_car_row(
                # Peso físicamente implausible (< 100 kg) y dimensión imposible.
                Length="120 mm",
                **{"Kerb Weight": "45 kg", "Max Torque": "5Nm@ 1000rpm"},
            ),
            _make_raw_car_row(
                fuel_type="Electric",
                Displacement=0,
                **{"Values per Cylinder": 4, "Max Power": "66kW@ 4000rpm"},
            ),
        ])

        cleaned, model_features, _ = clean_data(df)

        # No se elimina ningún anuncio por valores físicamente implausibles.
        self.assertEqual(len(cleaned), 3)

        # Fila 0: valores con comas recuperados correctamente y sin marcas implausibles.
        self.assertEqual(cleaned.loc[0, "Length"], 3690.0)
        self.assertEqual(cleaned.loc[0, "Kerb Weight"], 1060.0)
        self.assertFalse(bool(cleaned.loc[0, "has_implausible_values"]))

        # Fila 1: marcada como implausible y valores pasados a NaN para imputación.
        self.assertTrue(bool(cleaned.loc[1, "has_implausible_values"]))
        self.assertTrue(bool(cleaned.loc[1, "Kerb Weight_implausible"]))
        self.assertTrue(bool(cleaned.loc[1, "Length_implausible"]))
        self.assertTrue(bool(cleaned.loc[1, "Max Torque_implausible"]))
        self.assertTrue(np.isnan(cleaned.loc[1, "Kerb Weight"]))
        self.assertTrue(np.isnan(cleaned.loc[1, "Length"]))
        self.assertTrue(np.isnan(cleaned.loc[1, "Max Torque"]))

        # Fila 2: coche eléctrico conserva Displacement=0 como válido y convierte kW a bhp.
        self.assertEqual(cleaned.loc[2, "Displacement"], 0.0)
        self.assertFalse(bool(cleaned.loc[2, "Displacement_implausible"]))
        self.assertAlmostEqual(
            cleaned.loc[2, "Max Power"],
            round(66.0 * KW_TO_BHP, 4),
        )
        self.assertIn("mileage_unit", model_features)

    def test_summarize_cleaning_changes_before_and_after(self):
        raw = pd.DataFrame([
            _make_raw_car_row(
                Length="3,690 mm",
                Width="1,680 mm",
                Height="1,505 mm",
                **{
                    "Wheel Base": "2,450 mm",
                    "Kerb Weight": "1,060 kg",
                    "Max Torque": "11.5@ 4,500(kgm@ rpm)",
                    "Max Power": "66kW@ 6000rpm",
                },
            ),
            _make_raw_car_row(
                Length="3995 mm",
                Width="1745 mm",
                Height="1510 mm",
                **{
                    "Wheel Base": "2,520 mm",
                    "Kerb Weight": "1,250 kg",
                    "Max Torque": "190Nm@ 2000rpm",
                    "Max Power": "88.50bhp (66kW)@ 6000rpm",
                },
            ),
            _make_raw_car_row(
                **{"Kerb Weight": "55 kg"},
            ),
        ])

        summary = summarize_cleaning_changes(raw).set_index("column")

        self.assertEqual(summary.loc["Length", "recovered_from_null"], 1)
        self.assertEqual(summary.loc["Width", "recovered_from_null"], 1)
        self.assertEqual(summary.loc["Height", "recovered_from_null"], 1)
        self.assertEqual(summary.loc["Wheel Base", "recovered_from_null"], 2)

        # Kerb Weight: fila 0 ("1,060 kg" -> antes 30.5 < 100) y fila 2 ("55 kg" < 100)
        # daban 2 filas < 100 kg antes; después quedan 0 filas < 100 kg.
        self.assertEqual(summary.loc["Kerb Weight", "under_100_before"], 2)
        self.assertEqual(summary.loc["Kerb Weight", "under_100_after"], 0)
        self.assertEqual(summary.loc["Kerb Weight", "values_recalculated"], 2)
        self.assertEqual(summary.loc["Kerb Weight", "implausible_marked"], 1)

        # Max Torque: 1 fila en kgm recalculada a Nm.
        self.assertEqual(summary.loc["Max Torque", "values_recalculated"], 1)
        # Max Power: solo la fila exclusivamente en kW cambia; la que ya tenía bhp primero se mantiene.
        self.assertEqual(summary.loc["Max Power", "values_recalculated"], 1)


if __name__ == "__main__":
    unittest.main()
