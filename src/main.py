"""Entrena y compara modelos de tasación de coches usados."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_cleaning import clean_data


DATA_PATH = "./data/cars_details_merges.csv"
TARGET = "dynx_totalvalue_x"
RANDOM_STATE = 42
MAINSTREAM_PRICE_LIMIT = 5_000_000


def calculate_metrics(y_true, predictions):
    """Calcula las métricas de regresión en la escala original de rupias."""
    return {
        "mae": mean_absolute_error(y_true, predictions),
        "rmse": mean_squared_error(y_true, predictions) ** 0.5,
        "r2": r2_score(y_true, predictions),
    }


def print_metrics(model_name, train_metrics, test_metrics):
    """Muestra las métricas de entrenamiento y test de forma consistente."""
    print(model_name)
    print()
    print("Entrenamiento")
    print(f"MAE:  ₹{train_metrics['mae']:,.0f}")
    print(f"RMSE: ₹{train_metrics['rmse']:,.0f}")
    print(f"R²:   {train_metrics['r2']:.3f}")
    print()
    print("Test")
    print(f"MAE:  ₹{test_metrics['mae']:,.0f}")
    print(f"RMSE: ₹{test_metrics['rmse']:,.0f}")
    print(f"R²:   {test_metrics['r2']:.3f}")


def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """Evalúa un modelo ya entrenado y devuelve sus predicciones de test."""
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_metrics = calculate_metrics(y_train, train_predictions)
    test_metrics = calculate_metrics(y_test, test_predictions)

    print_metrics(model_name, train_metrics, test_metrics)
    return test_predictions


# 1. Carga, limpieza y división de los datos.
df = pd.read_csv(DATA_PATH, low_memory=False)
df, features, categorical_cols = clean_data(df)

# El tasador generalista solo cubre el mercado de hasta ₹5 millones. Los coches
# premium se conservan para análisis o para un futuro modelo especializado.
df_premium = df[df[TARGET] > MAINSTREAM_PRICE_LIMIT].copy()
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
    col for col in categorical_cols if col in X_train.columns
]
numeric_features = [
    col for col in X_train.columns if col not in categorical_features
]


# 2. Preprocesamiento para modelos lineales.
linear_numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

linear_preprocessor = ColumnTransformer(
    transformers=[
        ("num", linear_numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# 3. Regresión lineal: baseline.
linear_model = TransformedTargetRegressor(
    regressor=Pipeline(steps=[
        ("preprocessor", linear_preprocessor),
        ("model", LinearRegression()),
    ]),
    func=np.log1p,
    inverse_func=np.expm1,
)

linear_model.fit(X_train, y_train)
evaluate_model(
    linear_model,
    "Linear Regression",
    X_train,
    y_train,
    X_test,
    y_test,
)


# 4. Ridge: regularización de la regresión lineal.
ridge_model = TransformedTargetRegressor(
    regressor=Pipeline(steps=[
        ("preprocessor", linear_preprocessor),
        ("model", Ridge()),
    ]),
    func=np.log1p,
    inverse_func=np.expm1,
)

ridge_search = GridSearchCV(
    estimator=ridge_model,
    param_grid={
        "regressor__model__alpha": [0.01, 0.1, 1, 10, 100, 1_000],
    },
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1,
)

ridge_search.fit(X_train, y_train)
best_ridge_model = ridge_search.best_estimator_

print("Mejor alpha:", ridge_search.best_params_)
ridge_test_predictions = evaluate_model(
    best_ridge_model,
    "Ridge Regression",
    X_train,
    y_train,
    X_test,
    y_test,
)


# 5. Random Forest: preprocesamiento sin escalado.
rf_numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
])

rf_preprocessor = ColumnTransformer(
    transformers=[
        ("num", rf_numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

random_forest_model = TransformedTargetRegressor(
    regressor=Pipeline(steps=[
        ("preprocessor", rf_preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=400,
            max_depth=30,
            min_samples_leaf=2,
            max_features=0.5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ]),
    func=np.log1p,
    inverse_func=np.expm1,
)

random_forest_model.fit(X_train, y_train)
rf_test_predictions = evaluate_model(
    random_forest_model,
    "Random Forest ajustado",
    X_train,
    y_train,
    X_test,
    y_test,
)


# 6. Análisis del error medio por tramo de precios.
comparison = pd.DataFrame({
    "real_price": y_test,
    "ridge_prediction": ridge_test_predictions,
    "rf_prediction": rf_test_predictions,
})

comparison["price_range"] = pd.qcut(
    comparison["real_price"],
    q=5,
    duplicates="drop",
)

comparison["ridge_abs_error"] = (
    comparison["real_price"] - comparison["ridge_prediction"]
).abs()
comparison["rf_abs_error"] = (
    comparison["real_price"] - comparison["rf_prediction"]
).abs()

error_by_price_range = comparison.groupby(
    "price_range",
    observed=True,
)[["ridge_abs_error", "rf_abs_error"]].mean()

print("\nError absoluto medio por tramo de precios")
print(error_by_price_range)
