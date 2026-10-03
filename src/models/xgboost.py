"""XGBoost con búsqueda aleatoria de hiperparámetros."""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from modeling import build_preprocessor, with_log_target


def train_model(X_train, y_train, numeric_features, categorical_features, random_state):
    """Selecciona XGBoost por MAE mediante validación cruzada de tres folds."""
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=False,
    )
    estimator = with_log_target(Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device="cpu",
            n_jobs=1,
            random_state=random_state,
        )),
    ]))

    parameter_distributions = {
        "regressor__model__n_estimators": [400, 700, 1_000],
        "regressor__model__learning_rate": [0.02, 0.05, 0.1],
        "regressor__model__max_depth": [4, 6, 8],
        "regressor__model__min_child_weight": [1, 5, 10],
        "regressor__model__subsample": [0.7, 0.85, 1.0],
        "regressor__model__colsample_bytree": [0.7, 0.85, 1.0],
        "regressor__model__reg_alpha": [0, 0.1, 1],
        "regressor__model__reg_lambda": [1, 5, 10],
    }
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_distributions,
        n_iter=30,
        scoring="neg_mean_absolute_error",
        cv=3,
        n_jobs=4,
        random_state=random_state,
        verbose=1,
    )
    search.fit(X_train, y_train)

    metadata = {
        "best_params": search.best_params_,
        "cv_mae": float(-search.best_score_),
        "strategy": "RandomizedSearchCV(30 candidatos, 3 folds) + log1p(target)",
    }
    return search.best_estimator_, metadata
