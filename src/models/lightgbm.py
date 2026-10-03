"""LightGBM con búsqueda aleatoria de hiperparámetros."""

from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from modeling import build_preprocessor, with_log_target


def train_model(X_train, y_train, numeric_features, categorical_features, random_state):
    """Selecciona LightGBM mediante MAE y validación cruzada de tres folds."""
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=False,
    )
    estimator = with_log_target(Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LGBMRegressor(
            random_state=random_state,
            n_jobs=1,
            subsample_freq=1,
            verbosity=-1,
        )),
    ]))

    parameter_distributions = {
        "regressor__model__n_estimators": [300, 500, 800, 1200],
        "regressor__model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "regressor__model__num_leaves": [15, 31, 63, 127],
        "regressor__model__max_depth": [-1, 6, 10, 15],
        "regressor__model__min_child_samples": [10, 20, 40, 80],
        "regressor__model__subsample": [0.7, 0.85, 1.0],
        "regressor__model__colsample_bytree": [0.7, 0.85, 1.0],
        "regressor__model__reg_lambda": [0, 0.1, 1, 5],
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
