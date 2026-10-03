"""Regresión Ridge con selección de la regularización."""

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from modeling import build_preprocessor, with_log_target


def train_model(X_train, y_train, numeric_features, categorical_features, random_state):
    """Selecciona ``alpha`` por MAE con validación cruzada de cinco folds."""
    del random_state

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=True,
    )
    estimator = with_log_target(Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge()),
    ]))

    search = GridSearchCV(
        estimator=estimator,
        param_grid={
            "regressor__model__alpha": [0.01, 0.1, 1, 10, 100, 1_000],
        },
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    metadata = {
        "best_params": search.best_params_,
        "cv_mae": float(-search.best_score_),
        "strategy": "GridSearchCV(5 folds) + log1p(target)",
    }
    return search.best_estimator_, metadata
