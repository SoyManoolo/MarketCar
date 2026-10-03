"""Extra Trees con búsqueda aleatoria de hiperparámetros."""

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from modeling import build_preprocessor, with_log_target


def train_model(X_train, y_train, numeric_features, categorical_features, random_state):
    """Selecciona Extra Trees por MAE mediante validación cruzada de tres folds."""
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=False,
    )
    estimator = with_log_target(Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", ExtraTreesRegressor(
            random_state=random_state,
            n_jobs=1,
        )),
    ]))

    parameter_distributions = {
        "regressor__model__n_estimators": [200, 400, 600],
        "regressor__model__max_depth": [None, 20, 30, 40],
        "regressor__model__min_samples_split": [2, 5, 10],
        "regressor__model__min_samples_leaf": [1, 2, 4],
        "regressor__model__max_features": [0.3, 0.5, 0.7, 1.0],
    }
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_distributions,
        n_iter=24,
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
        "strategy": "RandomizedSearchCV(24 candidatos, 3 folds) + log1p(target)",
    }
    return search.best_estimator_, metadata
