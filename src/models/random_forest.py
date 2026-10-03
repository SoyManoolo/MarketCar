"""Random Forest ajustado para la tasación generalista."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from modeling import build_preprocessor, with_log_target


def train_model(X_train, y_train, numeric_features, categorical_features, random_state):
    """Entrena el bosque con la configuración seleccionada previamente."""
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=False,
    )
    parameters = {
        "n_estimators": 400,
        "max_depth": 30,
        "min_samples_leaf": 2,
        "max_features": 0.5,
        "random_state": random_state,
        "n_jobs": -1,
    }
    model = with_log_target(Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(**parameters)),
    ]))
    model.fit(X_train, y_train)

    return model, {
        "parameters": parameters,
        "strategy": "RandomForest ajustado + log1p(target)",
    }
