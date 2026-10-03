"""CatBoost para categorías de alta cardinalidad."""

from catboost import CatBoostRegressor


def train_model(X_train, y_train, numeric_features, categorical_features, random_state):
    """Entrena CatBoost con los parámetros seleccionados por validación cruzada."""
    del numeric_features

    parameters = {
        "iterations": 1000,
        "depth": 10,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3,
        "random_strength": 0.5,
        "bagging_temperature": 2,
        "verbose": 0,
        "cat_features": categorical_features,
        "random_seed": random_state,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    model = CatBoostRegressor(**parameters)
    model.fit(X_train, y_train)

    return model, {
        "parameters": {
            key: value for key, value in parameters.items()
            if key not in {"cat_features", "verbose", "allow_writing_files", "thread_count"}
        },
        "strategy": "Parámetros seleccionados por RandomizedSearchCV previo",
    }
