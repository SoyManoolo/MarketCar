"""Baseline de regresión lineal."""

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from modeling import build_preprocessor, with_log_target


def train_model(X_train, y_train, numeric_features, categorical_features, random_state):
    """Entrena una regresión lineal con variables numéricas escaladas."""
    del random_state

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=True,
    )
    model = with_log_target(Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ]))
    model.fit(X_train, y_train)

    return model, {"strategy": "LinearRegression + log1p(target)"}
