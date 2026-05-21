"""Logistic regression model (scaled numerics + one-hot categoricals)."""

from __future__ import annotations

from scipy.stats import loguniform
from sklearn.pipeline import Pipeline

from kaggle_pipeline.models.base import Model
from kaggle_pipeline.models.registry import register_model


@register_model(name="LogisticRegression", purposes="single_target_prob_pred", lower=2, upper=5)
class LogisticRegressionModel(Model):
    def generate_distribution(self, complexity):
        k = complexity
        return {
            "model__random_state": self.ctx.config.seed,
            "model__max_iter": int(400 * k),
            "model__C": loguniform(1e-2 * k, 1.0 * k),
            # "model__l1_ratio": uniform(0.05, 0.9),
            "model__solver": "lbfgs",
            "model__class_weight": "balanced",
        }

    def build_pipeline(self, param):
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        num_prep = Pipeline(steps=[("scaler", StandardScaler())])
        cat_prep = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="error"))])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_prep, self.ctx.num_cols_x),
                ("cat", cat_prep, self.ctx.cat_cols_x),
            ]
        )

        pipe = Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression())])
        pipe.set_params(**param)
        return pipe
