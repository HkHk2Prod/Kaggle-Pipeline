"""Logistic regression model (scaled numerics + one-hot categoricals)."""

from __future__ import annotations

from scipy.stats import loguniform
from sklearn.pipeline import Pipeline

from kaggle_pipeline.models.base import Model
from kaggle_pipeline.models.registry import register_model


@register_model(
    name="LogisticRegression", purposes="single_target_prob_pred", lower=0.02, upper=0.05
)
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
        from sklearn.preprocessing import StandardScaler

        from kaggle_pipeline.preprocessing import categorical_transformer_specs

        # LogisticRegression cannot consume raw categoricals: encode each per the
        # run's resolved plan (default: frequency), then scale numerics together
        # with the now-numeric encoded columns so all features share a scale.
        cat_specs = categorical_transformer_specs(
            self.ctx.categorical_encoding, self.ctx.cat_cols_x
        )
        preprocessor = ColumnTransformer(
            transformers=[("num", "passthrough", self.ctx.num_cols_x), *cat_specs]
        )

        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression()),
            ]
        )
        pipe.set_params(**param)
        return pipe
