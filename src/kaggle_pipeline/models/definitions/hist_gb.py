"""Histogram gradient boosting classifier with native categorical support."""

from __future__ import annotations

from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline

from kaggle_pipeline.models.base import Model
from kaggle_pipeline.models.registry import register_model


@register_model(name="HistGBClassifier", purposes="single_target_prob_pred")
class HistGBClassifierModel(Model):
    def generate_distribution(self, complexity):
        k = complexity
        return {
            "model__early_stopping": True,
            "model__scoring": self.ctx.config.scoring,
            "model__validation_fraction": 0.1,
            "model__n_iter_no_change": randint(int(10 * k), int(20 * k)),
            "model__random_state": self.ctx.config.seed,
            "model__max_iter": randint(int(500 * k), int(1000 * k)),
            "model__learning_rate": uniform(0.001, 0.1 / k),
            "model__max_depth": randint(max(2, int(3 * k)), max(3, int(10 * k))),
            "model__max_leaf_nodes": randint(max(2, int(20 * k)), max(3, int(150 * k))),
            "model__min_samples_leaf": randint(1, max(2, int(50 / k))),
            "model__l2_regularization": uniform(0.0, max(1e-6, 1.0 / k)),
            "model__max_features": uniform(min(0.99, 0.5 + 0.5 * (1 - 1 / k)), min(0.01, 0.5 / k)),
            "model__class_weight": ["balanced"],
        }

    def build_pipeline(self, param):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingClassifier

        numerical_columns = self.ctx.num_cols_x
        categorical_columns = self.ctx.cat_cols_x
        cat_indices = list(
            range(len(numerical_columns), len(numerical_columns) + len(categorical_columns))
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("passthrough", "passthrough")]), numerical_columns),
                ("cat", Pipeline([("passthrough", "passthrough")]), categorical_columns),
            ]
        )

        pipe = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", HistGradientBoostingClassifier(categorical_features=cat_indices)),
            ]
        )
        pipe.set_params(**param)
        return pipe
