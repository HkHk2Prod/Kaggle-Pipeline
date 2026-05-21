"""CatBoost classifier with native categorical feature handling."""

from __future__ import annotations

from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline

from kaggle_pipeline.models.base import Model
from kaggle_pipeline.models.registry import register_model


@register_model(name="CatBoostClassifier", purposes="single_target_prob_pred")
class CatBoostClassifierModel(Model):
    def generate_distribution(self, complexity):
        k = complexity
        return {
            "model__bootstrap_type": "Bernoulli",
            "model__verbose": 0,
            "model__random_seed": self.ctx.config.seed,
            "model__thread_count": 1,
            "model__iterations": randint(int(50 * k), int(150 * k)),
            "model__learning_rate": uniform(0.001, max(0.001, 0.1 / k)),
            "model__depth": randint(min(14, int(3 * k)), min(16, int(10 * k))),
            "model__min_data_in_leaf": randint(1, max(2, int(50 / k))),
            "model__subsample": uniform(0.5, 0.5),
            "model__colsample_bylevel": uniform(0.5, 0.5),
            "model__l2_leaf_reg": uniform(0.0, max(0.01, 10.0 / k)),
            "model__border_count": randint(min(32, int(32 * k)), min(255, int(255 * k))),
            "model__auto_class_weights": "Balanced",
        }

    def build_pipeline(self, param):
        from catboost import CatBoostClassifier
        from sklearn.compose import ColumnTransformer

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
                (
                    "model",
                    CatBoostClassifier(cat_features=cat_indices, allow_writing_files=False),
                ),
            ]
        )
        pipe.set_params(**param)
        return pipe
