"""CatBoost classifier with native categorical feature handling."""

from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.pipeline import Pipeline

from kaggle_pipeline.models.base import Model
from kaggle_pipeline.models.registry import register_model


@register_model(name="CatBoostClassifier", purposes="single_target_prob_pred")
class CatBoostClassifierModel(Model):
    # Native categorical handling (ordered target statistics); capability wins,
    # so the raw columns are passed through and ``categorical_encoding`` is unused.
    handles_categoricals = True

    def generate_distribution(self):
        # Wide fixed ranges: depth up to catboost's hard cap of 16-ish, iterations
        # up to 600, log-spaced learning rate and l2 for a diverse spectrum.
        return {
            "model__bootstrap_type": "Bernoulli",
            "model__verbose": 0,
            "model__random_seed": self.ctx.config.seed,
            "model__thread_count": 1,
            "model__iterations": randint(50, 600),
            "model__learning_rate": loguniform(0.005, 0.5),
            "model__depth": randint(2, 12),
            "model__min_data_in_leaf": randint(1, 200),
            "model__subsample": uniform(0.3, 0.7),  # [0.3, 1.0]
            "model__colsample_bylevel": uniform(0.3, 0.7),
            "model__l2_leaf_reg": loguniform(1.0, 100.0),
            "model__border_count": randint(32, 255),
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
