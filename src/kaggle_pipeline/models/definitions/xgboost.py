"""XGBoost classifier with native categorical support."""

from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.pipeline import Pipeline

from kaggle_pipeline.models.base import Model
from kaggle_pipeline.models.registry import register_model


@register_model(name="XGBClassifier", purposes="single_target_prob_pred")
class XGBClassifierModel(Model):
    # Native categorical handling via ``enable_categorical``; capability wins, so
    # the raw category columns are passed through and ``categorical_encoding`` is unused.
    handles_categoricals = True

    def generate_distribution(self):
        # Wide fixed ranges spanning shallow stumps through deep trees, with
        # regularizers across many orders of magnitude for a diverse spectrum.
        return {
            "model__tree_method": "hist",
            "model__eval_metric": "auc",
            "model__enable_categorical": True,
            "model__verbosity": 0,
            "model__n_jobs": 1,
            "model__random_state": self.ctx.config.seed,
            "model__n_estimators": randint(50, 600),
            "model__learning_rate": loguniform(0.005, 0.5),
            "model__max_depth": randint(2, 24),
            "model__min_child_weight": loguniform(0.1, 100.0),
            "model__subsample": uniform(0.3, 0.7),  # [0.3, 1.0]
            "model__colsample_bytree": uniform(0.3, 0.7),
            "model__gamma": uniform(0.0, 5.0),
            "model__reg_alpha": loguniform(1e-8, 100.0),
            "model__reg_lambda": uniform(0.0, 200.0),
        }

    def build_pipeline(self, param):
        from sklearn.compose import ColumnTransformer
        from xgboost import XGBClassifier

        numerical_columns = self.ctx.num_cols_x
        categorical_columns = self.ctx.cat_cols_x

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("passthrough", "passthrough")]), numerical_columns),
                ("cat", Pipeline([("passthrough", "passthrough")]), categorical_columns),
            ]
        ).set_output(transform="pandas")

        pipe = Pipeline([("preprocessor", preprocessor), ("model", XGBClassifier())])
        pipe.set_params(**param)
        return pipe
