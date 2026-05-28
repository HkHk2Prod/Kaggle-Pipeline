"""LightGBM classifier with ordinal-encoded categorical features."""

from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.pipeline import Pipeline

from kaggle_pipeline.models.base import Model
from kaggle_pipeline.models.registry import register_model


@register_model(name="LGBMClassifier", purposes="single_target_prob_pred")
class LGBMClassifierModel(Model):
    # Native categorical handling (ordinal-encoded then ``categorical_feature``);
    # capability wins, so ``categorical_encoding`` is unused for this model.
    handles_categoricals = True
    _fit_params = {"model__categorical_feature": "auto"}

    def generate_distribution(self):
        # Wide fixed ranges: leaves up to 1024, depth up to 32, reg spanning many
        # orders of magnitude so one draw may yield a tight stub and the next a
        # very expressive model.
        return {
            "model__verbose": -1,
            "model__random_state": self.ctx.config.seed,
            "model__n_jobs": 1,
            "model__n_estimators": randint(50, 600),
            "model__learning_rate": loguniform(0.005, 0.5),
            "model__max_depth": randint(2, 32),
            "model__num_leaves": randint(4, 1024),
            "model__min_child_samples": randint(1, 500),
            "model__subsample": uniform(0.3, 0.7),  # [0.3, 1.0]
            "model__subsample_freq": randint(1, 10),
            "model__colsample_bytree": uniform(0.3, 0.7),
            "model__reg_alpha": loguniform(1e-8, 100.0),
            "model__reg_lambda": uniform(0.0, 200.0),
            "model__class_weights": "balanced",
        }

    def build_pipeline(self, param):
        from lightgbm import LGBMClassifier
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

        numerical_columns = self.ctx.num_cols_x
        categorical_columns = self.ctx.cat_cols_x

        # During CV each fold refits this encoder on only its training rows, so a
        # validation fold can hold high-cardinality levels that fold never saw.
        # The default handle_unknown="error" would raise on them; map unseen levels
        # to -1 instead (it becomes its own ``category`` below, which LightGBM bins
        # like any other level). Mirrors the "ordinal" strategy in encoders.py.
        cat_prep = Pipeline(
            [
                ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ("to_cat", FunctionTransformer(lambda X: X.astype("category"))),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("passthrough", "passthrough")]), numerical_columns),
                ("cat", cat_prep, categorical_columns),
            ]
        ).set_output(transform="pandas")

        pipe = Pipeline([("preprocessor", preprocessor), ("model", LGBMClassifier())])
        pipe.set_params(**param)
        return pipe
