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

    def generate_distribution(self, complexity):
        k = complexity
        return {
            "model__verbose": -1,
            "model__random_state": self.ctx.config.seed,
            "model__n_jobs": 1,
            "model__n_estimators": randint(int(50 * k), int(150 * k)),
            "model__learning_rate": uniform(0.001, max(0.001, 0.1 / k)),
            "model__max_depth": randint(max(2, int(3 * k)), max(3, int(10 * k))),
            "model__num_leaves": randint(max(2, int(20 * k)), max(3, int(150 * k))),
            "model__min_child_samples": randint(1, max(2, int(50 / k))),
            "model__subsample": uniform(0.5, 0.5),
            "model__subsample_freq": randint(1, 10),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__reg_alpha": loguniform(max(1e-8, 1e-5 / k), max(1e-7, 1.0 / k)),
            "model__reg_lambda": uniform(0.0, max(0.01, 10.0 / k)),
            "model__class_weights": "balanced",
        }

    def build_pipeline(self, param):
        from lightgbm import LGBMClassifier
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

        numerical_columns = self.ctx.num_cols_x
        categorical_columns = self.ctx.cat_cols_x

        cat_prep = Pipeline(
            [
                ("ordinal", OrdinalEncoder()),
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
