"""Random forest classifier with one-hot encoded categoricals."""

from __future__ import annotations

from scipy.stats import randint
from sklearn.pipeline import Pipeline

from kaggle_pipeline.models.base import Model
from kaggle_pipeline.models.registry import register_model


@register_model(
    name="RandomForestClassifier", purposes="single_target_prob_pred", lower=3, upper=10
)
class RandomForestClassifierModel(Model):
    def generate_distribution(self, complexity):
        k = complexity
        return {
            "model__n_jobs": 1,
            "model__random_state": self.ctx.config.seed,
            "model__n_estimators": randint(int(100 * k), int(200 * k)),
            "model__max_depth": randint(int(5 * k), int(20 * k)),
            "model__min_samples_split": randint(2, max(3, int(20 / k))),
            "model__min_samples_leaf": randint(1, max(2, int(5 / k))),
            "model__max_features": ["sqrt", "log2"],
            "model__class_weight": "balanced_subsample",
        }

    def build_pipeline(self, param):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import OneHotEncoder

        numerical_columns = self.ctx.num_cols_x
        categorical_columns = self.ctx.cat_cols_x

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("passthrough", "passthrough")]), numerical_columns),
                (
                    "cat",
                    Pipeline([("ohe", OneHotEncoder(handle_unknown="error"))]),
                    categorical_columns,
                ),
            ]
        )

        pipe = Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier())])
        pipe.set_params(**param)
        return pipe
