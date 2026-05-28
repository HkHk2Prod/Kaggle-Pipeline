"""Random forest classifier with one-hot encoded categoricals."""

from __future__ import annotations

from scipy.stats import randint
from sklearn.pipeline import Pipeline

from kaggle_pipeline.models.base import Model
from kaggle_pipeline.models.registry import register_model


@register_model(
    name="RandomForestClassifier", purposes="single_target_prob_pred", lower=0.03, upper=0.10
)
class RandomForestClassifierModel(Model):
    def generate_distribution(self):
        # Wide fixed ranges: shallow stubs through deep, near-unconstrained trees;
        # `max_features` mixes the classic heuristics with a few fractions so the
        # spectrum spans high-bias to high-variance models.
        return {
            "model__n_jobs": 1,
            "model__random_state": self.ctx.config.seed,
            "model__n_estimators": randint(100, 600),
            "model__max_depth": randint(2, 40),
            "model__min_samples_split": randint(2, 40),
            "model__min_samples_leaf": randint(1, 30),
            "model__max_features": ["sqrt", "log2", 0.3, 0.6, 1.0],
            "model__class_weight": "balanced_subsample",
        }

    def build_pipeline(self, param):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier

        from kaggle_pipeline.preprocessing import categorical_transformer_specs

        numerical_columns = self.ctx.num_cols_x
        categorical_columns = self.ctx.cat_cols_x

        # RandomForest cannot consume raw categoricals, so each is encoded per the
        # run's resolved plan (default: frequency) instead of one-hot -- which
        # avoids exploding on high-cardinality columns and tolerates unseen levels.
        cat_specs = categorical_transformer_specs(
            self.ctx.categorical_encoding, categorical_columns
        )
        preprocessor = ColumnTransformer(
            transformers=[("num", "passthrough", numerical_columns), *cat_specs]
        )

        pipe = Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier())])
        pipe.set_params(**param)
        return pipe
