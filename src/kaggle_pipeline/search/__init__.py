"""Model search: shared cross-validation primitives used by the evolutionary trainer."""

from kaggle_pipeline.search.cv import CrossValScore, make_cv_splitter

__all__ = ["CrossValScore", "make_cv_splitter"]
