"""Baselines (owner: Person C). Every baseline exposes ``fit`` / ``predict`` / ``predict_cate``."""

from dopfn.baselines.simple import GroupMeanDifference, ObservationalRegression

__all__ = ["GroupMeanDifference", "ObservationalRegression"]
