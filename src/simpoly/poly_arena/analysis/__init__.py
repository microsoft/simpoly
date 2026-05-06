"""Tg analysis package: cooling-scan + 21-step protocols."""

from .bootstrap import bootstrap_tgs
from .experimental_values import EXP_RHO_300K, EXP_TG
from .fitting import patrone_equation, self_consistent_fit, wrapped_patrone_fit
from .pipeline import aggregate_seed_tgs, fit_dataset

__all__ = [
    "EXP_TG",
    "EXP_RHO_300K",
    "patrone_equation",
    "wrapped_patrone_fit",
    "self_consistent_fit",
    "fit_dataset",
    "aggregate_seed_tgs",
    "bootstrap_tgs",
]
