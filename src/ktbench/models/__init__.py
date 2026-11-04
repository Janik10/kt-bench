"""Model registry. Imports are lazy so the baselines run without torch or pyBKT loaded."""

from __future__ import annotations

from importlib import import_module

from ktbench.models.base import KTModel

REGISTRY = {
    "skill_avg": "ktbench.models.skill_avg:SkillAverage",
    "bkt": "ktbench.models.bkt:BKT",
    "pfa": "ktbench.models.pfa:PFA",
    "beta": "ktbench.models.beta:BetaPosterior",
    "dkt": "ktbench.models.dkt:DKT",
    "sakt": "ktbench.models.sakt:SAKT",
}


def get(name: str) -> type[KTModel]:
    module, cls = REGISTRY[name].split(":")
    return getattr(import_module(module), cls)


def build(name: str, n_skills: int, n_items: int, **params) -> KTModel:
    return get(name)(n_skills=n_skills, n_items=n_items, **params)
