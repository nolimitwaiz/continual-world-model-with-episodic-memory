"""Preparation-only paired endpoint arithmetic; never grants G1 or test access.

Inputs must come from a separately sealed and authorized evaluator. This module
does not generate worlds, load checkpoints, train, access files, or contact Slurm.
Its strict rectangular cohort keeps all four questions from an episode together
when resampling episodes within family/lag strata.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Mapping, Sequence

from cmwm.evaluation.conformance import PRIMARY_TASKS
from cmwm.evaluation.replication import EXPECTED_SEEDS, paired_development_effect
from cmwm.schemas import DatasetSplit, EntityKind, Episode


FAMILIES = ("HF-A", "HF-B")
LAGS = (64, 128)
CONDITIONS = (
    "context_only", "frame_memory", "fixed_chunks", "random_matched", "oracle_events"
)


def validate_heldout_episode_structure(
    episode: Episode, *, approved_templates: Mapping[str, tuple[str, ...]]
) -> None:
    """Reject legacy one-object test worlds; not a replacement for the full G0 audit.

    The template allowlist must come from the future frozen protocol, not from
    observed test scores. No such approved production allowlist exists yet.
    """
    if set(approved_templates) != set(FAMILIES) or any(not x for x in approved_templates.values()):
        raise ValueError("both independently reviewed held-out template families are required")
    meta = episode.split_metadata
    if meta.dataset_split is not DatasetSplit.TEST or meta.held_out_family not in FAMILIES:
        raise ValueError("expected a declared held-out test family")
    if meta.template_id not in approved_templates[meta.held_out_family]:
        raise ValueError("unapproved held-out template; never silently reuse legacy templates")
    entities = episode.initial_state.entities
    if len({entity.entity_id for entity in entities}) != len(entities):
        raise ValueError("entity IDs must be unique")
    counts = Counter(entity.kind for entity in entities)
    if counts != {EntityKind.OBJECT: 2, EntityKind.CONTAINER: 2}:
        raise ValueError("held-out worlds must preserve two-object/two-container confusability")


@dataclass(frozen=True)
class ConfirmationQuery:
    query_id: str
    episode_id: str
    family: str
    task: str
    lag: int
    answer_sha256: str


@dataclass(frozen=True)
class ConfirmationPrediction:
    seed: int
    condition: str
    query: ConfirmationQuery
    correct: bool
    checkpoint_step: int = 1000


def validate_cohort(
    queries: Sequence[ConfirmationQuery], *, examples_per_cell: int
) -> dict[tuple[str, int], tuple[str, ...]]:
    """Require all 16 cells and exactly four task queries per episode.

    Family/lag episode strata prevent empty-cell bootstrap draws. This is an
    explicit proposed endpoint restriction, not a claim about current test data.
    """
    if type(examples_per_cell) is not int or examples_per_cell < 2:
        raise ValueError("at least two independent episodes per cell are required")
    if len({q.query_id for q in queries}) != len(queries):
        raise ValueError("duplicate query identity")
    by_episode: dict[str, list[ConfirmationQuery]] = defaultdict(list)
    counts: Counter = Counter()
    for q in queries:
        if not q.query_id or not q.episode_id:
            raise ValueError("query and episode IDs cannot be empty")
        if q.family not in FAMILIES or q.task not in PRIMARY_TASKS or q.lag not in LAGS:
            raise ValueError("unknown family, task, or lag")
        if len(q.answer_sha256) != 64 or any(c not in "0123456789abcdef" for c in q.answer_sha256):
            raise ValueError("answer identity must be a lowercase SHA256")
        counts[(q.family, q.task, q.lag)] += 1
        by_episode[q.episode_id].append(q)
    expected = {(f, t, lag): examples_per_cell for f in FAMILIES for t in PRIMARY_TASKS for lag in LAGS}
    if dict(counts) != expected:
        raise ValueError("cohort is incomplete or unbalanced across the 16 endpoint cells")
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for episode_id, group in by_episode.items():
        if len(group) != len(PRIMARY_TASKS) or {q.task for q in group} != set(PRIMARY_TASKS):
            raise ValueError("each episode must supply exactly one query per task")
        if len({(q.family, q.lag) for q in group}) != 1:
            raise ValueError("an episode cannot cross family/lag strata")
        strata[(group[0].family, group[0].lag)].append(episode_id)
    return {key: tuple(sorted(value)) for key, value in sorted(strata.items())}


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lo = math.floor(position)
    hi = math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize_confirmation(
    rows: Iterable[ConfirmationPrediction],
    *,
    cohort: Sequence[ConfirmationQuery],
    examples_per_cell: int = 64,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 20260930,
) -> dict:
    """Paired statistics only. Positive numbers cannot authorize a scientific gate.

    All five conditions, five fixed fitted seeds and identical gold/query
    identities are mandatory. The primary comparator is fixed chunks, selected
    from E3 development, not whichever baseline looks weakest on held-out data.
    Both fixed and random contrasts are always returned.
    """
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 2:
        raise ValueError("bootstrap_replicates must be an integer >= 2")
    strata = validate_cohort(cohort, examples_per_cell=examples_per_cell)
    identities = {q.query_id: q for q in cohort}
    indexed: dict[tuple[int, str, str], int] = {}
    for row in rows:
        if row.seed not in EXPECTED_SEEDS or row.condition not in CONDITIONS:
            raise ValueError("unexpected condition or fitted seed")
        if type(row.correct) is not bool or row.checkpoint_step != 1000:
            raise ValueError("only boolean correctness at the fixed final checkpoint is accepted")
        if identities.get(row.query.query_id) != row.query:
            raise ValueError("prediction and sealed cohort identities disagree")
        key = (row.seed, row.condition, row.query.query_id)
        if key in indexed:
            raise ValueError("duplicate prediction")
        indexed[key] = int(row.correct)
    if len(indexed) != len(EXPECTED_SEEDS) * len(CONDITIONS) * len(cohort):
        raise ValueError("missing paired predictions; incomplete runs cannot be scored")

    queries_by_episode: dict[str, list[ConfirmationQuery]] = defaultdict(list)
    for query in cohort:
        queries_by_episode[query.episode_id].append(query)
    # The validated rectangular design makes these nested means exactly the
    # equal-weight task/lag/family macro, not a sample-size-weighted pooled score.
    episode_scores = {
        (seed, condition, episode): fmean(indexed[(seed, condition, q.query_id)] for q in group)
        for seed in EXPECTED_SEEDS
        for condition in CONDITIONS
        for episode, group in queries_by_episode.items()
    }
    scores = {
        condition: [
            fmean(
                fmean(episode_scores[(seed, condition, episode)] for episode in episodes)
                for episodes in strata.values()
            )
            for seed in EXPECTED_SEEDS
        ]
        for condition in CONDITIONS
    }
    task_lag_family = {
        condition: {
            f"{family}/{task}@{lag}": fmean(
                indexed[(seed, condition, q.query_id)]
                for seed in EXPECTED_SEEDS
                for q in cohort if (q.family, q.task, q.lag) == (family, task, lag)
            )
            for family in FAMILIES for task in PRIMARY_TASKS for lag in LAGS
        }
        for condition in CONDITIONS
    }
    contrasts = {}
    for baseline in ("fixed_chunks", "random_matched"):
        rng = random.Random(bootstrap_seed)
        draws = []
        for _ in range(bootstrap_replicates):
            seed_draw = rng.choices(EXPECTED_SEEDS, k=len(EXPECTED_SEEDS))
            seed_effects = []
            for seed in seed_draw:
                stratum_effects = []
                for episodes in strata.values():
                    picked = rng.choices(episodes, k=len(episodes))
                    stratum_effects.append(fmean(
                        episode_scores[(seed, "oracle_events", episode)]
                        - episode_scores[(seed, baseline, episode)] for episode in picked
                    ))
                seed_effects.append(fmean(stratum_effects))
            draws.append(fmean(seed_effects))
        sensitivity = paired_development_effect(scores[baseline], scores["oracle_events"])
        family_deltas = {
            family: fmean(
                episode_scores[(seed, "oracle_events", episode)]
                - episode_scores[(seed, baseline, episode)]
                for seed in EXPECTED_SEEDS
                for (f, _), episodes in strata.items() if f == family
                for episode in episodes
            )
            for family in FAMILIES
        }
        low, high = _quantile(draws, .025), _quantile(draws, .975)
        contrasts[baseline] = {
            "mean_delta": sensitivity["mean_delta"],
            "relative_error_reduction": sensitivity["relative_error_reduction"],
            "paired_hierarchical_ci_95": [low, high],
            "paired_seed_t_sensitivity": sensitivity,
            "family_deltas": family_deltas,
            "numerical_checks_only": {
                "positive_hierarchical_lower_bound": low > 0,
                "practical_effect": sensitivity["mean_delta"] >= .05
                or sensitivity["relative_error_reduction"] >= .20,
                "both_families_positive": all(x > 0 for x in family_deltas.values()),
                "four_of_five_seeds_positive": sensitivity["positive_seed_count"] >= 4,
            },
        }
    return {
        "artifact_type": "cmwm_confirmation_statistics_preparation_v1",
        "claim_eligible": False,
        "g1_decision": "not_assessed_requires_independent_provenance_and_budget_audits",
        "primary_baseline": "fixed_chunks",
        "mandatory_secondary_baseline": "random_matched",
        "seeds": list(EXPECTED_SEEDS),
        "unique_queries": len(cohort),
        "unique_episodes": len(queries_by_episode),
        "scores_by_condition": scores,
        "task_lag_family_means": task_lag_family,
        "contrasts": contrasts,
        "bootstrap": {
            "replicates": bootstrap_replicates, "seed": bootstrap_seed,
            "method": "paired seeds then episodes within family/lag strata; all tasks kept together",
        },
        "limitations": [
            "Statistics do not validate source, dataset novelty, budget equality or test authorization.",
            "The fixed two generator families are not a sample of all possible worlds.",
            "The nested interval follows the existing protocol; shared-world cross-seed dependence needs sensitivity review.",
        ],
    }
