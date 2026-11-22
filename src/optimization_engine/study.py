"""Versioned research interface for policy search and out-of-sample evaluation."""

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Mapping, Protocol

import optuna


@dataclass(frozen=True, order=True)
class Policy:
    hedge_check_minutes: int
    position_contracts: int
    min_entry_dte: int
    max_entry_dte: int
    exit_dte: int
    delta_band_shares: float

    def __post_init__(self) -> None:
        integers = fields(self)[:-1]
        if any(type(getattr(self, field.name)) is not int for field in integers):
            raise ValueError("Timing, position size, and DTE must be integers")
        if self.hedge_check_minutes < 1 or self.position_contracts < 1:
            raise ValueError("Hedge interval and position size must be positive")
        if not 0 <= self.exit_dte < self.min_entry_dte <= self.max_entry_dte:
            raise ValueError("Require 0 <= exit DTE < minimum entry DTE <= maximum entry DTE")
        if not isfinite(self.delta_band_shares) or self.delta_band_shares < 0:
            raise ValueError("Delta band must be finite and nonnegative")


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if any(bound.utcoffset() != timedelta(0) for bound in (self.start, self.end)):
            raise ValueError("Window endpoints must use UTC")
        if self.start >= self.end:
            raise ValueError("Window must have positive duration")


@dataclass(frozen=True)
class Fold:
    train: Window
    validation: Window
    test: Window


class CostScenario(str, Enum):
    BASE = "base"
    FAVORABLE = "favorable"
    ADVERSE = "adverse"


@dataclass(frozen=True)
class Metrics:
    net_return: float
    drawdown: float
    turnover: float
    liquidity_violations: int
    risk_breaches: int
    accounting_reconciled: bool
    data_complete: bool

    def __post_init__(self) -> None:
        if not isfinite(self.net_return):
            raise ValueError("Net return must be finite")
        if not isfinite(self.drawdown) or not 0 <= self.drawdown <= 1:
            raise ValueError("Drawdown must be a fraction from 0 to 1")
        if not isfinite(self.turnover) or self.turnover < 0:
            raise ValueError("Turnover must be finite and nonnegative")
        for count in (self.liquidity_violations, self.risk_breaches):
            if type(count) is not int or count < 0:
                raise ValueError("Violation counts must be nonnegative integers")


@dataclass(frozen=True)
class Constraints:
    max_drawdown: float
    max_turnover: float

    def __post_init__(self) -> None:
        Metrics(0, self.max_drawdown, self.max_turnover, 0, 0, True, True)

    def violations(self, metrics: Metrics) -> tuple[str, ...]:
        checks = {
            "drawdown": metrics.drawdown > self.max_drawdown,
            "turnover": metrics.turnover > self.max_turnover,
            "liquidity": metrics.liquidity_violations > 0,
            "risk": metrics.risk_breaches > 0,
            "accounting": metrics.accounting_reconciled is not True,
            "data_coverage": metrics.data_complete is not True,
        }
        return tuple(reason for reason, failed in checks.items() if failed)

    def accepts(self, metrics: Metrics) -> bool:
        return not self.violations(metrics)


@dataclass(frozen=True)
class Provenance:
    code_revision: str
    governance_revision: str
    simulator_version: str
    risk_version: str
    dataset_version: str
    forecast_version: str
    cost_model_version: str
    environment_version: str
    strategy_version: str
    capital_config_version: str
    schema_version: str = "1"


class SharedSimulator(Protocol):
    """Run each [start, end) window independently and point-in-time.

    The adapter owns isolated initial state, input availability and boundary
    handling, forecast consumption, canonical accounting, and independent risk
    decisions. Use the same seeds and assumptions for matched comparisons.
    """

    @property
    def provenance(self) -> Provenance: ...

    def evaluate(
        self, policy: Policy, window: Window, scenario: CostScenario, seed: int
    ) -> Metrics: ...


@dataclass(frozen=True)
class TestResult:
    policy: Policy
    metrics: Mapping[CostScenario, Metrics]


@dataclass(frozen=True)
class FoldResult:
    fold: Fold
    selected: Policy
    training: tuple[tuple[Policy, Metrics], ...]
    validation: tuple[tuple[Policy, Metrics], ...]
    test: TestResult
    baseline: TestResult
    neighbors: tuple[TestResult, ...]


@dataclass(frozen=True)
class FrozenCandidate:
    policy: Policy
    baseline: Policy
    constraints: Constraints
    seed: int
    provenance: Provenance
    earliest_holdout_start: datetime


@dataclass(frozen=True)
class StudyResult:
    frozen: FrozenCandidate
    folds: tuple[FoldResult, ...]
    candidates: tuple[Policy, ...]
    shortlist_size: int
    embargo: timedelta
    optuna_version: str
    objective_version: str = "constrained-net-return-v1"

    def to_json(self) -> str:
        """Serialize settings and evidence for the experiment registry."""
        def encode(value: object) -> str | float:
            if isinstance(value, datetime):
                return value.isoformat()
            if isinstance(value, timedelta):
                return value.total_seconds()
            raise TypeError(f"Cannot serialize {type(value).__name__}")

        return json.dumps(asdict(self), default=encode, indent=2, allow_nan=False)


@dataclass(frozen=True)
class _CheckpointedSimulator:
    simulator: SharedSimulator
    checkpoint: Path

    @property
    def provenance(self) -> Provenance:
        return self.simulator.provenance

    def evaluate(
        self, policy: Policy, window: Window, scenario: CostScenario, seed: int
    ) -> Metrics:
        key = json.dumps({
            "provenance": asdict(self.provenance),
            "policy": asdict(policy),
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "scenario": scenario.value,
            "seed": seed,
        }, sort_keys=True)
        with closing(sqlite3.connect(self.checkpoint)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS evaluations_v1 "
                "(inputs TEXT PRIMARY KEY, metrics TEXT NOT NULL)"
            )
            row = connection.execute(
                "SELECT metrics FROM evaluations_v1 WHERE inputs = ?", (key,)
            ).fetchone()
            if row is not None:
                return Metrics(**json.loads(row[0]))
            metrics = self.simulator.evaluate(policy, window, scenario, seed)
            connection.execute(
                "INSERT OR IGNORE INTO evaluations_v1 VALUES (?, ?)",
                (key, json.dumps(asdict(metrics), allow_nan=False)),
            )
            return metrics


def _evaluate_costs(
    simulator: SharedSimulator, policy: Policy, window: Window, seed: int
) -> TestResult:
    return TestResult(policy, {
        scenario: simulator.evaluate(policy, window, scenario, seed)
        for scenario in CostScenario
    })


def _neighbors(policy: Policy, candidates: tuple[Policy, ...]) -> tuple[Policy, ...]:
    neighbors = set()
    for field in fields(policy):
        line = sorted(
            (other for other in candidates if all(
                getattr(other, axis.name) == getattr(policy, axis.name)
                for axis in fields(policy) if axis != field
            )),
            key=lambda other: getattr(other, field.name),
        )
        index = line.index(policy)
        neighbors.update(line[max(0, index - 1):index + 2])
    neighbors.remove(policy)
    return tuple(sorted(neighbors))


def walk_forward(
    simulator: SharedSimulator,
    *,
    candidates: tuple[Policy, ...],
    baseline: Policy,
    folds: tuple[Fold, ...],
    constraints: Constraints,
    embargo: timedelta,
    shortlist_size: int,
    seed: int,
    checkpoint: Path | None = None,
) -> StudyResult:
    """Train a constrained Optuna grid per fold, select on validation, then test."""
    candidates = tuple(sorted(set(candidates)))
    if len(folds) < 2 or not candidates or shortlist_size < 1:
        raise ValueError("Provide multiple folds, candidates, and a positive shortlist size")
    if embargo < timedelta(0) or not 0 <= seed < 2**32:
        raise ValueError("Embargo must be nonnegative and seed must be an unsigned 32-bit integer")
    for fold in folds:
        if (fold.validation.start - fold.train.end < embargo
                or fold.test.start - fold.validation.end < embargo):
            raise ValueError("Require chronological train, validation, and test with embargo")
    for previous, current in zip(folds, folds[1:]):
        if previous.test.end > current.test.start:
            raise ValueError("Test windows must be ordered and nonoverlapping")

    provenance = simulator.provenance
    if provenance.schema_version != "1" or any(
        not getattr(provenance, field.name).strip() for field in fields(provenance)
    ):
        raise ValueError("Require complete simulator provenance using schema version 1")

    if checkpoint is not None:
        simulator = _CheckpointedSimulator(simulator, checkpoint)

    reports = []
    for fold in folds:
        training: dict[Policy, Metrics] = {}
        indices = list(range(len(candidates)))
        sampler = optuna.samplers.GridSampler({"policy": indices}, seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            policy = candidates[trial.suggest_categorical("policy", indices)]
            metrics = simulator.evaluate(policy, fold.train, CostScenario.BASE, seed)
            training[policy] = metrics
            if not constraints.accepts(metrics):
                raise optuna.TrialPruned("Training constraints failed")
            return metrics.net_return

        study.optimize(objective, n_trials=len(indices))
        eligible = sorted(
            (policy for policy, metrics in training.items() if constraints.accepts(metrics)),
            key=lambda policy: (-training[policy].net_return, policy),
        )
        validation = tuple(
            (policy, simulator.evaluate(policy, fold.validation, CostScenario.BASE, seed))
            for policy in eligible[:shortlist_size]
        )
        feasible = [
            (policy, metrics) for policy, metrics in validation if constraints.accepts(metrics)
        ]
        if not feasible:
            raise ValueError(f"No feasible validation policy for fold starting {fold.test.start}")
        selected, _ = min(feasible, key=lambda entry: (-entry[1].net_return, entry[0]))
        reports.append(FoldResult(
            fold, selected, tuple(sorted(training.items())), validation,
            _evaluate_costs(simulator, selected, fold.test, seed),
            _evaluate_costs(simulator, baseline, fold.test, seed),
            tuple(
                _evaluate_costs(simulator, neighbor, fold.test, seed)
                for neighbor in _neighbors(selected, candidates)
            ),
        ))

    last = reports[-1]
    frozen = FrozenCandidate(
        last.selected, baseline, constraints, seed, provenance, last.fold.test.end + embargo,
    )
    return StudyResult(frozen, tuple(reports), candidates, shortlist_size, embargo, optuna.__version__)


def evaluate_holdout(
    simulator: SharedSimulator, frozen: FrozenCandidate, window: Window
) -> tuple[TestResult, TestResult]:
    """Evaluate an externally sealed window after recording the frozen candidate."""
    if window.start < frozen.earliest_holdout_start:
        raise ValueError("Holdout must begin after the final test window and embargo")
    if simulator.provenance != frozen.provenance:
        raise ValueError("Holdout provenance does not match the frozen experiment")
    return (
        _evaluate_costs(simulator, frozen.policy, window, frozen.seed),
        _evaluate_costs(simulator, frozen.baseline, window, frozen.seed),
    )
