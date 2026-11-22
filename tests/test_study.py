import json

from datetime import datetime, timedelta, timezone

import pytest

from optimization_engine.study import (
    Constraints,
    CostScenario,
    Fold,
    Metrics,
    Policy,
    Provenance,
    Window,
    evaluate_holdout,
    walk_forward,
)


def window(start, end):
    origin = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return Window(origin + timedelta(days=start), origin + timedelta(days=end))


BASELINE = Policy(30, 1, 20, 45, 5, 10)
WINNER = Policy(30, 1, 20, 45, 5, 20)
RISKY = Policy(30, 1, 20, 45, 5, 30)
FOLDS = (Fold(window(0, 3), window(4, 6), window(7, 9)),
         Fold(window(4, 8), window(9, 11), window(12, 14)))


class FixtureSimulator:
    provenance = Provenance("code", "governance", "sim", "risk", "data",
                            "forecast", "cost", "environment", "strategy", "capital")

    def __init__(self):
        self.calls = []

    def evaluate(self, policy, interval, scenario, seed):
        self.calls.append((policy, interval, scenario, seed))
        return Metrics(policy.delta_band_shares / 100, 0.05, 1, 0, int(policy == RISKY), True, True)


def run(simulator, folds=FOLDS, checkpoint=None):
    return walk_forward(
        simulator, candidates=(RISKY, WINNER, BASELINE), baseline=BASELINE,
        folds=folds, constraints=Constraints(0.10, 2), embargo=timedelta(days=1),
        shortlist_size=3, seed=42, checkpoint=checkpoint,
    )


def test_constrained_search_freezes_validation_winner_and_reports_neighbors():
    simulator = FixtureSimulator()
    result = run(simulator)
    assert result == run(FixtureSimulator())
    assert result.frozen.policy == WINNER
    manifest = json.loads(result.to_json())
    assert manifest['frozen']['policy']['delta_band_shares'] == 20
    assert manifest['embargo'] == 86400
    for report in result.folds:
        assert {policy for policy, _ in report.validation} == {BASELINE, WINNER}
        assert set(report.test.metrics) == set(CostScenario)
        assert report.baseline.policy == BASELINE
        assert [entry.policy for entry in report.neighbors] == [BASELINE, RISKY]
        assert not result.frozen.constraints.accepts(report.neighbors[1].metrics[CostScenario.BASE])
    holdout = window(15, 18)
    assert all(interval != holdout for _, interval, _, _ in simulator.calls)
    candidate, baseline = evaluate_holdout(simulator, result.frozen, holdout)
    assert (candidate.policy, baseline.policy) == (WINNER, BASELINE)


def test_blocks_missing_embargo_before_simulator_reads_data():
    simulator = FixtureSimulator()
    invalid = Fold(window(0, 4), window(4, 6), window(7, 9))
    with pytest.raises(ValueError, match="embargo"):
        run(simulator, (invalid, FOLDS[1]))
    assert simulator.calls == []


def test_blocks_overlapping_holdout():
    simulator = FixtureSimulator()
    result = run(simulator)
    with pytest.raises(ValueError, match="final test window"):
        evaluate_holdout(simulator, result.frozen, window(14, 16))


@pytest.mark.parametrize("changes", [
    (0, 1, 20, 45, 5, 10),
    (30, 0, 20, 45, 5, 10),
    (30, 1, 20, 45, 20, 10),
    (30, 1, 20, 45, 5, float("nan")),
])
def test_invalid_policy_is_rejected(changes):
    with pytest.raises(ValueError):
        Policy(*changes)


def test_resume_reuses_completed_evaluations(tmp_path):
    class InterruptedSimulator(FixtureSimulator):
        def evaluate(self, policy, interval, scenario, seed):
            if len(self.calls) == 2:
                raise RuntimeError("interrupted")
            return super().evaluate(policy, interval, scenario, seed)

    checkpoint = tmp_path / "experiment.sqlite"
    first = InterruptedSimulator()
    with pytest.raises(RuntimeError, match="interrupted"):
        run(first, checkpoint=checkpoint)
    resumed = FixtureSimulator()
    result = run(resumed, checkpoint=checkpoint)
    assert result == run(FixtureSimulator())
    assert set(first.calls).isdisjoint(resumed.calls)


def test_constraints_reject_incomplete_or_unreconciled_results():
    constraints = Constraints(0.10, 2)
    metrics = Metrics(1, 0.10, 2, 0, 0, False, False)
    assert constraints.violations(metrics) == ("accounting", "data_coverage")


def test_validation_selects_before_test_and_test_cannot_replace_winner():
    class ChangingReturns(FixtureSimulator):
        def evaluate(self, policy, interval, scenario, seed):
            metrics = super().evaluate(policy, interval, scenario, seed)
            if interval in {fold.validation for fold in FOLDS} and policy == BASELINE:
                return Metrics(0.40, 0.05, 1, 0, 0, True, True)
            return metrics

    result = run(ChangingReturns())
    assert result.frozen.policy == BASELINE
    for report in result.folds:
        assert report.selected == BASELINE
        assert report.neighbors[0].policy == WINNER
        assert (
            report.neighbors[0].metrics[CostScenario.BASE].net_return
            > report.test.metrics[CostScenario.BASE].net_return
        )
