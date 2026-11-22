# Optimization Engine

Constrained, walk-forward strategy and risk-parameter optimization for the Denson Group options trading system. It consumes forecasts and applies downstream cost, hedge, sizing, and constraint logic through the shared simulator without changing prediction targets or canonical calculations.

The authoritative system context is pinned under [`governance/`](./governance/). Start by reading [`AGENTS.md`](./AGENTS.md).


## Phase 5 research workflow

The Python 3.12 library searches an explicit grid of approved candidate policies.
Each policy specifies the combined scheduled delta-band check interval in minutes,
position size in contracts (strategy-defined size applied by the simulator),
entry DTE range, exit DTE, and the soft delta band in share equivalents. The
adapter applies the governed calendar conventions, trigger behavior and strategy
rules. Independent hard risk limits are part of the versioned risk configuration.

Install with `python -m pip install -e '.[test]'` and run `python -m pytest`.

Use `optimization_engine.study.walk_forward` with a versioned
`SharedSimulator` adapter, a candidate grid, a baseline policy, explicit UTC
train/validation/test windows, risk constraints, an embargo, a shortlist size,
and a seed. Each simulator call uses the half-open interval `[start, end)`. Set
the embargo to cover the longest information horizon shared across adjacent
splits. The adapter must enforce point-in-time input availability, purge
observations whose outcomes cross a window boundary, and use identical capital,
state initialization, seeds, and scenario assumptions for matched comparisons.

Optuna ranks candidates by training net return subject to drawdown, turnover,
and zero liquidity violations or risk breaches. Incomplete data or unreconciled
accounting also make a trial ineligible. The feasible validation winner
then runs on the test window alongside the baseline and neighboring grid points
under base, favorable, and adverse cost profiles. Net return and drawdown are
fractions, and turnover is traded notional divided by capital under the adapter's
versioned measurement convention. Baseline is a deterministic policy supported
by the same adapter. Full benchmark comparisons and ledger attribution should
come from the shared simulator's reporting interface.

The study result records trial metrics, fold windows, matched test results,
provenance, seed, candidate grid, and the final validation winner as the frozen
candidate. Use `StudyResult.to_json()` to archive it before releasing the sealed holdout.
Pass a local SQLite `checkpoint` path to resume completed simulator evaluations
with matching provenance and inputs after an interruption. Call
`evaluate_holdout` from the holdout owner’s workflow with that frozen candidate
and matching provenance. The holdout owner controls one-time access. A completed
study is research evidence for review. Promotion requires demonstrating
persistent improvement, parameter stability, and compliance under the governed
evaluation protocol.

The integration contract is defined here. This checkout still requires the
canonical simulator adapter, validated deterministic baselines, pinned market
data and forecast inputs, and independent risk integration for an end-to-end
historical run.
