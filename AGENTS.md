# Optimization Engine Codex Instructions

Before planning, answering, or changing files:

1. Read `governance/MASTER_PROJECT_CONTEXT.md` completely.
2. Read `governance/DELTA_NEUTRAL_HEDGING_SPEC.md` completely.
3. Read `governance/CURRENT_STATUS.md` and `governance/INTEGRATION_VERSIONS.yaml`.
4. Read this repository's `README.md`.
5. Verify the pinned submodule with `git submodule status governance`.
6. Report the verification block below at the beginning of a new Codex chat.

```text
Governance verification:
- Governance commit:
- System version:
- Component: Optimization Engine
- Delivery phase: Phase 5
- Relevant acceptance gate: A frozen configuration passes walk-forward evaluation after costs while respecting drawdown, liquidity, turnover, and risk constraints.
```

Stop if `governance/MASTER_PROJECT_CONTEXT.md` is missing or unreadable. Use the shared simulator, chronological validation, and constrained or multi-objective studies; never optimize raw return alone.

Consume forecasts without changing their meaning. Apply costs, constraints, hedge bands, timing, sizing, and economic objectives only in the shared simulation/optimization layer. Never duplicate or redefine market-data timestamps, Greek units, delta aggregation, hedge quantity, fills, or P&L accounting.

