# MEIRP Execution Plan

Date: 2026-07-04

This plan starts from the current `meirp_project` state and proceeds to a
complete MEIRP experiment suite.  Inventory dynamics must remain exactly aligned
with the existing project; routing is added around the current inventory model,
not by replacing it with the old parent-project environment.

## Execution Status

Status on 2026-07-04: stages 0 through 7 are implemented and verified.

- Stage 0 created and verified the local restore snapshot listed below.
- Stage 1 produced the parent-code audit and per-file marking table.
- Stage 2 added routing-aware environment modeling while preserving inventory
  dynamics when routing transport cost is zero.
- Stage 3 added heuristic inventory plus heuristic routing baselines.
- Stage 4 added GA/ALNS inventory solvers composed with heuristic routing.
- Stage 5 added GA routing and integrated `ALNS-IRP`.
- Stage 6 added an OR-Tools rolling-horizon solver.  It is covered by unit
  tests and kept opt-in in `full_compare.py` via `--include-ortools`, because
  repeated CP-SAT calls caused a native Windows access violation in the full
  experiment process.
- Stage 7 unified `full_compare.py` JSON/CSV output with per-seed raw records,
  aggregate metrics, preserved legacy metrics, route metrics, and runtimes.

Latest green gate:

- `python -m py_compile core\config.py core\env.py evaluation\metrics.py baselines\base.py baselines\routed.py routing\*.py solvers\*.py scripts\full_compare.py`
- `python tests\test_core.py`: 32 passed
- `python tests\test_baselines.py`: 37 passed
- `python tests\test_evaluation.py`: 19 passed
- `python tests\test_bdq.py`: 32 passed
- `python -m pytest tests\test_routing.py tests\test_metaheuristic_solvers.py tests\test_ortools_solver.py -v`: 14 passed
- Smoke output: `results/stage7_smoke.json` and
  `results/stage7_smoke_summary.csv`

## Return Point

Before further code changes, a verified local restore point was created:

- Snapshot: `archive/restore_points/20260704_210407/meirp_project_snapshot.zip`
- Manifest: `archive/restore_points/20260704_210407/MANIFEST.txt`
- Git status record: `archive/restore_points/20260704_210407/git_status_short.txt`
- Verification: the zip was expanded to a temp directory and required files were
  found.  Verified extracted file count: `137`.

Restore procedure if a later stage goes wrong:

1. Stop editing.
2. Extract `meirp_project_snapshot.zip` to a temporary directory.
3. Copy the extracted files back into `D:\Project\Python\MaSCL\meirp_project`.
4. Re-run the Stage 0 smoke checks below.

The parent git repository is rooted at `D:\Project\Python\MaSCL`, not at
`meirp_project`.  Because the parent worktree already contains many unrelated
changes, the initial safety point is a local project snapshot instead of a broad
git commit.

## Folder Ownership

### Active Source

| Folder | Ownership |
| --- | --- |
| `core/` | Current MEIRP environment, config, network, demand, allocation, reward. Inventory logic lives here and must remain the source of truth. |
| `routing/` | New routing package to be added: distance, route types, savings, nearest insertion, 2-opt, GA routing. No inventory state updates here. |
| `baselines/` | Simple policy baselines, especially BaseStock and `(s,S)`, plus composed inventory+routing baselines. |
| `solvers/` | Optimization and metaheuristic solvers: GA inventory, ALNS inventory, integrated ALNS/IRP, OR-Tools IRP. |
| `algorithms/` | RL algorithms only: IQL, QMIX, MAPPO, GNN variants. No classical routing code here. |
| `models/` | Neural encoders: MLP, GRU, GAT/GNN. |
| `collaboration/` | SEA/advisor/escalation code. Keep separate from baseline algorithms. |
| `evaluation/` | Metric aggregation, statistics, plots, ablation helpers. Must preserve all existing metrics. |
| `scripts/` | Supported command-line entry points only: train, evaluate, full_compare. Avoid adding one-off experiment scripts. |
| `configs/` | Current scenario configs. Routing-related config values go here after Stage 2. |
| `tests/` | Unit, integration, solver, routing, and experiment smoke tests. |

### Project Metadata / Planning

| Path | Ownership |
| --- | --- |
| `Agents.md` | Local coding instructions. |
| `PARENT_CODE_AUDIT.md` | Parent-folder audit summary. |
| `PARENT_FILE_MARKING.csv` | Full parent-folder per-file marking table. |
| `PROJECT_EXECUTION_PLAN.md` | This staged execution contract. |

### Archive / Reference

| Folder | Ownership |
| --- | --- |
| `archive/obsolete_scripts/` | Old scripts not used by the current supported workflow. |
| `archive/obsolete_unused/` | Old unused source, including archived GA/ALNS/CP-SAT references. Code can be revived only through a new active module. |
| `archive/obsolete_results/` | Old experiment logs/results. Historical only. |
| `archive/restore_points/` | Local rollback snapshots. Do not import from here. |

### Generated / Ignored

| Folder | Ownership |
| --- | --- |
| `results/` | Generated experiment output only. Do not put source code here. |
| `__pycache__/`, `.pytest_cache/`, `.idea/`, `.vscode/` | Generated/tooling artifacts. Not algorithm source. |

## Stage Gates

Each stage has an exit gate.  A later stage starts only after the current gate is
green.

### Stage 0: Safety Baseline

Goal: establish a rollback point and confirm the current project still runs.

Work:

- Create and verify the snapshot listed above.
- Record current git status.
- Run current minimal tests before routing changes.

Exit gate:

- Snapshot zip exists and expands successfully.
- Required files exist in extracted snapshot:
  `core/env.py`, `core/config.py`, `scripts/full_compare.py`,
  `baselines/base_stock.py`, `baselines/ss_policy.py`.
- Current smoke tests pass:
  - `python -m py_compile core/env.py core/config.py scripts/full_compare.py`
  - `python tests/test_core.py`
  - `python tests/test_baselines.py`
  - `python tests/test_bdq.py`

### Stage 1: Folder Classification

Goal: make module ownership explicit before adding new code.

Work:

- Keep active source in the folders listed above.
- Do not move old parent-project code directly into active folders.
- If parent code is reused, port it into a current-style module with tests.

Exit gate:

- `PROJECT_EXECUTION_PLAN.md` exists.
- `PARENT_CODE_AUDIT.md` and `PARENT_FILE_MARKING.csv` exist.
- No new active code is placed in `results/`, `archive/restore_points/`, or
  `archive/obsolete_*`.

### Stage 2: Routing-Aware Environment

Goal: add routing and transport cost without changing current inventory
dynamics.

Work:

- Add `routing/` package:
  `types.py`, `distance.py`, `two_opt.py`, `savings.py`,
  `nearest_insertion.py`, `policy.py`.
- Add routing fields to config:
  `coords`, `vehicle_capacity`, `vehicle_count`, `transport_cost`,
  `routing_enabled`.
- Extend env info/reward with:
  `routes`, `route_distance`, `transport_cost`, `total_transport_cost`.
- When `routing_enabled=False` or `transport_cost=0`, existing inventory
  behavior must be numerically unchanged.

Exit gate:

- Routing unit tests pass:
  - route distance recomputation absolute error `<= 1e-6`
  - every route starts and ends at its depot
  - route demand `<= vehicle_capacity + 1e-9`
  - 2-opt never increases route distance on deterministic test routes
  - nearest insertion visits each demanded customer exactly once
- Inventory invariance test passes:
  - same seed, same actions, `routing_enabled=False`: old and new `inv`, `bl`,
    `pipe`, reward arrays match with max absolute error `<= 1e-9`
  - same seed, same actions, `routing_enabled=True` and `transport_cost=0`:
    inventory states still match with max absolute error `<= 1e-9`
- Existing tests from Stage 0 still pass.

### Stage 3: Heuristic Inventory + Heuristic Routing

Goal: cover BaseStock/`(s,S)` + savings/nearest-insertion/2-opt baselines.

Work:

- Keep `BaseStockPolicy` and `SSPolicy` order logic unchanged.
- Add composition layer for inventory policy + routing policy.
- Add full_compare names for:
  `BaseStock+Savings`, `sS+Savings`,
  `BaseStock+Savings2Opt`, `sS+Savings2Opt`,
  `BaseStock+NearestInsertion`, `sS+NearestInsertion`.

Exit gate:

- Baseline action validity:
  - all BDQ action values are integers in `[0, max_order]`
  - non-root action length equals number of upstream nodes
- Routing feasibility:
  - route distance finite and `>= 0`
  - route capacity constraints pass for every route
- Metric preservation:
  - all existing metric keys still appear in `full_compare` output
  - new route metrics appear without replacing old metric names
- Smoke experiment:
  - one small config, one seed, all Stage 3 baselines complete
  - every numeric metric is finite; fill-rate-like metrics are in `[0, 1]`

### Stage 4: Metaheuristic Inventory + Heuristic Routing

Goal: revive current-schema GA/ALNS inventory solvers, then compose them with
heuristic routing.

Work:

- Port archived `ga_solver.py` into `solvers/ga_inventory_solver.py`.
- Port archived `alns_solver.py` or old `alns2.py` ideas into
  `solvers/alns_inventory_solver.py`.
- The solver objective may simulate inventory cost, but final evaluation cost
  must come from the actual environment.

Exit gate:

- Solver validity:
  - generated actions are valid BDQ actions
  - no NaN or inf in simulated objective
  - deterministic seed produces identical first action
- Baseline dominance sanity:
  - because BaseStock is seeded into the initial population/solution, best
    internal objective must be `<= BaseStock internal objective + 1e-6`
    on the deterministic tiny test case.
- Smoke experiment:
  - `GAInventory+Savings`
  - `GAInventory+Savings2Opt`
  - `ALNSInventory+Savings`
  - `ALNSInventory+Savings2Opt`
  all complete on one small config and one seed.

### Stage 5: Metaheuristic Routing and Integrated ALNS/IRP

Goal: cover heuristic inventory + GA routing, plus an integrated ALNS/IRP
baseline.

Work:

- Add `routing/ga_routing.py`.
- Add integrated `solvers/alns_irp_solver.py`.
- Integrated ALNS objective includes holding, backlog, stockout, transport cost,
  and optional order smoothness only if explicitly configured.

Exit gate:

- GA routing feasibility:
  - visits each demanded customer exactly once
  - every route capacity constraint passes
  - final route distance `<= initial random feasible route distance + 1e-6`
    on deterministic tiny route test
- Integrated ALNS feasibility:
  - valid actions for every step in planning horizon
  - finite objective
  - best objective `<= initial solution objective + 1e-6`
- Smoke experiment:
  - `BaseStock+GARouting`
  - `sS+GARouting`
  - `ALNS-IRP`
  complete on one small config and one seed.

### Stage 6: OR-Tools Exact / Rolling-Horizon IRP

Goal: add exact or near-exact IRP baselines for small scenarios.

Work:

- Add `solvers/ortools_irp_solver.py`.
- Use OR-Tools CP-SAT as the primary exact backend.
- Keep Gurobi old code as reference only.
- Provide two modes:
  - `exact_small`: small instance, route arc variables, short horizon.
  - `rolling_horizon`: practical comparison baseline.

Exit gate:

- If OR-Tools is installed:
  - solver returns `OPTIMAL` or `FEASIBLE` on tiny exact test
  - extracted actions are valid BDQ actions
  - route capacity constraints pass
  - exact tiny objective `<= BaseStock+Savings tiny objective + 1e-6`
- If OR-Tools is not installed:
  - tests skip cleanly with an explicit skip reason
  - full_compare records unavailable solver clearly, without crashing.

### Stage 7: Full Experiment Matrix

Goal: run complete MEIRP experiments with all required algorithm families and
preserve all metrics.

Work:

- Extend `scripts/full_compare.py` only after prior stages are green.
- Add scenario matrix:
  2-layer, 3-layer, 4-layer; small/medium; poisson/normal/seasonal/surge/merton
  as supported by current demand code.
- Add algorithms:
  BaseStock, `(s,S)`, heuristic-routed variants, GA/ALNS variants,
  OR-Tools variants, RL variants, GNN variants, and SEA variants if retained.

Exit gate:

- For every completed algorithm-run:
  - raw per-seed record exists
  - aggregate mean/std record exists
  - all existing metrics are preserved
  - route metrics are present for routing-enabled algorithms
  - all numeric fields are finite, except explicitly marked unavailable solvers
  - runtime is recorded
- Final output files:
  - JSON results
  - CSV summary table
  - optional plots
- Re-running with the same seed reproduces deterministic baselines exactly.

## Execution Rule

Do not advance to the next stage if the current stage exit gate fails.  Fix the
current stage first, rerun its checks, and only then continue.
