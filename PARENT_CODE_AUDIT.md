# Parent Folder Code Audit

Date: 2026-07-04

Scope: `D:\Project\Python\MaSCL`, excluding `.git` object files.  The full
per-file marking table is in `PARENT_FILE_MARKING.csv`.

## Labels

- `A`: directly useful / high priority for MEIRP migration.
- `B`: useful, but needs refactor or is mainly a reference.
- `C`: historical reference, old tests, old RL utilities, or old data.
- `D`: generated output, checkpoints, cache/tooling, or obsolete artifacts.
- `E`: active file in the current `meirp_project`; integration target.

## Main Findings

The useful code is concentrated in four places:

- Routing heuristics: `meirp_madrl/env/cvrp_solver.py`, `envs/network/cvrp_solver.py`,
  and their `CVRPRoutingPolicy` wrappers.
- Old routing-aware environment modeling: `meirp_madrl/env/meirp_env.py`,
  `envs/environments/irp_net2_decoupled.py`, and `envs/network/network_builder.py`.
- Metaheuristics for inventory/order decisions: `alns_implementation/alns2.py`,
  plus archived current-style `ga_solver.py` and `alns_solver.py`.
- Exact / OR-Tools references: `alns_implementation/alns3.py`,
  `envs/environments/MILP.py`, and archived `cp_sat_solver.py`.

No complete nearest-insertion routing implementation was found.  No complete GA
routing solver was found.  Existing GA/ALNS code optimizes inventory/order
quantities, not routes.  Existing route heuristic code is mostly Clarke-Wright
savings plus 2-opt.

## Requested Baseline Coverage

| Required comparison type | Existing useful code | Status |
| --- | --- | --- |
| Heuristic inventory + heuristic routing | Current `BaseStockPolicy` / `SSPolicy`; old `baseStock.py`, `ss_policy.py`, `ss_policy_v2.py`; `cvrp_solver.py` savings + 2-opt | Feasible after adding routing cost/model to current env. Nearest insertion still needs new implementation. |
| Heuristic inventory + metaheuristic routing | Inventory side exists; no GA routing implementation found | Need to write GA route optimizer, likely seeded by savings/2-opt. |
| Integrated metaheuristic, e.g. GA/ALNS overall | `alns_implementation/alns2.py`; archived `ga_solver.py` / `alns_solver.py` | ALNS/GA can be revived for order decisions. To be true IRP, objective must include routing cost from the new routing module. |
| Metaheuristic inventory + heuristic routing | archived `ga_solver.py` / `alns_solver.py` + savings/2-opt routing | Good migration path: revive current-style GA/ALNS for orders, then compose with savings/2-opt. |
| Exact MILP / OR-Tools | `alns_implementation/alns3.py`, `envs/environments/MILP.py`, archived `cp_sat_solver.py` | Use OR-Tools CP-SAT as main exact path. Current `solvers/milp_solver.py` is order/allocation only and has no real route variables. |

## High-Priority Files

| Label | File | Use |
| --- | --- | --- |
| A | `meirp_madrl/env/cvrp_solver.py` | Best concise source for savings + 2-opt CVRP routing. Needs tests and small correctness cleanup before porting. |
| A | `meirp_madrl/env/cvrp_routing_policy.py` | Adapter pattern for solving per-layer routes from graph nodes. |
| A | `meirp_madrl/env/network_builder.py` | Coordinates, layer topology, `(s,S)` parameters, and vehicle config generation. |
| A | `meirp_madrl/env/meirp_env.py` | Clean old environment preserving CVRP routing and transport cost; best blueprint for current env split. |
| A | `meirp_madrl/env/demand_generator.py` | Poisson/normal/Merton demand generator; current project already has a demand module, but this is useful for parity. |
| A | `envs/network/cvrp_solver.py` | Duplicate/alternate savings + 2-opt implementation. Compare against `meirp_madrl` version. |
| A | `envs/network/network_builder.py` | More complete topology builder, including fully connected variant and vehicle config. |
| A | `envs/policies/policies.py` | Simple `InventoryPolicy` / `RoutingPolicy` abstractions. |
| A | `envs/policies/cvrp_routing_policy.py` | Older graph routing wrapper. |
| A | `alns_implementation/alns2.py` | Richest ALNS inventory/order optimizer; useful for integrated metaheuristic baseline. |
| A | `alns_implementation/alns3.py` | Rolling-horizon OR-Tools CP-SAT with routing arcs; strongest exact IRP reference. |
| A | `envs/environments/MILP.py` | Larger duplicate of OR-Tools order+routing model; cross-check with `alns3.py`. |
| A | `meirp_project/archive/obsolete_unused/solvers/ga_solver.py` | Current-schema GA over order quantities; revive/refactor, not direct import. |
| A | `meirp_project/archive/obsolete_unused/solvers/alns_solver.py` | Current-schema ALNS over order quantities; easier to revive than old graph ALNS. |
| A | `meirp_project/archive/obsolete_unused/solvers/cp_sat_solver.py` | Current-schema CP-SAT oracle for orders/allocation; no routing. |

## Useful But Not Direct

| Label | File | Reason |
| --- | --- | --- |
| B | `envs/environments/irp_net2_decoupled.py` | Huge old environment with routing/MILP/human logic. Mine concepts only. |
| B | `two_stage_algorithm/two_stage_algorithm.py` | OR-Tools inventory + savings route idea, but route calculation is partly commented and paths/results are hardcoded. |
| B | `envs/environments/heuristic/heuristic_solver.py` | Old heuristic comparison harness; tied to old env. |
| B | `envs/environments/heuristic/baseStock.py` | Graph BaseStock variants, including demand/service-level target inventory. |
| B | `envs/environments/heuristic/ss_policy.py` | Graph `(s,S)` variants; mostly duplicate of current baseline semantics. |
| B | `envs/environments/heuristic/savings_algorithm.py` | Older savings solver; use only as reference. |
| B | `envs/policies/ss_policy_v2.py` | Old graph `SSPolicy`; useful for checking effective-inventory semantics. |
| B | `envs/policies/MILP/OrToolsMILP1.py` | OR-Tools skeleton/preprocessing; incomplete. |
| B | `envs/D_MEIRP_Solver.py` | Gurobi-based exact-model attempt; old deps and missing/old imports, reference only. |
| B | `alns_implementation/rolling_window_alns.py` | Earlier ALNS version, superseded by `alns2.py`. |
| B | `test_algorithm_comparison.py` | Old comparison runner for sS/BaseStock/Two-Stage/ALNS; useful experiment structure. |
| B | `results/algorithm_comparison.json` | Historical result snapshot only, not current benchmark evidence. |
| B | `project_desc.md`, `prompt.md` | Useful old architecture/design intent documents. |
| B | `configs/*.yaml`, `configs/best332` | Scenario/config values; schema needs adaptation. |

## Folder Marking

| Folder | Mark | Role |
| --- | --- | --- |
| `.codebuddy`, `.workbuddy`, `.idea`, `.pytest_cache`, `__pycache__` | D | Tooling/cache metadata. |
| `algorithms/` | C | Legacy HAPPO/MADRL training utilities; not needed for requested classical baselines. |
| `alns_implementation/` | A/B | Main source for ALNS and OR-Tools CP-SAT references. |
| `configs/` | B | Old scenario configs and tuning values. |
| `deprecated/` | D | Explicitly obsolete old files. |
| `envs/` | A/B/C | Old environment stack; contains valuable routing/exact pieces but also much legacy code. |
| `eval_res/`, `results/`, `visualizations/` | D/B | Historical outputs; only `algorithm_comparison.json` is mildly useful as history. |
| `meirp_madrl/` | A/C/D | Useful env/routing subset; training scripts/checkpoints are historical. |
| `origin/` | C | Old serial environment/routing prototype. |
| `test_data/`, `test_utils/` | C | Old demand data and plotting helpers. |
| `two_stage_algorithm/` | B | Two-stage optimization idea, not directly reliable. |
| `utils/` | C | Legacy RL utility code. |
| `meirp_project/` | E/A/C/D | Current project. Active files are integration target; archived solvers contain useful GA/ALNS/CP-SAT material. |

## Recommended Migration Order

1. Add a current-project routing module: `savings`, `two_opt`, `nearest_insertion`,
   route distance/cost utilities, and route result dataclasses.
2. Extend current config/env with coordinates, vehicle capacity/count, and
   transport cost. Use `meirp_madrl/env/meirp_env.py` and `network_builder.py`
   as the blueprint, but keep current BDQ action handling.
3. Implement composed baselines:
   `BaseStock+Savings`, `sS+Savings`, `BaseStock+NearestInsertion`,
   `sS+NearestInsertion`, and 2-opt variants.
4. Revive archived `ga_solver.py` / `alns_solver.py` as current-schema
   inventory/order metaheuristics, then compose them with heuristic routing.
5. Add new GA routing solver for the missing heuristic-inventory +
   metaheuristic-routing category.
6. Build a new `ORToolsIRPSolver` from `alns3.py` / `MILP.py`, targeting small
   scenarios first because full route variables scale quickly.
7. Update `scripts/full_compare.py` only after the routing-aware env and
   solver APIs are stable; keep all existing metrics.
