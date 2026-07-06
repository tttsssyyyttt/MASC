# AGENTS.md

## Role

You are modifying an existing Inventory Routing Problem project.

Your goal is to make the requested code changes with the smallest possible amount of code.

Do not over-engineer the solution.

---

## Main Rules

1. Keep the code simple.
2. Write as little new code as possible.
3. Prefer modifying existing functions over adding new modules.
4. Do not introduce advanced Python features unless absolutely necessary.
5. Do not redesign the whole project.
6. Do not add unnecessary abstractions.
7. Do not add complex class hierarchies.
8. Do not add configuration systems unless the user asks for them.
9. Do not add logging frameworks unless the user asks for them.
10. Do not add type-heavy or enterprise-style code.

---

## Coding Style

Use plain Python.

Prefer:

* simple functions
* simple loops
* simple dictionaries
* simple lists
* clear if/else logic
* direct calculations

Avoid:

* decorators
* dataclasses
* inheritance
* complex OOP
* factories
* protocols
* generics
* dependency injection
* async code
* metaprogramming
* unnecessary type annotations
* unnecessary helper files
* unnecessary wrappers

The code does not need to be highly standardized or polished.

Correctness and simplicity are more important than style.

---

## Modification Policy

Before editing, inspect the existing code.

When making changes:

1. Change only the files related to the user request.
2. Keep the existing project structure.
3. Keep existing function names unless changing them is necessary.
4. Do not rewrite large parts of the project.
5. Do not remove existing behavior unless the user asks for it.
6. Do not add features that were not requested.

If there are multiple possible solutions, choose the simplest one.

---

## IRP-Specific Rules

This project is about an Inventory Routing Problem.

Do not change the mathematical meaning of the problem unless the user asks.

Be careful with:

* demand
* inventory
* vehicle capacity
* routing cost
* unmet demand
* replenishment
* planning periods
* feasibility checks

If a change affects the IRP logic, explain briefly what changed.

Do not make the environment easier or harder unless the user explicitly asks.

---

## Testing

After modifying code, run the smallest relevant test or script.

Do not create a large test framework unless the user asks.

If no tests exist, run a simple smoke test if possible.

If something fails, report the error clearly.

---

## Output

When finished, summarize briefly:

1. what files changed,
2. what was changed,
3. whether the code was tested,
4. any remaining issue.

Keep the summary short.
