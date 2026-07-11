# `tools/` Map of the Tool Layer

This is a map, not a spec. It shows where each tool's three coordinates
live — parameter contract, dispatch entry, implementation — so you don't
have to open four files to find one tool. The actual behavior always
lives in the source files linked below; if this table and the code ever
disagree, the code is right and this file is stale (fix it).

## Golden Rule reminder

`contracts/` → nothing. `ir/` → `contracts/` + `base_model` only.
`models/` → `ir/` + `contracts/` + `base_model`. `tools/` → `contracts/`
+ `tools/_shared/` only. Never the reverse.

## The Tool Registry (one row per `ToolName` member)

| `ToolName` | Param Model (`ir/`) | Dispatch Function (`tools/`) | Enum Vocabulary (`contracts/`) |
|---|---|---|---|
| `analyze_data` | `AnalyzeDataParams` — `ir/analyze_data.py` | `analyze_data()` — `tools/analyze_data.py` | `Aggregation`, `FilterOperator` |
| `filter_dataset` | `FilterDatasetParams` — `ir/wrangling.py` | `filter_dataset()` — `tools/wrangling.py` | `FilterOperator` |
| `normalize_column` | `NormalizeColumnParams` — `ir/wrangling.py` | `normalize_column()` — `tools/wrangling.py` | `NormalizeMethod` |
| `reset_dataset` | `NoParams` — `ir/wrangling.py` | `reset_dataset()` — `tools/wrangling.py` | — |
| `correlation` | `CorrelationParams` — `ir/statistics.py` | `correlation()` — `tools/statistics.py` | `CorrelationMethod` |
| `regression` | `RegressionParams` — `ir/statistics.py` | `regression()` — `tools/statistics.py` | `RegressionMethod` |
| `detect_outliers` | `OutlierParams` — `ir/statistics.py` | `detect_outliers()` — `tools/statistics.py` | `OutlierMethod` |
| `request_column_stats` | `ColumnStatsParams` — `ir/statistics.py` | `request_column_stats()` — `tools/statistics.py` | — |
| `generate_plot` | `PlotParams` — `ir/visualization.py` | `generate_plot()` — `tools/visualization.py` | `PlotType` |
| `derive_column` | `DeriveColumnParams` — `ir/arithmetic.py` | `derive_column()` — `tools/arithmetic.py` | — |
| `evaluate_metrics` | `EvaluateMetricsParams` — `ir/arithmetic.py` | `evaluate_metrics()` — `tools/arithmetic.py` | — |

Every row above also has a matching entry in `models/registry.py`'s
`TOOL_PARAM_MODELS` and `agents/executor_node.py`'s `_TOOL_DISPATCHER`.
**Adding a tool means touching all four places** (contracts enum if
needed, `ir/` param model, `tools/` function, both registries) — there
is no single point that auto-wires the rest.

## `profiler_dataframe` is not in this table on purpose

`tools/profiler.py`'s `profile_dataframe()` is called directly by
`profiler_node.py`, not dispatched through `ToolName`/`_TOOL_DISPATCHER`.
It's deterministic pre-processing that runs once, before the Planner is
ever invoked — it isn't a tool the LLM selects.

## Shared logic lives in `tools/_shared/`, not in a base class

- `_shared/filtering.py` → `apply_filters(df, filters)` — used by both
  `analyze_data.py` (inline filter phase) and `wrangling.py`'s
  `filter_dataset`. One filter grammar, one implementation.
- `_shared/column_stats.py` → `compute_detailed_stats(df, columns)` —
  used by both `profiler.py` (initial profile) and `statistics.py`'s
  `request_column_stats` (lazy fetch), guaranteeing identical output
  shape between the two.

This is composition, not inheritance. tools call these as plain
functions, not through a shared parent class. See the "why not ABC"
discussion in review history if this comes up again: dispatch dicts +
`_shared/` composition is the deliberate pattern here, not a gap.

## Per-tool-family sub-dispatch (inside individual tool files)

A few tool files have their own internal dispatch dict, one level down
from `_TOOL_DISPATCHER`, keyed by the tool's own enum:

- `wrangling.py`: `_NUMERIC_METHODS`, `_STRING_METHODS` (both keyed by
  `NormalizeMethod`) — which one applies depends on the target column's
  dtype, checked at runtime.
- `statistics.py`: `_REGRESSION_METHODS` (keyed by `RegressionMethod`),
  `_OUTLIER_METHODS` (keyed by `OutlierMethod`).
- `visualization.py`: `_PLOT_HANDLERS` (keyed by `PlotType`).

Same idiom throughout: `StrEnum` member → function reference. No
exceptions, no class hierarchy — see `llm_factory.py`'s provider
resolution for the one non-tool place this same pattern is reused.

## Executor kwargs-injection special cases

`executor_node.py` doesn't pass every tool the same arguments — it
special-cases three tools before generic injection:

| Tool | Gets instead of `current_csv_path` |
|---|---|
| `analyze_data` | `df` — a loaded `DataFrame`, not a path |
| `reset_dataset` | `original_csv_path` — the immutable baseline path |
| `evaluate_metrics` | neither — operates purely on `values`, no file I/O |

Every other tool gets `current_csv_path` injected automatically. If you
add a tool with a different data-access pattern, this is the place to
extend, in `executor_node.py`'s kwargs-prep block — not in the tool
itself.

## Return contract for CSV-mutating tools

`filter_dataset`, `normalize_column`, and `reset_dataset` all return
`{"current_csv_path": str(new_or_reset_path), ...}`. The executor reads
this exact key to update `AgentState.current_csv_path` for subsequent
steps. If you add a new wrangling tool that writes a transformed CSV,
it must return this same key name, or the multi-step composition
pattern (filter → analyze) silently breaks — the executor falls back to
the *old* path with no error.

## Validation boundary

Every tool's raw string enum inputs (`aggregation`, `operator`, `method`,
etc.) get coerced against their `contracts/` `StrEnum` in **two** places
for different reasons:

1. **`ir/` param models** (e.g. `AnalyzeDataParams.metrics[].aggregation:
   Aggregation`) — Pydantic rejects bad values before the `Step` is even
   constructed, at Planner-output-validation time.
2. **Tool functions themselves** (e.g. `normalize_column`'s
   `NormalizeMethod(method)` coercion) — defense-in-depth for any caller
   that bypasses `Step` validation (tests calling the function directly,
   for instance).

This is deliberate redundancy, not dead code — but the coercion call in
tool functions must always be wrapped in `try/except` and re-raise a
clear message.