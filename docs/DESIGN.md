# Design Document – scrygent

## Purpose

This document captures every design decision, trade-off, and architectural commitment for scrygent. It is the reference you return to when adding features, onboarding, or justifying choices. If a decision isn't here, it hasn't been made yet.

---

## Core Problem

Build a general-purpose, autonomous CSV analytics engine that:

- Accepts any user-provided CSV and a natural language query.
- Profiles data, plans an analytical route, and executes it deterministically.
- Produces a structured report (insights, statistics, plots) with zero hallucinated numbers.
- Runs entirely on free-tier models and infrastructure (Streamlit Cloud).
- Operates as an automated report generator rather than a conversational chatbot.

---

## Fundamental Design Philosophy

**"The LLM is the planner, not the computer."**

All numerical computation is performed by deterministic Python functions. The LLM's only job is to decide what to compute and to interpret the verified results.

**Layered Architecture (Dependency Rule)**

Flow: `UI → Graph → Nodes → Tools → Data`

- Upper layers may depend on lower layers; the reverse is forbidden.
- The UI never touches data; tools never render HTML.

---

## Why Not a Supervisor Agent?

We originally considered a Supervisor pattern (dynamic routing per step). We switched to Plan-and-Execute for three reasons:

1. **Transparency:** The full plan is visible upfront, which builds trust and makes debugging trivial.
2. **Efficiency:** Only one planning LLM call; subsequent steps are deterministic tool invocations or lightweight LLM summaries.
3. **Portfolio Clarity:** Plan-and-execute is a well-understood, production-proven pattern that neatly separates reasoning (planner) from action (executor).

---

## Hybrid Safety Tier (The Core of Our Reliability)

We do not hand the LLM arbitrary Python execution by default. Instead, we use a two-tier safety model.

### Tier 1 – Unified Declarative Tool Suite (Default)

Rather than exposing 15+ low-level Pandas wrappers to the LLM, we use a **Unified Declarative Analyst** approach. The core query operations (filter, group, aggregate, sort, top-N) are consolidated into a single highly parameterized tool: `analyze_data`. Supporting tools cover correlation, regression, outlier detection, visualization, data loading, column stats fetching, and wrangling.

This approach was chosen over the original wrapper-per-operation design for two reasons:

1. **Planner simplicity:** Giving the LLM 15+ Pandas wrappers forces it to think like a Python programmer and guess which function to call. Giving it `analyze_data` lets it think like a data analyst, expressing the intent of a query in one step. This maps cleanly to 90%+ of DABench questions.
2. **Architectural integrity:** Individual wrapper tools (e.g., `filter_aggregate`, `groupby_agg`) would need to return DataFrames to chain results, violating the stateless-tools rule. `analyze_data` resolves this by performing the full operation internally and returning a scalar or structured dictionary.

All Tier 1 tools:

- Accept typed parameters (column names, operations, filters).
- Return small, structured `dict`/`list` — never DataFrames.
- Are implemented in pure Pandas/NumPy/SciPy — no LLM-generated code runs.
- Apply `_safe_cast_metric` to strip all `np.nan`, `np.inf`, `np.int64`, and other Pandas native C-types before returning, ensuring JSON-safe output that the Groq API can parse without 400 errors.

**Guarantee:** All numbers in reports produced via Tier 1 come from auditable functions. Zero hallucination.

### Tier 2 – Sandboxed Code Execution (Local Escalation Only)

For rare, genuinely novel calculations not covered by Tier 1, the agent may use `execute_python`. This is invoked only locally (benchmarking, development) and is **fully disabled on the public Streamlit Cloud deployment**.

**Why disabled on Streamlit Cloud:** Local `exec`-based sandboxes are notoriously escapable in shared cloud environments. A malicious actor could use a sandbox step to access environment variables, including `GROQ_API_KEY`. This is an unacceptable security risk for a public portfolio demo. The `DISABLE_SANDBOX=true` environment variable in the Streamlit Cloud dashboard enforces this. The README and UI clearly communicate the limitation.

When active locally, the sandbox runs LLM-generated Pandas code in a restricted environment:

- Only `pandas`, `numpy`, `scipy.stats` are importable.
- No file system, network, or `eval`/`exec` access.
- Timeout of 5 seconds.
- A copy of the DataFrame is provided read-only; the original state is never mutated.
- The result is captured as a string.
- The Planner writes only a natural-language instruction for sandbox steps, never code. The `code_writer_chain` (a separate just-in-time LLM call in the Executor) generates the actual code using the current state and instruction. This prevents blind code generation that fails because earlier steps changed the data.

---

## Model Choice

- **Planning/Reasoning:** Groq's `llama-3.3-70b-versatile` (free tier, fast, tool-calling capable). Used with LangChain's `.with_structured_output()` to enforce Pydantic schemas directly.
- **Fallback:** Google Gemini Flash (free tier) if needed.
- **Rationale:** Both support structured output, and our tool descriptions are kept minimal to avoid prompt bloat.

---

## The Fixed Plan-and-Execute Loop

1. **Profiler Node (deterministic):** Runs first. Calls `profile_dataframe`. Builds a two-level profile:
   - `global_schema`: name and dtype for every column in the CSV, always complete.
   - `detailed_stats`: full statistical metrics for columns matched by a **strict whole-word regex** against the user query (pattern: `(?<!\w)<column_name>(?!\w)`) plus the top 15 most populated numerical/categorical columns. The strict regex replaces the original substring `in` check, preventing false positives (e.g., a column named `id` matching the word `dividend`).
   - A 3-row data sample with `None` substituted for NaN cells (via `.where(pd.notna(df), None)`) for format inference.
   - All metrics are passed through `_safe_cast_metric` before storage.

2. **Planner Node (LLM):** Receives the user query and the full `CSVProfile` (both levels). The Planner sees the complete `global_schema`, so it is never blind to column existence. If it determines that detailed stats are needed for columns not in `detailed_stats`, it plans a `request_column_stats` step. Outputs a `Plan` (Pydantic list of `Step` objects). Sets `execution_status` to `"running"`.

3. **Executor Node (hybrid node):** Pops the next step. The Executor dispatches deterministic Tier 1 tools *and* runs LLM chains internally (`correction_chain`, `code_writer_chain`). It is not a purely deterministic dispatcher.
   - **Tool step:** calls the registered Tier 1 tool with step parameters. On Pydantic validation failure, the internal `correction_chain` (LLM prompt with bad params + schema + error) runs a localised correction, max 2 retries. This is a Python-level loop inside the node function, not a LangGraph graph edge.
   - **Sandbox step (local only):** invokes `code_writer_chain` (LLM) to generate Python, executes in sandbox, stores string output. If `DISABLE_SANDBOX=true`, logs a skipped-step warning to `error_log` and continues.
   - Sets `execution_status` to `"aborted"` if a `required: True` step exhausts retries.

4. **Reporter Node (LLM, required):** Behaviour depends on `eval_mode`.
   - **Report mode (`eval_mode = False`, default):** produces a full `AnalysisReport`. The Pydantic schema requires `primary_answer` (the direct answer to the user's query as a string) to be populated before any secondary content. `additional_insights` is an optional list populated only if the tool outputs contain relevant supporting observations. This isolation guarantees the user's question is answered accurately even if the LLM would otherwise bury it in narrative.
   - **Eval mode (`eval_mode = True`):** produces a `DirectAnswer` — the specific answer value only, in the format the benchmark evaluation harness expects. No narrative. No plots. No secondary insights.
   - In both modes, all numerical figures must be derived from verified tool outputs in context.

5. **Loop Termination:** Repeats until plan exhausted or required step fails, then routes via `execution_status`.

---

## What We Decided Not to Build

### Explorer Node

We considered adding an Explorer Node to proactively surface anomalies beyond the user's explicit query. We rejected this for two reasons:

1. **Mission creep:** An open-ended anomaly scan consumes tokens and adds latency without a guaranteed payoff.
2. **Answer burial risk:** Merging the direct query answer with proactive exploration in a single unstructured LLM call risks the primary answer being lost in narrative. The `primary_answer` field in `AnalysisReport` solves this problem at the Reporter level without requiring a separate node.

Secondary observations are surfaced only through `additional_insights`, populated exclusively from tool outputs already in context.

---

## Project Layout

```text
scrygent/
├── app.py
├── pyproject.toml
├── uv.lock
├── README.md
├── docs/
│   ├── DESIGN.md
│   └── ARCHITECTURE.md
├── .streamlit/
│   └── secrets.toml
├── src/
│   ├── __init__.py
│   ├── state.py
│   ├── models.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── io.py
│   │   ├── profiler.py
│   │   ├── analyze_data.py       # Unified declarative tool (filter/group/agg/sort/top-N)
│   │   ├── column_stats.py       # request_column_stats (lazy fetch)
│   │   ├── statistics.py         # correlation, linear_regression, trend_detection, histogram_data
│   │   ├── wrangling.py          # fill_missing, cast_column_type, normalize_column,
│   │   │                         #   create_derived_column, date_extract, string_filter
│   │   ├── outliers.py
│   │   └── visualization.py
│   ├── sandbox/
│   │   ├── __init__.py
│   │   └── executor.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── profiler_node.py
│   │   ├── planner_node.py
│   │   ├── executor_node.py
│   │   └── reporter_node.py
│   └── graph/
│       ├── __init__.py
│       └── builder.py
├── data/                        # Git-ignored. Local benchmarking only — never commit.
│   ├── dabench/                 #   InfiAgent-DABench CSVs and question files
│   ├── databench_lite/          #   HuggingFace DataBench Lite (auto-downloaded)
│   └── poisoned/                #   Manually generated robustness test CSVs
└── tests/
    ├── fixtures/                # Small committed CSVs for CI unit tests only
    │   └── sales_sample.csv
    ├── test_tools/
    ├── test_agents/
    └── test_graph/
```

> **`data/` vs `tests/fixtures/`:** `tests/fixtures/` contains only small CSVs committed to the repository and run by CI on every push. `data/` is listed in `.gitignore` and holds the full benchmark datasets — DABench CSVs, DataBench Lite downloads, and poisoned variants — which are too large to commit. Never put benchmark data in `tests/fixtures/`; doing so would bloat the repository and slow CI.

---

## Tool Suite Reference (Tier 1)

All tools are pure functions in `src/tools/`. They never call the LLM. All return values are JSON-safe (no Pandas/NumPy native types).

| Tool File | Key Functions | Purpose |
|---|---|---|
| `io.py` | `load_csv`, `get_column_sample` | Safe CSV loading with strict error chaining; 3-row sample with `None` for NaN cells |
| `profiler.py` | `profile_dataframe`, `_safe_cast_metric` | Two-level profile (`global_schema` + `detailed_stats`); scrubs all non-JSON-safe types |
| `analyze_data.py` | `analyze_data` | **Unified declarative tool.** Accepts filter conditions, group-by columns, aggregation operation, sort direction, and top-N limit in a single call. Returns a scalar or structured dict. Covers 90%+ of DABench aggregation and query questions. |
| `column_stats.py` | `request_column_stats` | Lazy fetch: computes full statistical metrics for a named column on demand. Called during execution when the Planner determines the profiler's truncated set was insufficient. |
| `statistics.py` | `correlation`, `linear_regression`, `trend_detection`, `histogram_data`, `arithmetic` | Targeted statistical operations not covered by `analyze_data`. `arithmetic` uses `asteval`/`numexpr` exclusively. |
| `wrangling.py` | `fill_missing`, `cast_column_type`, `normalize_column`, `create_derived_column`, `date_extract`, `string_filter` | Data preparation. Transforming tools write a new temporary CSV, return its path, and the Executor updates `csv_path` in state. |
| `outliers.py` | `detect_outliers` (zscore, iqr) | Outlier detection with configurable method |
| `visualization.py` | `generate_plot` | Generates graphs, saves to disk, returns file paths and metadata. Plots are never base64-encoded into state. |

**Key tool detail – `analyze_data`:**
The unified declarative tool. Parameters: `column` (target), `filter_col`/`filter_op`/`filter_val` (optional row filter), `group_by` (optional grouping column), `agg` (operation: `mean`, `sum`, `count`, `min`, `max`, `median`, `std`), `sort_by`/`sort_dir` (optional sort), `top_n` (optional limit). Performs the full operation internally in a single Pandas chain and returns a clean scalar or dict. This eliminates the DataFrame-passing anti-pattern that would arise from chaining separate filter → aggregate tools.

**Key tool detail – `request_column_stats`:**
Lazy fetch for column details. Takes a column name, reads the CSV at `csv_path`, and returns the same statistical summary `profile_dataframe` would have generated for that column. Allows the Planner to request on-demand profiling without re-running the entire profiler node.

**Key tool detail – `arithmetic`:**
Safely evaluates basic mathematical expressions (e.g., `"a / b"`) with provided scalar variables. Relies exclusively on `asteval` or `numexpr`.

**Key tool detail – `visualization.py`:**
Plots are never converted into base64 blobs inside `AgentState`. They are saved to temporary storage and only their file path and conceptual metadata are committed to state, preventing memory bloat in Streamlit Cloud container runtimes.

**Key tool detail – `date_extract`:**
Accepts a date column and an extraction target (year, month, day, weekday). Also computes the difference in days between two date columns. Handles both datetime and string columns, attempting automatic parsing before raising a clean error.

**Key tool detail – `string_filter`:**
Filters rows where a string column satisfies a condition (contains, starts_with, ends_with, equals, not_equals). Case-insensitive by default. Returns a filtered row count and sample as a structured dict. Does not return a DataFrame.

---

## Sandbox Execution Details (Tier 2, Local Only)

**Example planner step:**

```json
{
  "action": "sandbox",
  "instruction": "Calculate the 95th percentile of normalized_sales after recent transformations."
}
```

**Executor handling sequence:**

1. Checks `DISABLE_SANDBOX` env var. If set, logs to `error_log` and skips.
2. Assembles a prompt containing the instruction, current targeted data profile, and relevant intermediate outputs.
3. Calls `code_writer_chain` (lightweight LLM) to produce Python code.
4. Passes the generated code and a read-only DataFrame copy to `execute_python`.
5. Stores the returned string output in state.

This avoids both the "code generated blindly at planning time" trap and the public security exposure.

---

## State Validation & Error Recovery Strategy

### Local Pydantic Execution Correction

When a tool execution encounters a validation anomaly, the graph does not trigger a global back-edge loop to the Planner Node. Instead, the `ExecutorNode` contains a dedicated internal `correction_chain`. This chain issues a localised LLM correction prompt — passing the bad parameters, the target schema, and the Pydantic error details — forcing a rapid correction within the current step (capped at 2 retries). This is a Python-level loop internal to the executor function, not a LangGraph graph edge.

### Explicit Failure Control

Every `Step` carries a mandatory `required: bool` field (defaulting to `True`).

- If a step marked `required: True` exhausts retries, the executor sets `execution_status` to `"aborted"`, the graph routes to the terminal abort state, and a clear error is surfaced.
- If `required: False`, the executor logs the warning to `error_log` and proceeds.

---

## JSON Serialization

Pandas operations produce Pandas/NumPy native C-types (`np.nan`, `np.inf`, `np.int64`, `np.float32`, etc.) that violate strict JSON specifications. Passing them raw into LangGraph state causes the downstream Groq API to throw 400 Bad Request errors.

**Mitigations applied at every tool boundary:**

- `_safe_cast_metric(value)` — a helper in `profiler.py` that converts: `np.nan` → `None`, `np.inf`/`-np.inf` → `None`, `np.integer` subclasses → `int`, `np.floating` subclasses → `float`. Applied to every metric before it enters state.
- Row samples from `get_column_sample` replace NaN cells with `None` via `.where(pd.notna(df), None)` before calling `.to_dict(orient="records")`.

---

## Data Fidelity Guarantee

- Raw bulk data never enters the LLM prompt. The 3-row sample is passed solely for format inference (date patterns, string delimiters). The dataset remains isolated.
- All numbers in reports are either returned by Tier 1 tools or captured from Tier 2 sandbox output strings.
- This guarantee is stated explicitly in the README.

---

## UI State Synchronization

scrygent is a single-pass fire-and-forget engine, not a multi-turn chatbot. LangGraph's `MemorySaver` checkpointer is designed for pause-and-resume multi-turn flows and is not used.

Instead, `app.py` caches the entire final `AnalysisReport` (and the full `AgentState`) in `st.session_state` after each `graph.invoke` call. When Streamlit rerenders the UI due to widget interaction, the cached result is read directly — the graph is not re-invoked. This is both cheaper and more predictable than maintaining a checkpointer.

---

## Testing Strategy

| Layer | Approach |
|---|---|
| Unit tests | Every tool function tested with known DataFrames |
| JSON safety tests | Verify `_safe_cast_metric` converts all Pandas native types correctly |
| Node tests | Planner, executor, and reporter tested with mocked LLM responses |
| Routing tests | Conditional router tested for all three `execution_status` values |
| Integration tests | Full graph run with a sample CSV, asserting `primary_answer` populated and `AnalysisReport` valid |
| Security tests | Verify `execute_python` rejects forbidden imports and times out; verify `DISABLE_SANDBOX=true` skips sandbox steps gracefully |
| CI | Pytest runs on every push via `uv run pytest` |

---

## Deployment Notes

- Streamlit Cloud handles deployment and reads `pyproject.toml` for dependencies.
- `uv` and `uv.lock` are used for fast, deterministic local development. Streamlit Cloud installs via standard `pip`.
- **Pandas 3.x is required.** The tool suite and profiler rely on Pandas 3.x strict C-type array behavior (no implicit upcasting of string columns into float, `LossySetitemError` on type-unsafe assignments). Running on Pandas 2.x or 1.x will cause silent behavioral differences in `profiler.py` type inference and may cause the data-poisoning robustness tests to pass incorrectly. Pin `pandas>=3.0` in `pyproject.toml`.
- `GROQ_API_KEY` and `DISABLE_SANDBOX=true` are both set in the Streamlit Cloud dashboard secrets/environment.
- The entire `graph.invoke` result is cached in `st.session_state` after each run; no rerenders trigger a re-invocation.
- Temporary files (plots, intermediate wrangling CSVs) are cleaned by Streamlit's container session lifecycle.

---

## Known Limitations & Open Design Questions

**Filter + correlate gap:** `analyze_data` handles filter/aggregate in one pass and returns a scalar or dict. If a question requires filtering rows and then computing a cross-column correlation on the filtered subset, no single Tier 1 tool covers the combined operation. On public deployment (sandbox disabled), such queries fail gracefully. On local deployment, the Planner can route to the sandbox.

**Wrangling tool chaining:** Wrangling steps that produce a transformed dataset write a new temporary CSV and update `csv_path` in state. All subsequent tool calls read from the updated path. The original CSV becomes inaccessible after a wrangling step. If a plan needs both pre- and post-transformation values, the Planner must schedule the pre-transformation tool calls first.

---

## Formal Evaluation

scrygent is evaluated against two published benchmarks. Set `eval_mode = True` in `AgentState` before `graph.invoke` to activate evaluation mode.

**Primary: InfiAgent-DABench (ICML 2024)**
603 questions, 124 CSV files. GPT-4 baseline 78.99%. Toolkit at `github.com/InfiAgent/InfiAgent`. Size under 100MB. Run locally.

**Secondary: DataBench Lite (SemEval 2025 Task 8)**
80 datasets, 1,822 questions, maximum 20 rows per dataset. HuggingFace `cardiffnlp/databench`. Downloads automatically.

**Model backbones**
Run with at minimum two backbones: Llama 3.3 70B via Groq (free tier) and one premium model (GPT-4o or Claude 3.5 Sonnet). Report both scores against the 78.99% baseline. The Tier 1 deterministic advantage is expected to be most visible on standard statistical and aggregation questions where code-generation agents hallucinate but scrygent cannot.
