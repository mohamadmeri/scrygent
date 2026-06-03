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

We originally considered a Supervisor pattern (dynamic routing per step). We switched to Plan-and-Execute for four reasons:

1. **Transparency:** The full plan is visible upfront, which builds trust and makes debugging trivial.
2. **Efficiency:** Only one planning LLM call; subsequent steps are deterministic tool invocations or lightweight LLM summaries.
3. **Portfolio Clarity:** Plan-and-execute is a well-understood, production-proven pattern that neatly separates reasoning (planner) from action (executor).
4. **Accumulated context cost:** A supervisor makes one LLM call per step, and each call receives the full accumulated conversation history: the original query, every previous tool output, every routing decision so far. By step 6 of an 8-step analysis, the supervisor is processing a long and growing context just to decide what to do next. This burns tokens on repeated context the model has already seen, increases latency on every step, and on Groq's free tier hits the tokens-per-minute rate limit faster. Plan-and-execute avoids this entirely: the planner reads the profile once, the executor dispatches steps with no LLM call for Tier 1 tools, and the reporter reads all outputs once at the end.

---

## Hybrid Safety Tier (The Core of Our Reliability)

We do not hand the LLM arbitrary Python execution by default. Instead, we use a two-tier safety model.

### Tier 1 – Deterministic Tool Suite (Default)

A curated set of pre-defined, fully tested statistical and data-wrangling functions. They:

- Accept typed parameters (column names, operations).
- Return small, structured dictionaries of numbers/strings.
- Are implemented in pure Pandas/NumPy/SciPy — no LLM-generated code runs.
- Cover 90%+ of real-world CSV questions (summarization, correlation, outliers, grouping, etc.).

**Guarantee:** All numbers in reports produced via Tier 1 come from auditable functions. Zero hallucination.

### Tier 2 – Sandboxed Code Execution (Escalation)

For rare, genuinely novel calculations not covered by Tier 1, the agent may use a single `execute_python` tool. This tool runs LLM-generated Pandas code in a heavily restricted environment. Crucially, the Planner never writes the actual code; it only writes a natural-language instruction. When the Executor reaches a sandbox step, a separate just-in-time LLM call (a `code_writer_chain`) generates the code based on the current state and the instruction. This prevents blind code generation that fails because earlier steps changed the data.

**Sandbox constraints:**

- Only `pandas`, `numpy`, `scipy.stats` are importable.
- No file system, network, or `eval`/`exec` access.
- Timeout of 5 seconds.
- A copy of the DataFrame is provided read-only; the original state is never mutated.
- The result is captured as a string; it never becomes the sole source of a final answer without explicit flagging in the UI.

The UI shows when Tier 2 is activated. The README explains the escalation logic. The sandbox is clearly separated in code.

---

## Model Choice and LLM Resilience

- **Planning/Reasoning:** Groq's `llama-3.3-70b-versatile` (free tier, fast, tool-calling capable). Used with LangChain's `.with_structured_output()` to enforce Pydantic schemas directly.
- **Fallback provider:** Google Gemini Flash (free tier). Documented as a known extension point. Not activated in the current implementation because structured output schema compatibility between providers must be verified by real testing before a fallback can be trusted. Activating an untested fallback in a critical path is worse than a clean failure.
- **Retry logic:** All LLM calls (planner, reporter, correction chain, code writer) are wrapped in a shared thin retry wrapper. It catches `RateLimitError` and `APIConnectionError`, retries up to 3 times with exponential backoff, then surfaces a specific, user-readable error rather than a raw exception. This is the only resilience mechanism implemented and tested in the current version.
- **Rationale:** Tool descriptions are kept minimal to avoid prompt bloat. The retry wrapper is shared across all nodes via a single utility function in `src/utils/llm.py`.

---

## The Fixed Plan-and-Execute Loop

1. **Profiler Node (deterministic):** Runs first. Calls `profile_dataframe` and `load_csv`. Loads the CSV into memory and stores the resulting DataFrame as `state.df`. For the profile payload sent to the Planner, it applies a token count check first. If the full profile for all columns fits within a safe threshold, it sends complete statistics for every column with no truncation. If the full profile exceeds the threshold, it falls back to intelligent truncation: full statistical details for columns explicitly named in the user query, plus the top 15 most populated numerical/categorical columns. In both cases it always includes the name and dtype of every column in the payload, so the Planner can reference any column by name regardless of whether its detailed stats were included. It also extracts a 3-row data sample for format inference. No LLM involved.

2. **Planner Node (LLM):** Receives the user query and the optimized profile payload. Outputs a `Plan` (Pydantic list of `Step` objects). For sandbox steps, it writes only a natural-language instruction, never code. Sets `execution_status` to `"running"`.

3. **Executor Node (deterministic dispatcher):** Pops the next step. If the step type is `"tool"`, it calls the registered tool with step parameters. If the step type is `"sandbox"`, it invokes the code-writer LLM to generate Python code, then executes it in the sandbox. All tool calls are wrapped in a broad `try...except Exception` block. Both Pydantic `ValidationError` and runtime errors (`KeyError`, `ValueError`, etc.) are handled locally within this node — all exceptions are converted to a string message and passed to the internal `correction_chain` for the same retry logic. Updates `execution_status` to `"complete"` when all steps finish, or `"aborted"` if a required step fails.

4. **Reporter Node (LLM, required):** Synthesizes all tool outputs and plan execution states into a final `AnalysisReport`. It must base all numbers on the validated outputs provided. Its prompt explicitly instructs: if any step output resembles a Python traceback or error string rather than a result, it must treat that step as failed, omit its output from the report's numbers, and advise the user to rephrase the relevant part of their query. It must never interpret an error string as a data result.

5. **Loop Termination:** The loop repeats until the plan is exhausted or a required step fails, routing based on the `execution_status` field, then terminates.

---

## Project Layout

```text
scrygent/
├── app.py
├── pyproject.toml           # Core dependencies
├── uv.lock                  # uv lockfile for deterministic local builds
├── README.md
├── docs/
│   ├── DESIGN.md
│   └── ARCHITECTURE.md
├── .streamlit/
│   └── secrets.toml
├── src/
│   ├── __init__.py
│   ├── state.py             # Pydantic BaseModel for graph state (includes df: Any)
│   ├── models.py            # Pydantic schemas (Plan, Step, Report, etc.)
│   ├── utils/
│   │   ├── __init__.py
│   │   └── llm.py           # Shared retry wrapper for all LLM calls
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── io.py
│   │   ├── profiler.py
│   │   ├── statistics.py
│   │   ├── wrangling.py     # Includes filter_rows; wrangling tools return (DataFrame, dict)
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
└── tests/
    ├── fixtures/
    │   └── sales_sample.csv
    ├── test_tools/
    ├── test_agents/
    └── test_graph/
```

---

## Tool Suite Reference (Tier 1)

All tools are pure functions in `src/tools/`. They never call the LLM.

| Tool File | Key Functions | Purpose |
|---|---|---|
| `io.py` | `load_csv`, `get_column_sample` | Safe data loading; extracts a strict 3-row sample purely for LLM formatting context |
| `profiler.py` | `profile_dataframe` | Structural summary: dtypes, nulls, stats per column |
| `statistics.py` | `calculate`, `arithmetic`, `correlation`, `linear_regression`, `groupby_agg`, `trend_detection`, `histogram_data` | All core statistical computations |
| `wrangling.py` | `fill_missing`, `cast_column_type`, `drop_columns`, `normalize_column`, `create_derived_column`, `filter_rows` | Data cleaning, transformation, and row filtering |
| `outliers.py` | `detect_outliers` (zscore, iqr) | Outlier detection with configurable method |
| `visualization.py` | `generate_plot` | Generates graphs, saves to disk, returns file paths and metadata |

**Key tool detail – `calculate`:**
A parameterised tool supporting operations: `mean`, `median`, `std`, `var`, `min`, `max`, `range`, `percentile`, `skew`, `kurtosis`, `iqr`, `outlier_bounds`, `correlate_with`. This single tool eliminates the need for a dozen separate functions.

**Key tool detail – `arithmetic`:**
Safely evaluates basic mathematical expressions (e.g., `"a / b"`) with provided scalar variables. Relies exclusively on `asteval` or `numexpr` to avoid all security risks associated with hand-rolled `eval()` strings.

**Key tool detail – wrangling tools (`fill_missing`, `normalize_column`, `create_derived_column`, `filter_rows`, etc.):**
Wrangling tools are the only tools that modify the dataset. Their return contract is a `(DataFrame, summary_dict)` tuple. The executor receives both, updates `state.df` with the new DataFrame, and stores the summary dict in `step_outputs`. This ensures all downstream tools and sandbox steps operate on the most recent state of the data without reloading from disk.

**Key tool detail – `filter_rows`:**
Filters the working DataFrame to rows matching a condition expressed as a column name, a restricted operator, and a scalar value. Permitted operators: `>`, `<`, `>=`, `<=`, `==`, `!=`, `isin`, `notna`. No arbitrary code is accepted. This covers the large class of queries such as "only rows where sales > 100" without escalating to Tier 2, and is essential to the claim that Tier 1 covers 90%+ of real-world CSV questions.

**Key tool detail – `visualization.py`:**
Plots are never converted into base64 blobs inside `AgentState`. They are saved to temporary storage and only their file path and conceptual metadata are committed to state, preventing memory bloat in Streamlit Cloud container runtimes.

---

## Sandbox Execution Details (Tier 2)

**Example planner step:**

```json
{
  "action": "sandbox",
  "instruction": "Calculate the 95th percentile of normalized_sales after recent transformations."
}
```

**Executor handling sequence:**

1. Detects sandbox step.
2. Assembles a prompt containing the instruction, the current targeted data profile, and relevant intermediate outputs.
3. Calls a lightweight LLM (`code_writer_chain`) to produce Python code.
4. Passes the generated code and a read-only DataFrame copy to `execute_python`.
5. Stores the returned string output in state.

This avoids the "code generated blindly at planning time" trap.

---

## State Validation & Error Recovery Strategy

### Broad Exception Handling in the Executor

All tool calls and sandbox executions are wrapped in a `try...except Exception` block. The correction chain is not limited to Pydantic `ValidationError` — any exception (including `KeyError` when a column is missing, `ValueError` from a failed operation, or a runtime error from the sandbox) is caught, converted to a string, and passed to the correction chain with the same retry logic (capped at 2 retries). This ensures no unhandled exception can bypass the graceful abort path.

### Local Pydantic Execution Correction

When a tool execution encounters a validation anomaly, the graph does not trigger a global back-edge loop to the Planner Node. Instead, the `ExecutorNode` contains a dedicated internal `correction_chain`. This chain issues a localised LLM correction prompt — passing the bad parameters, the target schema, and the Pydantic error details — forcing a rapid correction within the current step execution phase itself (capped at 2 retries). This is a Python-level loop internal to the executor function, not a LangGraph graph edge.

### Explicit Failure Control

Every `Step` in the plan carries a mandatory `required: bool` field (defaulting to `True`).

- If a step marked `required: True` exhausts its internal recovery retries, the executor sets `execution_status` to `"aborted"`, the graph routes to the terminal abort state, and a clear error message is surfaced to the user.
- If `required: False`, the executor logs the warning to `error_log` and proceeds to the next step in the plan.

---

## Data Fidelity Guarantee

- Raw **bulk** data never enters the LLM prompt. A strict 3-row sample is passed via the profiler solely to allow the Planner and Code Writer to infer string formats (e.g., date formatting). The entire dataset remains isolated.
- All numbers in reports are either directly returned by Tier 1 tools or captured from Tier 2 sandbox output strings.
- This guarantee is stated explicitly in both this document and the README.

---

## Testing Strategy

| Layer | Approach |
|---|---|
| Unit tests | Every tool function tested with known DataFrames |
| Node tests | Planner, executor, and reporter tested with mocked LLM |
| Integration tests | Full graph run with a sample CSV, asserting final report structure |
| Security tests | Verify `execute_python` rejects forbidden imports and times out |
| Retry tests | Verify retry wrapper fires on `RateLimitError` and `APIConnectionError`, surfaces clean error after exhausting retries |
| CI | Pytest runs on every push via `uv run pytest` |

---

## Deployment Notes

- Streamlit Cloud handles deployment and reads `pyproject.toml` for dependencies.
- `uv` and `uv.lock` are used for fast, deterministic local development. Streamlit Cloud executes installation via standard `pip`.
- To prevent temporary plot files from disappearing when Streamlit rerenders the UI, `app.py` mirrors all generated file paths from `AgentState` into `st.session_state`.
- `GROQ_API_KEY` is set in the Streamlit dashboard secrets. The `src/utils/llm.py` retry wrapper reads this key once at startup.

---

## What We Are Not Building (Yet)

- `skills/` evaluation framework — complex, needs a working agent first.
- Persistent database architectures — relying on local disk lifecycles managed by Streamlit's container session.
- Multi-user support or authentication.
