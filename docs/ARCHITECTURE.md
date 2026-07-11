# Architecture Document – Scrygent

This document defines the structural constraints, data flow, and design decisions of the Scrygent engine. 

## 1. The Dependency Golden Rule

Scrygent enforces a strict, top-down dependency hierarchy. Lower layers must NEVER import from upper layers. This guarantees zero circular dependencies and highly testable modules.

1.  **`contracts/`**: The absolute bottom layer. Contains closed-vocabulary `StrEnum` definitions (e.g., `ToolName`, `FilterOperator`). Imports nothing.
2.  **`ir/`**: The Intermediate Representation layer. Defines the strict Pydantic schemas that the tools consume (e.g., `AnalyzeDataParams`). Imports from `contracts/`.
3.  **`models/`**: The State layer. Defines `AgentState`, the execution `Plan`, and the `TOOL_PARAM_MODELS` registry. Imports from `ir/` and `contracts/`.
4.  **`tools/`**: The deterministic execution engine. Pure Python functions that consume standard types and return standard dictionaries. Imports from `ir/` (for type hints) and `contracts/`.
5.  **`agents/`**: The LangGraph Nodes (`profiler_node`, `planner_node`, etc.). Imports tools, models, and contracts.
6.  **`graph/` & `app.py`**: The orchestration and UI layers. Ties the nodes together.

## 2. Core Design Decisions

### The Plan-and-Execute Compiler
Traditional agents expose Python wrappers (e.g., `filter()`, `groupby()`) and ask the LLM to write a script. This fails because LLMs struggle to track invisible data state across multiple steps. 
Scrygent uses an **Intermediate Representation (IR)**. The `AnalyzeDataParams` schema allows the LLM to output a single, declarative JSON block describing *what* to calculate. The Python compiler handles the *how*.

### Strict Determinism (No Sandbox)
Scrygent does not use `exec()` or sandboxed Python generation. All cross-column arithmetic and metric evaluation is securely routed through `numexpr` with `global_dict={}` to prevent namespace escapes. If a user request falls outside the deterministic tool suite, the system safely aborts rather than hallucinating code.

### The Multi-Step Composition Pattern
To handle complex workflows (e.g., filter the dataset, then run a regression on the subset), tools never pass massive DataFrames through the LangGraph state. 
Transforming tools (`filter_dataset`, `derive_column`) write their output to a secure, temporary CSV via `pathlib.Path` and update `AgentState.current_csv_path`. Subsequent steps naturally inherit the filtered data. A `reset_dataset` tool allows the Planner to revert to `original_csv_path` at any time.

## 3. The Orchestration Loop

Scrygent operates as a single-pass `StateGraph` driven by `AgentState.execution_status`.

```text
Profiler -> Planner -> Executor (Loop) -> Reporter -> END
```

### The Constrained Re-Plan Loop
If the Planner requires statistics for a column that was not initially profiled, it must halt and output a single-step plan: `request_column_stats`. 
The Executor runs this tool, enriches the `CSVProfile`, and sets `execution_status = "replan"`. LangGraph routes execution back to the Planner, which wakes up with the new data and writes the final, mathematically sound plan. This guarantees the LLM never guesses data distributions.

### The Self-Healing Correction Loop
If the Planner hallucinates a column name or violates a Pydantic IR constraint, the Executor traps the Python `ValueError`. It does not trigger a LangGraph edge. Instead, it triggers an internal LLM `correction_chain`, feeding the exact Python traceback back to the LLM to fix its parameter JSON mid-flight.

## 4. Semantic Memory (Serverless RAG)
Scrygent implements Experience Replay to learn from successful executions. 
Located in `src/scrygent/memory/`, the module connects to an Upstash Vector database via REST.
*   **Read:** The Planner queries the DB to find structurally similar past plans and injects them as Few-Shot examples in its prompt.
*   **Write:** If the Reporter node is reached (proving the plan executed flawlessly), the query and the validated JSON plan are embedded and committed to long-term memory.

## 5. JSON Boundary Sanitization
Pandas operations inherently produce C-types (`np.int64`, `np.nan`, `pd.Timestamp`) that crash standard JSON serializers. Scrygent mitigates this at the model boundary. `AgentState` inherits from `ScrygentBaseModel`, which utilizes a recursive `@model_validator` to scrub and cast all incoming data to native Python types before validation occurs. The state remains hermetically sealed and JSON-safe at all times.