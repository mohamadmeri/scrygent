<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark-theme.jpeg">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/logo-light-theme.jpeg">
    <img alt="Scrygent Logo" src="docs/assets/logo-light-theme.png" width="120">
  </picture>

  # Scrygent
  
  **A Strictly Typed Compiler Engine for Data Analysis**
  
  [![CI](https://github.com/mohamadmeri/scrygent/actions/workflows/ci.yml/badge.svg)](https://github.com/mohamadmeri/scrygent/actions/workflows/ci.yml)
  [![Codecov](https://codecov.io/gh/mohamadmeri/scrygent/branch/main/graph/badge.svg)](https://codecov.io/gh/mohamadmeri/scrygent)
  [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  [![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

<br/>

<div align="center">
  <!-- TODO: Insert final UI screenshot here -->
  <img src="docs/assets/ui_screenshot.png" alt="Scrygent UI" width="100%" style="border-radius: 8px; border: 1px solid #333;" />
  <p><em>The IDE-style compilation interface. (Deployment: WIP)</em></p>
</div>

<br/>

> **Scrygent is a strictly typed compiler engine for data analysis.** It translates natural language into static, immutable execution graphs. The LLM decides *what* to compute. The deterministic Python engine decides *how*. Zero code generation. Zero hallucinated mathematics.

---

## Navigation

| Module | Description |
| :--- | :--- |
| [**Architecture**](docs/ARCHITECTURE.md) | Deep dive into the 3-pass compiler, dependency hierarchy, and self-healing loops. |
| [**Benchmarks (WIP)**](#benchmarks--evaluation-wip) | Empirical evaluation metrics against DABench and DataBench Lite. *(WIP)* |

---

## System Architecture

Scrygent abandons the fragile "ReAct" loop of generating and executing arbitrary Python. Instead, it utilizes a **Plan-and-Execute Compiler** pipeline.

```mermaid
flowchart LR
    U[User Query] --> PL[3-Pass Planner LLM]
    CSV[(CSV Dataset)] --> P[Profiler Node]
    P -->|Global Schema & Stats| PL
    PL -->|Strict Pydantic IR| EX[Deterministic Executor]
    EX -->|Verified JSON Outputs| R[Reporter LLM]
    R --> OUT[Final Report]
    
    style U fill:#1C1A18,stroke:#5EEAD4,stroke-width:2px,color:#F5F0EB
    style CSV fill:#1C1A18,stroke:#7FB069,stroke-width:2px,color:#F5F0EB
    style P fill:#1C1A18,stroke:#7FB069,stroke-width:2px,color:#F5F0EB
    style PL fill:#1C1A18,stroke:#F59E0B,stroke-width:2px,color:#F5F0EB
    style EX fill:#1C1A18,stroke:#7FB069,stroke-width:2px,color:#F5F0EB
    style R fill:#1C1A18,stroke:#F59E0B,stroke-width:2px,color:#F5F0EB
    style OUT fill:#1C1A18,stroke:#5EEAD4,stroke-width:2px,color:#F5F0EB
```

1. **Profiler:** Extracts global schemas and query-aware statistics deterministically. This minimizes prompt size.
2. **Planner:** Translates intent into strict JSON using a 3-pass compiler (Parser → Optimizer → IR Emitter).
3. **Executor:** Dispatches validated payloads to a handwritten, stateless suite of pure Python tools.
4. **Reporter:** Synthesizes the final report. It strictly constrains output to verified tool results.

---

## Key Engineering Highlights

These implementation details form the core of Scrygent's reliability.

*   **The 3-Pass Compiler Pipeline:** Forcing an LLM to simultaneously reason about data logic and format nested JSON causes cognitive overload. Scrygent separates this into three distinct passes: an abstract Parser, a heuristic Optimizer, and a strict IR Emitter locked behind `json_mode`.
*   **The Hermetic JSON Boundary:** Pandas and NumPy operations produce C-types (`np.int64`, `pd.NaT`, `np.nan`). These C-types crash standard JSON serializers. Scrygent intercepts this via `ScrygentBaseModel`. It applies a recursive `@model_validator(mode="wrap")` to scrub and cast all data to native Python primitives at the exact moment of state assignment.
*   **Self-Healing Execution with Actionable Context:** Execution failures do not crash the system. The Python exception catches the error. It enriches the error with the *exact list of available columns*. The system routes this error through an internal LLM correction loop. The LLM repairs its own payload syntax mid-flight.
*   **The Multi-Step Composition Pattern:** Tools never pass massive DataFrames through the LangGraph state. Transforming tools (`filter_dataset`, `derive_column`) write their output to a secure, temporary CSV. They update `AgentState.current_csv_path`. Subsequent steps inherit the filtered data. This maintains the stateless-tools rule.
*   **Semantic Experience Replay:** Successful execution plans automatically embed and store in a Qdrant vector database. Future queries retrieve structurally similar past plans as few-shot examples. This allows the planner to improve over time without retraining.

---
## Benchmarks & Evaluation (WIP)

Scrygent uses industry-standard benchmarks to validate its deterministic approach against code-generation agents. This section is a Work in Progress.

### Active Benchmarks

| Benchmark | Dataset | Questions | Status |
| :--- | :--- | :--- | :--- |
| **InfiAgent-DABench** (ICML 2024) | 124 CSVs | 603 questions | 🟡 WIP |
| **DataBench Lite** (SemEval 2025) | 80 datasets | 1,822 questions | 🟡 WIP |

### Evaluation Methodology

- **Mode:** `eval_mode=True` in `AgentState` forces the Reporter to output only `DirectAnswer`. It drops narrative and plots.
- **Baseline:** GPT-4 achieves **78.99%** accuracy on DABench.
- **Target:** Llama 3.3 70B backbone with deterministic advantage on standard statistical queries.
- **Metrics Tracked:**
  - Planner schema validity rate
  - Correction-loop success rate  
  - End-to-end task completion
  - Latency breakdown by compiler pass
  - Schema failure rate (with vs. without 3-pass split)

### Robustness Testing

The system generates "poisoned" variants of clean benchmark CSVs to test error handling:
- UTF-16/CP1252 encodings
- Semicolon/pipe delimiters
- Mixed-type columns (numeric + string artifacts)
- Missing headers, offset data rows

**See [`scripts/run_benchmark.py`](scripts/run_benchmark.py) for the evaluation harness.**

---

## Technology Stack

| Layer | Technology | Rationale |
| :--- | :--- | :--- |
| **Workflow Orchestration** | LangGraph | Explicit graph structure, clean cyclic routing, strict state management. |
| **Data Validation** | Pydantic v2 | Recursive serialization and strict JSON schema enforcement at boundaries. |
| **Data Engine** | Pandas 3.x, NumPy | Standard, strictly-typed C-backend data manipulation. |
| **Safe Math** | numexpr | Securely evaluates row-wise math without exposing Python's `eval()`. |
| **LLM Providers** | Groq, OpenRouter | Provider-agnostic abstraction for fast structured JSON generation. |
| **Long-Term Memory** | Qdrant, HuggingFace | Serverless vector DB and embeddings for Experience Replay. |
| **UI** | Streamlit | Modular, IDE-style presentation layer. |
| **Dependency Mgmt** | uv | Fast, deterministic builds and strict lockfile resolution. |

---

## Local Development

Clone the repository and initialize the environment using `uv`:

```bash
git clone https://github.com/mohamadmeri/scrygent.git
cd scrygent
uv sync
```

Create a secrets file and populate it with your API credentials:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

```toml
# .streamlit/secrets.toml
GROQ_API_KEY = "..."
OPENROUTER_API_KEY = "..."
QDRANT_URL = "..."
QDRANT_API_KEY = "..."
HF_API_TOKEN = "..."
```

Run the application:

```bash
uv run streamlit run app.py
```

---

## Deep Dive Documentation

For a comprehensive explanation of the compiler architecture, the Dependency Golden Rule, graph routing mechanics, and the self-healing correction loops, see:

**[Read the Architecture Document](docs/ARCHITECTURE.md)**