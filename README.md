# scrygent

Upload a CSV, ask a question in plain English, get a structured report with statistics, findings, and charts. No back and forth, no prompting, no manual analysis.

---

## What it does

Most CSV analysis workflows look like this: open the file, write some pandas, tweak the query, write more pandas, export something, repeat. scrygent replaces that loop for the common case.

You upload a file and describe what you want to know. scrygent profiles the data, plans a sequence of analysis steps, runs them using deterministic Python functions, and hands you a report. The numbers in that report come from code, not from a language model guessing.

It is not a chatbot. It does not ask clarifying questions. It runs once and produces a complete output.

---

## How it works

scrygent uses a plan-and-execute architecture built on LangGraph.

1. **Profiler** reads the CSV and builds a structural summary: column types, null rates, statistics for relevant columns, and a small row sample for format inference. No LLM is involved at this stage.

2. **Planner** receives the profile and your query and produces a concrete list of steps to execute. Each step names a specific tool and its parameters.

3. **Executor** runs the steps one at a time. For standard operations (statistics, grouping, filtering, correlation, outlier detection, plots) it dispatches to a curated set of deterministic Python functions. For rare calculations that fall outside those functions, it generates and runs sandboxed Python code with strict constraints: no network access, no file access, a 5-second timeout, and a limited module whitelist.

4. **Reporter** receives all the tool outputs and writes the final report. Its prompt instructs it to use only the verified numbers from those outputs.

The language model decides what to compute. Python computes it.

---

## Reliability approach

The numbers in scrygent reports come from deterministic functions, not from the model producing them from memory. This is a hard architectural constraint, not a prompt instruction.

The standard tool suite covers summaries, correlations, group aggregations, outlier detection, linear regression, trend detection, row filtering, and chart generation. Custom sandbox execution is a fallback for cases outside that scope and is flagged clearly in the UI when it activates.

All LLM calls go through a retry wrapper with exponential backoff. If the API is unavailable after retries, the run fails with a specific error rather than a silent wrong answer.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| UI | Streamlit | Fast to build, free deployment tier |
| Agent framework | LangGraph | Explicit graph structure, clean state management |
| LLM | Groq `llama-3.3-70b-versatile` | Free tier, fast, structured output support |
| Data | pandas, NumPy, SciPy | Standard, well-tested, no surprises |
| Safe eval | asteval / numexpr | No native `eval()` anywhere |
| Testing | pytest + Hypothesis | Property-based tests for tool functions |
| CI | GitHub Actions | Runs `uv run pytest` on every push |
| Packaging | uv | Fast, deterministic local builds |

---

## Project status

Under active development. The tool layer is being built first. LangGraph graph assembly follows once the tool suite is complete and tested.

- [x] DESIGN.md and ARCHITECTURE.md finalized
- [ ] `src/tools/io.py` (in progress)
- [ ] `src/tools/profiler.py`
- [ ] `src/tools/statistics.py`
- [ ] `src/tools/wrangling.py`
- [ ] `src/tools/outliers.py`
- [ ] `src/tools/visualization.py`
- [ ] `src/sandbox/executor.py`
- [ ] Agent nodes
- [ ] Graph wiring
- [ ] Streamlit UI
- [ ] Deployment

---

## Local setup

```bash
git clone https://github.com/mohamadmeri/scrygent
cd scrygent
uv sync
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# add your GROQ_API_KEY to secrets.toml
uv run streamlit run app.py
```

---

## Running tests

```bash
uv run pytest
```

---

## Design and architecture

Full design decisions, trade-offs, and architectural commitments are in [`docs/DESIGN.md`](docs/DESIGN.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
