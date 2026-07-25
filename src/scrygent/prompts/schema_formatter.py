"""Translates the strict Pydantic IR into compact schema definitions for the Planner LLM.

Acts as the single source of truth for tool parameter shapes, preventing validation mismatches.
"""

TOOL_SPECIFICATIONS = """
TOOL SCHEMAS (Strictly enforce these parameter shapes. Do not invent fields):

1. analyze_data
   - metrics: list[{column: str, aggregation: "mean"|"sum"|"count"|"nunique"|"min"|"max"|"std"|"var"|"median", alias: str}] (optional)
     [PRO-TIP 1: To calculate proportions of a boolean column, use "mean".]
     [PRO-TIP 2: To find the most frequent/common items, group by the category and use the "count" aggregation, NOT "nunique"!]
   - filters: list[FilterObject] (optional, see Shared Filter Schema below)
   - group_by: list[str] (optional)
   - sort: {column: str, direction: "asc"|"desc"} (optional). CRITICAL: If 'metrics' exists, sort.column MUST be a metric 'alias' or 'group_by' column. If no 'metrics', it must be a raw dataset column.
   - limit: int (optional, >= 1)

2. filter_dataset
   - filters: list[FilterObject] (required)

3. normalize_column
   - column: str (required)
   - method: "min_max"|"z_score"|"log"|"strip"|"lowercase"|"uppercase"|"title_case" (required)

4. reset_dataset
   - {} (MUST be an exactly empty dictionary)

5. correlation
   - columns: list[str] (required, min 2)
   - method: "pearson"|"spearman"|"kendall" (optional, default "pearson")

6. regression
   - target: str (required)
   - features: list[str] (required, min 1)
   - method: "linear" (optional, default "linear")

7. detect_outliers
   - column: str (required)
   - method: "iqr"|"z_score" (optional, default "iqr")

8. request_column_stats
   - columns: list[str] (required)

9. generate_plot
   - plot_type: "bar"|"line"|"scatter"|"histogram"|"box"|"heatmap" (required)
   - columns: list[str] (required). For bar/line/scatter: EXACTLY TWO columns in order [x_axis_categorical, y_axis_numeric].
   - title: str (optional)

10. derive_column
    - new_column: str (required)
    - expression: str (required, e.g., "Revenue - Cost")

11. evaluate_metrics
    - expression: str (required, e.g., "profit / revenue")
    - values: dict[str, float] (required, e.g., {"profit": 500})

SHARED FILTER SCHEMA (Each object in a 'filters' list MUST match ONE of these exactly):
- Scalar: {column: str, operator: "=="|"!="|">"|"<"|">="|"<=", value: str|int|float|bool}
- List: {column: str, operator: "in"|"not in", value: list[str|int|float]}
- String: {column: str, operator: "contains"|"startswith"|"endswith", value: str}

CRITICAL RULES:
1. MULTI-VALUE: Use list values for "in"/"not in" (e.g., {"column": "country", "operator": "in", "value": ["US", "China"]}). Never nest dicts in 'value'.
2. SCALAR ONLY: The 'value' field must be a primitive. NEVER nest a tool call or dict inside 'value'.
3. EXACT MATCH: All column names MUST exactly match keys in `global_schema`. Do not pluralize or guess.
"""


def get_tool_specs() -> str:
    """Returns the compact, strict tool specifications for the Planner."""
    return TOOL_SPECIFICATIONS
