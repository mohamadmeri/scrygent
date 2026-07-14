"""Translates the strict Pydantic IR into readable markdown for the Planner LLM.

Acts as the single source of truth for the LLM's understanding of tool parameter shapes,
preventing Pydantic validation mismatches.
"""

TOOL_SPECIFICATIONS = """
When specifying `parameters` for a tool, you MUST exactly match the schemas below. Do not invent fields.

## 1. analyze_data
Filters, groups, aggregates, sorts, and limits data.
Parameters:
  - metrics (required list):
      - column (str): The exact name of the column to aggregate.
      - aggregation (str): "mean", "sum", "count", "nunique", "min", "max",
        "std", "var", "median". 
        *(PRO-TIP: To calculate the proportion or percentage of a boolean/binary column, use the "mean" aggregation!)*
      - alias (str): Name for the output key (must be unique).
  - filters (optional list): List of filter conditions (see Shared Filter Schema below).
  - group_by (optional list[str]): Columns to GROUP BY.
  - sort (optional object): {"column": "string", "direction": "asc" | "desc"}
  *(CRITICAL: If you provided 'metrics', sort.column MUST exactly match one of your metric 'alias' names or a 'group_by' column. Do not use raw dataset column names here. If 'metrics' is omitted, sort.column must be a raw dataset column.)*
  - limit (optional int): Max rows to return (must be >= 1).

## 2. filter_dataset
Filters the dataset to a subset and updates the current working file.
Parameters:
- filters (required list): List of filter conditions (see Shared Filter Schema below).

## 3. normalize_column
Transforms a column's values.
Parameters:
- column (required str): Exact column name.
- method (required str): "min_max", "z_score", "log", "strip", "lowercase", "uppercase", "title_case".

## 4. reset_dataset
Reverts the working dataset back to the original uploaded file.
Parameters: {} (Must be an exactly empty dictionary)

## 5. correlation
Calculates statistical correlation between columns.
Parameters:
- columns (required list[str]): At least 2 columns.
- method (optional str): "pearson" (default), "spearman", "kendall".

## 6. regression
Fits a linear regression model.
Parameters:
- target (required str): The dependent variable.
- features (required list[str]): Independent variables (at least 1).
- method (optional str): "linear" (default).

## 7. detect_outliers
Finds anomalies in a single column.
Parameters:
- column (required str): Exact column name.
- method (optional str): "iqr" (default), "z_score".

## 8. request_column_stats
Batch requests detailed statistics for columns missing from the profile.
Parameters:
- columns (required list[str]): Columns to fetch.

## 9. generate_plot
Saves a visualization to disk. Parameters:
  - plot_type (required str): "bar", "line", "scatter", "histogram", "box", "heatmap".
  - columns (required list[str]): Target columns. For "bar", "line", and "scatter", you MUST provide EXACTLY TWO columns in this exact order: [x_axis_categorical, y_axis_numeric].
  - title (optional str): Plot title.

## 10. derive_column
Executes row-wise math to create a new column.
Parameters:
- new_column (required str): Name of the new column.
- expression (required str): Math formula using existing column names (e.g., "Revenue - Cost").

## 11. evaluate_metrics
Executes math on scalar values (e.g., calculating a ratio from previous step outputs).
Parameters:
- expression (required str): Math formula (e.g., "profit / revenue").
- values (required dict): Map of variable names to numbers (e.g., {"profit": 500, "revenue": 1000}).

---

### SHARED FILTER SCHEMA (used in analyze_data and filter_dataset)
If you use filters, each filter object MUST match ONE of these three shapes exactly:

1. Scalar Filter (comparing against one value)
   - column (str)
   - operator (str): "==", "!=", ">", "<", ">=", "<="
   - value (str | int | float | bool)

2. List Membership Filter (checking if value is in a list)
   - column (str)
   - operator (str): "in", "not in"
   - value (list): A non-empty list of values.

3. String Operation Filter
   - column (str)
   - operator (str): "contains", "startswith", "endswith"
   - value (str): A non-empty string.
"""


def get_tool_specs() -> str:
    """Returns the Markdown formatted tool specifications for the Planner."""
    return TOOL_SPECIFICATIONS
