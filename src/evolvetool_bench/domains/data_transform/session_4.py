"""Domain A, Session 4: Aggregation & Statistics.

Session structure (11 tasks):
  Seed 1-3:  Use provided tools (read_csv, write_json, compute_hash)
  Gap 1:     Group-by aggregation on CSV data (sum, count, avg per group)
  Gap 2:     Compute running/rolling statistics (running mean, cumulative sum)
  Variant 1: Different grouping key and aggregation function — should REUSE gap_1's tool
  Variant 2: Different rolling window size — should REUSE gap_2's tool
  Compose 1: Parse CSV, aggregate by group, format as JSON report
  Regress 1: Re-run group-by aggregation — should still work
  Adversarial 1: Empty groups, single-element groups, all same group
  Adversarial 2: Numeric overflow, NaN values, negative numbers in running stats
"""

from ...types import Task, TaskType, Session
from .seed_tools import SEED_TOOLS


# ── Test data ────────────────────────────────────────────────────────

SALES_CSV = (
    "region,product,revenue,units\n"
    "East,Widget,1200,10\n"
    "West,Widget,800,8\n"
    "East,Gadget,1500,5\n"
    "West,Gadget,900,3\n"
    "East,Widget,600,6\n"
    "West,Gizmo,2000,4"
)

EMPLOYEE_CSV = (
    "department,name,salary,years\n"
    "Engineering,Alice,95000,5\n"
    "Engineering,Bob,88000,3\n"
    "Marketing,Charlie,72000,7\n"
    "Engineering,Diana,102000,8\n"
    "Marketing,Eve,68000,2\n"
    "Sales,Frank,78000,4"
)

TIMESERIES_DATA = [
    {"date": "2025-01-01", "value": 10},
    {"date": "2025-01-02", "value": 20},
    {"date": "2025-01-03", "value": 15},
    {"date": "2025-01-04", "value": 25},
    {"date": "2025-01-05", "value": 30},
    {"date": "2025-01-06", "value": 10},
    {"date": "2025-01-07", "value": 20},
]

STOCK_PRICES = [
    {"date": "2025-01-01", "price": 100.0},
    {"date": "2025-01-02", "price": 102.5},
    {"date": "2025-01-03", "price": 99.0},
    {"date": "2025-01-04", "price": 103.0},
    {"date": "2025-01-05", "price": 107.5},
]

EDGE_CASE_CSV = (
    "group,value\n"
    "A,10\n"
    "A,20\n"
    "A,30\n"
    "B,100\n"
    "C,5\n"
    "C,5"
)

OVERFLOW_DATA = [
    {"date": "day1", "value": 1e308},
    {"date": "day2", "value": 1e308},
    {"date": "day3", "value": -1e308},
    {"date": "day4", "value": 0},
    {"date": "day5", "value": float("nan")},
    {"date": "day6", "value": 42},
]


# ── Tasks ────────────────────────────────────────────────────────────

TASKS = [
    # Seed tasks — use provided tools
    Task(
        id="seed_1",
        description=(
            "Read this CSV and convert to JSON:\n" + SALES_CSV
        ),
        task_type=TaskType.SEED,
        expected=[
            {"region": "East", "product": "Widget", "revenue": "1200", "units": "10"},
            {"region": "West", "product": "Widget", "revenue": "800", "units": "8"},
            {"region": "East", "product": "Gadget", "revenue": "1500", "units": "5"},
            {"region": "West", "product": "Gadget", "revenue": "900", "units": "3"},
            {"region": "East", "product": "Widget", "revenue": "600", "units": "6"},
            {"region": "West", "product": "Gizmo", "revenue": "2000", "units": "4"},
        ],
    ),
    Task(
        id="seed_2",
        description='Compute the SHA-256 hash of the string "aggregation".',
        task_type=TaskType.SEED,
        expected="f9a554554474923f7dee7f8e36a2bc851ab904aab5cd642be789c8f54eed4684",
    ),
    Task(
        id="seed_3",
        description=(
            'Convert this summary to JSON:\n'
            '[{"group": "East", "total": 3300}, {"group": "West", "total": 3700}]'
        ),
        task_type=TaskType.SEED,
    ),

    # Gap tasks — require new tools
    Task(
        id="gap_1",
        description=(
            "Group this CSV data by 'region' and compute the sum of 'revenue' and the count of rows "
            "for each group. Return a list of dicts with keys: region, total_revenue, count.\n\n"
            f"{SALES_CSV}"
        ),
        task_type=TaskType.GAP,
        expected=[
            {"region": "East", "total_revenue": 3300, "count": 3},
            {"region": "West", "total_revenue": 3700, "count": 3},
        ],
        hidden_tests=[
            {
                "input": {
                    "csv_string": "cat,val\nA,10\nA,20\nB,5",
                    "group_by": "cat",
                    "aggregations": {"total": ("val", "sum"), "n": ("val", "count")},
                },
                "expected": [{"cat": "A", "total": 30, "n": 2}, {"cat": "B", "total": 5, "n": 1}],
            },
        ],
        adversarial_tests=[
            {"input": {"csv_string": "g,v\n", "group_by": "g", "aggregations": {"total": ("v", "sum")}}},
            {"input": {"csv_string": "g,v\nA,10", "group_by": "g", "aggregations": {"total": ("v", "sum")}}},
            {"input": {"csv_string": "g,v\nA,abc", "group_by": "g", "aggregations": {"total": ("v", "sum")}}},
        ],
    ),
    Task(
        id="gap_2",
        description=(
            "Compute a running (rolling) mean with a window size of 3 over the 'value' field. "
            "For the first elements where the full window is not available, use the available values. "
            "Return a list of dicts with 'date' and 'rolling_mean' (rounded to 2 decimal places).\n\n"
            f"Data: {TIMESERIES_DATA}"
        ),
        task_type=TaskType.GAP,
        expected=[
            {"date": "2025-01-01", "rolling_mean": 10.0},
            {"date": "2025-01-02", "rolling_mean": 15.0},
            {"date": "2025-01-03", "rolling_mean": 15.0},
            {"date": "2025-01-04", "rolling_mean": 20.0},
            {"date": "2025-01-05", "rolling_mean": 23.33},
            {"date": "2025-01-06", "rolling_mean": 21.67},
            {"date": "2025-01-07", "rolling_mean": 20.0},
        ],
        hidden_tests=[
            {
                "input": {
                    "data": [{"t": "a", "v": 2}, {"t": "b", "v": 4}, {"t": "c", "v": 6}, {"t": "d", "v": 8}],
                    "value_field": "v",
                    "date_field": "t",
                    "window": 2,
                },
                "expected": [
                    {"t": "a", "rolling_mean": 2.0},
                    {"t": "b", "rolling_mean": 3.0},
                    {"t": "c", "rolling_mean": 5.0},
                    {"t": "d", "rolling_mean": 7.0},
                ],
            },
        ],
        adversarial_tests=[
            {"input": {"data": [], "value_field": "v", "window": 3}},
            {"input": {"data": [{"v": 5}], "value_field": "v", "window": 10}},
        ],
    ),

    # Variant tasks — should REUSE tools from gap tasks
    Task(
        id="variant_1",
        description=(
            "Group this employee CSV by 'department' and compute the average 'salary' and "
            "the max 'years' for each department. Return dicts with: department, avg_salary, max_years.\n\n"
            f"{EMPLOYEE_CSV}"
        ),
        task_type=TaskType.VARIANT,
        reuses_task="gap_1",
        expected=[
            {"department": "Engineering", "avg_salary": 95000.0, "max_years": 8},
            {"department": "Marketing", "avg_salary": 70000.0, "max_years": 7},
            {"department": "Sales", "avg_salary": 78000.0, "max_years": 4},
        ],
    ),
    Task(
        id="variant_2",
        description=(
            "Compute a rolling mean with a window size of 2 over the 'price' field. "
            "Return dicts with 'date' and 'rolling_mean' (rounded to 2 decimal places).\n\n"
            f"Data: {STOCK_PRICES}"
        ),
        task_type=TaskType.VARIANT,
        reuses_task="gap_2",
        expected=[
            {"date": "2025-01-01", "rolling_mean": 100.0},
            {"date": "2025-01-02", "rolling_mean": 101.25},
            {"date": "2025-01-03", "rolling_mean": 100.75},
            {"date": "2025-01-04", "rolling_mean": 101.0},
            {"date": "2025-01-05", "rolling_mean": 105.25},
        ],
    ),

    # Compose task — parse CSV, aggregate, format as JSON report
    Task(
        id="compose_1",
        description=(
            "Parse this CSV, group by 'region', compute the sum of 'revenue' per region, "
            "then output the result as a formatted JSON string.\n\n"
            f"{SALES_CSV}"
        ),
        task_type=TaskType.COMPOSE,
        composes_tasks=["gap_1", "seed_3"],
    ),

    # Regress task — re-test aggregation
    Task(
        id="regress_1",
        description=(
            "Group this CSV by 'product' and compute the sum of 'units' for each product.\n\n"
            f"{SALES_CSV}"
        ),
        task_type=TaskType.REGRESS,
        reuses_task="gap_1",
        expected=[
            {"product": "Widget", "total_units": 24},
            {"product": "Gadget", "total_units": 8},
            {"product": "Gizmo", "total_units": 4},
        ],
    ),

    # Adversarial tasks — break naive implementations
    Task(
        id="adversarial_1",
        description=(
            "Group this CSV by 'group' and compute: sum, count, average, min, and max of 'value'. "
            "Handle single-element groups correctly (avg = value, min = max = value).\n\n"
            f"{EDGE_CASE_CSV}"
        ),
        task_type=TaskType.ADVERSARIAL,
        breaks_task="gap_1",
        expected=[
            {"group": "A", "sum": 60, "count": 3, "avg": 20.0, "min": 10, "max": 30},
            {"group": "B", "sum": 100, "count": 1, "avg": 100.0, "min": 100, "max": 100},
            {"group": "C", "sum": 10, "count": 2, "avg": 5.0, "min": 5, "max": 5},
        ],
    ),
    Task(
        id="adversarial_2",
        description=(
            "Compute running mean with window 3 on this data that includes extreme floats "
            "and NaN. NaN values should be excluded from the window (treat as missing). "
            "Return 'rolling_mean' rounded to 2 decimal places, or null if all values in window are NaN.\n\n"
            f"Data: {OVERFLOW_DATA}"
        ),
        task_type=TaskType.ADVERSARIAL,
        breaks_task="gap_2",
    ),
]


def create_session() -> Session:
    return Session(
        id="data_transform_s4",
        name="Aggregation & Statistics",
        domain="data_transform",
        tasks=TASKS,
        seed_tools=SEED_TOOLS,
        description="Tests tool creation for group-by aggregation and running statistics, "
                    "with reuse across grouping keys, composition, and adversarial edge cases.",
    )
