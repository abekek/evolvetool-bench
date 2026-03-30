"""Domain A, Session 1: CSV/JSON transformation with hashing.

Session structure (12 tasks):
  Seed 1-3:  Use provided tools (read_csv, write_json, compute_hash)
  Gap 1:     Parse custom delimiter CSV (pipe-separated)
  Gap 2:     Convert nested JSON to flat CSV
  Variant 1: Parse custom delimiter CSV (tab-separated) — should REUSE gap_1's tool
  Variant 2: Convert nested JSON with arrays to flat CSV — should REUSE gap_2's tool
  Compose 1: Parse pipe CSV → compute hash of each row → write JSON — chains gap_1 + seed
  Regress 1: Re-run the pipe CSV task — should still work
  Adversarial 1: Parse CSV with embedded delimiters in quoted fields — should REFINE gap_1
  Adversarial 2: Convert JSON with null values and mixed types — should REFINE gap_2
"""

from ...types import Task, TaskType, Session


# ── Seed tools (provided to agent) ──────────────────────────────────

SEED_TOOLS = [
    {
        "name": "read_csv",
        "description": "Read a standard comma-separated CSV string and return list of dicts.",
        "implementation": '''
def read_csv(csv_string: str) -> list[dict]:
    """Read a standard CSV string (comma-separated) into a list of dicts."""
    import csv
    import io
    reader = csv.DictReader(io.StringIO(csv_string))
    return [dict(row) for row in reader]
''',
    },
    {
        "name": "write_json",
        "description": "Convert a list of dicts to a JSON string.",
        "implementation": '''
def write_json(data: list[dict]) -> str:
    """Convert a list of dicts to a formatted JSON string."""
    import json
    return json.dumps(data, indent=2)
''',
    },
    {
        "name": "compute_hash",
        "description": "Compute SHA-256 hash of a string.",
        "implementation": '''
def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of a string."""
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()
''',
    },
]


# ── Test data ────────────────────────────────────────────────────────

PIPE_CSV = 'name|age|city\nAlice|30|NYC\nBob|25|SF\nCharlie|35|LA'
TAB_CSV = 'name\tage\tcity\nAlice\t30\tNYC\nBob\t25\tSF'
QUOTED_PIPE_CSV = 'name|bio|city\nAlice|"Data|Scientist"|NYC\nBob|"ML|Engineer"|SF'

NESTED_JSON = [
    {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}, "tags": ["admin", "user"]},
    {"name": "Bob", "address": {"city": "SF", "zip": "94102"}, "tags": ["user"]},
]

NESTED_JSON_NULLS = [
    {"name": "Alice", "address": {"city": "NYC", "zip": None}, "tags": []},
    {"name": None, "address": None, "tags": ["user"]},
    {"name": "Charlie", "address": {"city": "LA", "zip": "90001"}, "tags": None},
]


# ── Tasks ────────────────────────────────────────────────────────────

TASKS = [
    # Seed tasks — use provided tools
    Task(
        id="seed_1",
        description=(
            "Read this CSV and convert it to JSON:\n"
            "name,age,city\nAlice,30,NYC\nBob,25,SF"
        ),
        task_type=TaskType.SEED,
        expected=[{"name": "Alice", "age": "30", "city": "NYC"}, {"name": "Bob", "age": "25", "city": "SF"}],
    ),
    Task(
        id="seed_2",
        description='Compute the SHA-256 hash of the string "benchmark".',
        task_type=TaskType.SEED,
        expected="54c7460cda2519a31d4ec1b20c03e674e6e00753bbf1f8b4c2e8f35e41003171",
    ),
    Task(
        id="seed_3",
        description=(
            'Convert this data to JSON: [{"x": 1, "y": 2}, {"x": 3, "y": 4}]'
        ),
        task_type=TaskType.SEED,
    ),

    # Gap tasks — require new tools
    Task(
        id="gap_1",
        description=(
            "Parse this PIPE-SEPARATED CSV (using | as delimiter) and return as list of dicts:\n"
            f"{PIPE_CSV}"
        ),
        task_type=TaskType.GAP,
        expected=[
            {"name": "Alice", "age": "30", "city": "NYC"},
            {"name": "Bob", "age": "25", "city": "SF"},
            {"name": "Charlie", "age": "35", "city": "LA"},
        ],
        hidden_tests=[
            {"input": {"csv_string": "a|b\n1|2\n3|4", "delimiter": "|"}, "expected": [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]},
            {"input": {"csv_string": "x|y|z\n10|20|30", "delimiter": "|"}, "expected": [{"x": "10", "y": "20", "z": "30"}]},
        ],
        adversarial_tests=[
            {"input": {"csv_string": "", "delimiter": "|"}},
            {"input": {"csv_string": "a|b", "delimiter": "|"}},  # header only
            {"input": {"csv_string": "a|b\n1|2|3", "delimiter": "|"}},  # extra column
        ],
    ),
    Task(
        id="gap_2",
        description=(
            "Convert this nested JSON to a flat CSV string. Flatten nested objects with dot notation "
            "(e.g., address.city) and join arrays with semicolons:\n"
            f'{NESTED_JSON}'
        ),
        task_type=TaskType.GAP,
        hidden_tests=[
            {
                "input": {"data": [{"a": {"b": 1}, "c": [2, 3]}]},
                "verify": "'a.b' in result and '2;3' in result",
            },
        ],
        adversarial_tests=[
            {"input": {"data": []}},
            {"input": {"data": [{}]}},
            {"input": {"data": [{"a": {"b": {"c": "deep"}}}]}},  # 3 levels deep
        ],
    ),

    # Variant tasks — should REUSE tools from gap tasks
    Task(
        id="variant_1",
        description=(
            "Parse this TAB-SEPARATED CSV (using tab as delimiter) and return as list of dicts:\n"
            f"{TAB_CSV}"
        ),
        task_type=TaskType.VARIANT,
        reuses_task="gap_1",
        expected=[
            {"name": "Alice", "age": "30", "city": "NYC"},
            {"name": "Bob", "age": "25", "city": "SF"},
        ],
    ),
    Task(
        id="variant_2",
        description=(
            "Convert this nested JSON (with arrays) to flat CSV. Same rules as before — "
            "dot notation for nested objects, semicolons for arrays:\n"
            f'{[{"name": "Dan", "scores": {"math": 95, "english": 88}, "hobbies": ["chess", "coding"]}]}'
        ),
        task_type=TaskType.VARIANT,
        reuses_task="gap_2",
    ),

    # Compose task — chain multiple self-created tools
    Task(
        id="compose_1",
        description=(
            "Parse this pipe-separated CSV, then compute the SHA-256 hash of each person's name, "
            "and return a JSON array with objects containing 'name' and 'name_hash':\n"
            f"{PIPE_CSV}"
        ),
        task_type=TaskType.COMPOSE,
        composes_tasks=["gap_1", "seed_2"],
    ),

    # Regress task — re-test earlier capability
    Task(
        id="regress_1",
        description=(
            "Parse this PIPE-SEPARATED CSV and return as list of dicts:\n"
            "product|price|quantity\nWidget|9.99|100\nGadget|24.99|50"
        ),
        task_type=TaskType.REGRESS,
        reuses_task="gap_1",
        expected=[
            {"product": "Widget", "price": "9.99", "quantity": "100"},
            {"product": "Gadget", "price": "24.99", "quantity": "50"},
        ],
    ),

    # Adversarial tasks — break naive implementations
    Task(
        id="adversarial_1",
        description=(
            "Parse this PIPE-SEPARATED CSV where values contain pipes inside quoted fields:\n"
            f"{QUOTED_PIPE_CSV}"
        ),
        task_type=TaskType.ADVERSARIAL,
        breaks_task="gap_1",
        expected=[
            {"name": "Alice", "bio": "Data|Scientist", "city": "NYC"},
            {"name": "Bob", "bio": "ML|Engineer", "city": "SF"},
        ],
    ),
    Task(
        id="adversarial_2",
        description=(
            "Convert this nested JSON to flat CSV. Handle null values and None entries gracefully:\n"
            f'{NESTED_JSON_NULLS}'
        ),
        task_type=TaskType.ADVERSARIAL,
        breaks_task="gap_2",
    ),
]


def create_session() -> Session:
    return Session(
        id="data_transform_s1",
        name="CSV/JSON Transformation with Hashing",
        domain="data_transform",
        tasks=TASKS,
        seed_tools=SEED_TOOLS,
        description="Tests tool creation for custom CSV parsing and JSON flattening, "
                    "with reuse, composition, and adversarial variants.",
    )
