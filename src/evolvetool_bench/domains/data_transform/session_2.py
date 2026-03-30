"""Domain A, Session 2: Schema Validation & Mapping.

Session structure (11 tasks):
  Seed 1-3:  Use provided tools (read_csv, write_json, compute_hash)
  Gap 1:     Validate JSON data against a JSON-schema-like definition
  Gap 2:     Map fields between two different schemas (rename/restructure)
  Variant 1: Validate against a different schema — should REUSE gap_1's tool
  Variant 2: Map fields with a different mapping spec — should REUSE gap_2's tool
  Compose 1: Validate incoming data then map valid records to output schema
  Regress 1: Re-run validation with original schema — should still work
  Adversarial 1: Schema with optional fields and deeply nested required fields
  Adversarial 2: Mapping with missing source fields and type coercion
"""

from ...types import Task, TaskType, Session
from .seed_tools import SEED_TOOLS


# ── Test data ────────────────────────────────────────────────────────

USER_SCHEMA = {
    "type": "object",
    "required": ["name", "email", "age"],
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "age": {"type": "integer"},
        "role": {"type": "string"},
    },
}

PRODUCT_SCHEMA = {
    "type": "object",
    "required": ["sku", "price"],
    "properties": {
        "sku": {"type": "string"},
        "price": {"type": "number"},
        "description": {"type": "string"},
        "in_stock": {"type": "boolean"},
    },
}

NESTED_SCHEMA = {
    "type": "object",
    "required": ["id", "profile"],
    "properties": {
        "id": {"type": "integer"},
        "profile": {
            "type": "object",
            "required": ["first_name", "contact"],
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "contact": {
                    "type": "object",
                    "required": ["email"],
                    "properties": {
                        "email": {"type": "string"},
                        "phone": {"type": "string"},
                    },
                },
            },
        },
    },
}

VALID_USERS = [
    {"name": "Alice", "email": "alice@example.com", "age": 30},
    {"name": "Bob", "email": "bob@example.com", "age": 25, "role": "admin"},
]

INVALID_USERS = [
    {"name": "Charlie", "age": 28},  # missing email
    {"name": "Diana", "email": "diana@example.com", "age": "twenty"},  # wrong type
    {"name": "Eve", "email": "eve@example.com", "age": 35},  # valid
]

USER_TO_CONTACT_MAPPING = {
    "full_name": "name",
    "email_address": "email",
    "years_old": "age",
}

EMPLOYEE_TO_HR_MAPPING = {
    "employee_id": "id",
    "full_name": "name",
    "department_name": "dept",
    "annual_salary": "salary",
}

EMPLOYEES = [
    {"id": "E001", "name": "Alice Smith", "dept": "Engineering", "salary": 95000},
    {"id": "E002", "name": "Bob Jones", "dept": "Marketing", "salary": 72000},
]


# ── Tasks ────────────────────────────────────────────────────────────

TASKS = [
    # Seed tasks — use provided tools
    Task(
        id="seed_1",
        description=(
            "Read this CSV and convert it to JSON:\n"
            "id,name,email\n1,Alice,alice@test.com\n2,Bob,bob@test.com"
        ),
        task_type=TaskType.SEED,
        expected=[
            {"id": "1", "name": "Alice", "email": "alice@test.com"},
            {"id": "2", "name": "Bob", "email": "bob@test.com"},
        ],
    ),
    Task(
        id="seed_2",
        description='Compute the SHA-256 hash of the string "schema_validation".',
        task_type=TaskType.SEED,
        expected="b852786aa2fdcf159e1a6be61bc0ee20a8885c97dee0354f706833c2d322e123",
    ),
    Task(
        id="seed_3",
        description=(
            'Convert this data to a JSON string:\n'
            '[{"valid": true, "errors": []}, {"valid": false, "errors": ["missing field"]}]'
        ),
        task_type=TaskType.SEED,
    ),

    # Gap tasks — require new tools
    Task(
        id="gap_1",
        description=(
            "Validate each record in this list against the given schema and return a list of "
            "validation results. Each result should have 'valid' (bool) and 'errors' (list of strings).\n\n"
            f"Schema: {USER_SCHEMA}\n\n"
            f"Data: {INVALID_USERS}"
        ),
        task_type=TaskType.GAP,
        expected=[
            {"valid": False, "errors": ["missing required field: email"]},
            {"valid": False, "errors": ["field 'age' expected type integer, got str"]},
            {"valid": True, "errors": []},
        ],
        hidden_tests=[
            {
                "input": {
                    "schema": USER_SCHEMA,
                    "records": [{"name": "Test", "email": "t@t.com", "age": 1}],
                },
                "expected": [{"valid": True, "errors": []}],
            },
            {
                "input": {
                    "schema": USER_SCHEMA,
                    "records": [{}],
                },
                "verify": "result[0]['valid'] is False and len(result[0]['errors']) >= 3",
            },
        ],
        adversarial_tests=[
            {"input": {"schema": USER_SCHEMA, "records": []}},
            {"input": {"schema": {"type": "object", "required": [], "properties": {}}, "records": [{"any": "thing"}]}},
            {"input": {"schema": USER_SCHEMA, "records": [{"name": 123, "email": 456, "age": "abc"}]}},
        ],
    ),
    Task(
        id="gap_2",
        description=(
            "Map each record from the source schema to a target schema using the given field mapping. "
            "The mapping dict maps target_field -> source_field. Return a list of remapped dicts.\n\n"
            f"Mapping: {USER_TO_CONTACT_MAPPING}\n\n"
            f"Data: {VALID_USERS}"
        ),
        task_type=TaskType.GAP,
        expected=[
            {"full_name": "Alice", "email_address": "alice@example.com", "years_old": 30},
            {"full_name": "Bob", "email_address": "bob@example.com", "years_old": 25},
        ],
        hidden_tests=[
            {
                "input": {
                    "mapping": {"a": "x", "b": "y"},
                    "records": [{"x": 1, "y": 2, "z": 3}],
                },
                "expected": [{"a": 1, "b": 2}],
            },
        ],
        adversarial_tests=[
            {"input": {"mapping": {"a": "x"}, "records": [{}]}},
            {"input": {"mapping": {}, "records": [{"x": 1}]}},
            {"input": {"mapping": {"a": "missing_field"}, "records": [{"x": 1}]}},
        ],
    ),

    # Variant tasks — should REUSE tools from gap tasks
    Task(
        id="variant_1",
        description=(
            "Validate each record against the PRODUCT schema:\n\n"
            f"Schema: {PRODUCT_SCHEMA}\n\n"
            "Data: ["
            '{"sku": "WDG-001", "price": 9.99, "in_stock": true}, '
            '{"sku": "GDG-002", "price": "free"}, '
            '{"price": 14.99}'
            "]"
        ),
        task_type=TaskType.VARIANT,
        reuses_task="gap_1",
        expected=[
            {"valid": True, "errors": []},
            {"valid": False, "errors": ["field 'price' expected type number, got str"]},
            {"valid": False, "errors": ["missing required field: sku"]},
        ],
    ),
    Task(
        id="variant_2",
        description=(
            "Map these employee records to the HR schema:\n\n"
            f"Mapping: {EMPLOYEE_TO_HR_MAPPING}\n\n"
            f"Data: {EMPLOYEES}"
        ),
        task_type=TaskType.VARIANT,
        reuses_task="gap_2",
        expected=[
            {"employee_id": "E001", "full_name": "Alice Smith", "department_name": "Engineering", "annual_salary": 95000},
            {"employee_id": "E002", "full_name": "Bob Jones", "department_name": "Marketing", "annual_salary": 72000},
        ],
    ),

    # Compose task — validate then map
    Task(
        id="compose_1",
        description=(
            "First validate these records against the user schema. Then map ONLY the valid records "
            "to the contact schema using the field mapping. Return the mapped valid records.\n\n"
            f"Schema: {USER_SCHEMA}\n\n"
            f"Mapping: {USER_TO_CONTACT_MAPPING}\n\n"
            f"Data: {INVALID_USERS}"
        ),
        task_type=TaskType.COMPOSE,
        composes_tasks=["gap_1", "gap_2"],
        expected=[
            {"full_name": "Eve", "email_address": "eve@example.com", "years_old": 35},
        ],
    ),

    # Regress task — re-test validation
    Task(
        id="regress_1",
        description=(
            "Validate these records against the user schema:\n\n"
            f"Schema: {USER_SCHEMA}\n\n"
            f"Data: {VALID_USERS}"
        ),
        task_type=TaskType.REGRESS,
        reuses_task="gap_1",
        expected=[
            {"valid": True, "errors": []},
            {"valid": True, "errors": []},
        ],
    ),

    # Adversarial tasks — break naive implementations
    Task(
        id="adversarial_1",
        description=(
            "Validate this record against a schema with nested required fields. "
            "The validation must check required fields recursively in nested objects.\n\n"
            f"Schema: {NESTED_SCHEMA}\n\n"
            "Data: ["
            '{"id": 1, "profile": {"first_name": "Alice"}}, '
            '{"id": 2, "profile": {"first_name": "Bob", "contact": {"phone": "555-0100"}}}, '
            '{"id": 3, "profile": {"first_name": "Charlie", "contact": {"email": "c@test.com"}}}'
            "]"
        ),
        task_type=TaskType.ADVERSARIAL,
        breaks_task="gap_1",
        expected=[
            {"valid": False, "errors": ["missing required field: profile.contact"]},
            {"valid": False, "errors": ["missing required field: profile.contact.email"]},
            {"valid": True, "errors": []},
        ],
    ),
    Task(
        id="adversarial_2",
        description=(
            "Map these records using the given mapping. Some source fields are missing, "
            "and some values need type handling (None, nested dicts). "
            "Missing source fields should map to None.\n\n"
            "Mapping: {\"full_name\": \"name\", \"contact\": \"email\", \"status\": \"active\"}\n\n"
            "Data: ["
            '{"name": "Alice", "email": "a@test.com"}, '
            '{"name": "Bob"}, '
            '{"email": "c@test.com", "active": true}'
            "]"
        ),
        task_type=TaskType.ADVERSARIAL,
        breaks_task="gap_2",
        expected=[
            {"full_name": "Alice", "contact": "a@test.com", "status": None},
            {"full_name": "Bob", "contact": None, "status": None},
            {"full_name": None, "contact": "c@test.com", "status": True},
        ],
    ),
]


def create_session() -> Session:
    return Session(
        id="data_transform_s2",
        name="Schema Validation & Mapping",
        domain="data_transform",
        tasks=TASKS,
        seed_tools=SEED_TOOLS,
        description="Tests tool creation for JSON schema validation and field mapping, "
                    "with reuse across schemas, composition, and adversarial edge cases.",
    )
