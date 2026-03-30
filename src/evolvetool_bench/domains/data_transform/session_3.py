"""Domain A, Session 3: Text Processing & Extraction.

Session structure (11 tasks):
  Seed 1-3:  Use provided tools (read_csv, write_json, compute_hash)
  Gap 1:     Extract structured data from unstructured text (emails, log lines)
  Gap 2:     Regex pattern matching — find and extract all matches with named groups
  Variant 1: Extract from different text format (server logs) — should REUSE gap_1's tool
  Variant 2: Different regex pattern on different text — should REUSE gap_2's tool
  Compose 1: Extract log entries then aggregate counts by level
  Regress 1: Re-run extraction on original log format — should still work
  Adversarial 1: Malformed text with unicode, missing fields, multiline values
  Adversarial 2: Regex with catastrophic backtracking potential, overlapping matches
"""

from ...types import Task, TaskType, Session
from .seed_tools import SEED_TOOLS


# ── Test data ────────────────────────────────────────────────────────

EMAIL_TEXT = """From: alice@example.com
To: bob@example.com
Subject: Q3 Report
Date: 2025-10-15

Hi Bob,
Please find the Q3 report attached.
Best, Alice

---

From: charlie@example.com
To: diana@example.com
Subject: Meeting Tomorrow
Date: 2025-10-16

Diana,
Can we reschedule to 3pm?
Thanks, Charlie"""

SERVER_LOGS = """2025-10-15 08:23:01 [INFO] Server started on port 8080
2025-10-15 08:23:05 [INFO] Connected to database
2025-10-15 08:24:12 [WARN] Slow query detected (1532ms)
2025-10-15 08:25:00 [ERROR] Connection timeout to redis:6379
2025-10-15 08:25:01 [INFO] Retrying connection...
2025-10-15 08:25:03 [ERROR] Connection failed after 3 retries
2025-10-15 08:25:10 [INFO] Fallback to local cache"""

ACCESS_LOGS = """192.168.1.10 - - [15/Oct/2025:08:23:01 +0000] "GET /api/users HTTP/1.1" 200 1234
192.168.1.11 - - [15/Oct/2025:08:23:02 +0000] "POST /api/login HTTP/1.1" 200 567
10.0.0.5 - - [15/Oct/2025:08:24:00 +0000] "GET /api/products HTTP/1.1" 404 89
192.168.1.10 - - [15/Oct/2025:08:25:00 +0000] "DELETE /api/users/5 HTTP/1.1" 403 45"""

IP_PORT_TEXT = """Server alpha listening on 192.168.1.100:8080
Server beta listening on 10.0.0.1:3000
Proxy at 172.16.0.1:443 forwarding to 192.168.1.100:8080
Health check: 10.0.0.1:3000 OK, 172.16.0.1:443 OK"""

MALFORMED_LOGS = """2025-10-15 08:23:01 [INFO] Server started \u2014 ready to accept connections
2025-10-15 08:23:05 [] Empty level field
08:24:12 [WARN] Missing date prefix
2025-10-15 08:25:00 [ERROR] Stack trace:
  at Connection.connect (net.js:123)
  at Pool.acquire (pool.js:45)
2025-10-15 08:25:10 [INFO] Caf\u00e9 service resumed with \u00fcber-fast mode"""


# ── Tasks ────────────────────────────────────────────────────────────

TASKS = [
    # Seed tasks — use provided tools
    Task(
        id="seed_1",
        description=(
            "Read this CSV and convert to JSON:\n"
            "level,count\nINFO,4\nWARN,1\nERROR,2"
        ),
        task_type=TaskType.SEED,
        expected=[
            {"level": "INFO", "count": "4"},
            {"level": "WARN", "count": "1"},
            {"level": "ERROR", "count": "2"},
        ],
    ),
    Task(
        id="seed_2",
        description='Compute the SHA-256 hash of the string "text_processing".',
        task_type=TaskType.SEED,
        expected="40391d773134f49f30381f080cebfb370e88b5d3ac85028b12f56f5ec99c7455",
    ),
    Task(
        id="seed_3",
        description=(
            'Write this data as JSON:\n'
            '[{"pattern": "\\\\d+", "matches": 5}, {"pattern": "[a-z]+", "matches": 12}]'
        ),
        task_type=TaskType.SEED,
    ),

    # Gap tasks — require new tools
    Task(
        id="gap_1",
        description=(
            "Extract structured records from this unstructured text. Each email should become a dict "
            "with keys: from, to, subject, date, body. Return a list of extracted records.\n\n"
            f"{EMAIL_TEXT}"
        ),
        task_type=TaskType.GAP,
        expected=[
            {
                "from": "alice@example.com",
                "to": "bob@example.com",
                "subject": "Q3 Report",
                "date": "2025-10-15",
                "body": "Hi Bob,\nPlease find the Q3 report attached.\nBest, Alice",
            },
            {
                "from": "charlie@example.com",
                "to": "diana@example.com",
                "subject": "Meeting Tomorrow",
                "date": "2025-10-16",
                "body": "Diana,\nCan we reschedule to 3pm?\nThanks, Charlie",
            },
        ],
        hidden_tests=[
            {
                "input": {
                    "text": "From: x@y.com\nTo: a@b.com\nSubject: Test\nDate: 2025-01-01\n\nHello",
                    "format": "email",
                },
                "expected": [{"from": "x@y.com", "to": "a@b.com", "subject": "Test", "date": "2025-01-01", "body": "Hello"}],
            },
        ],
        adversarial_tests=[
            {"input": {"text": "", "format": "email"}},
            {"input": {"text": "Just plain text with no structure at all.", "format": "email"}},
        ],
    ),
    Task(
        id="gap_2",
        description=(
            "Extract all matches from text using a regex pattern with named groups. Return a list of "
            "dicts where keys are the group names and values are the matched strings.\n\n"
            "Pattern: `(?P<timestamp>\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) \\[(?P<level>\\w+)\\] (?P<message>.+)`\n\n"
            f"Text:\n{SERVER_LOGS}"
        ),
        task_type=TaskType.GAP,
        expected=[
            {"timestamp": "2025-10-15 08:23:01", "level": "INFO", "message": "Server started on port 8080"},
            {"timestamp": "2025-10-15 08:23:05", "level": "INFO", "message": "Connected to database"},
            {"timestamp": "2025-10-15 08:24:12", "level": "WARN", "message": "Slow query detected (1532ms)"},
            {"timestamp": "2025-10-15 08:25:00", "level": "ERROR", "message": "Connection timeout to redis:6379"},
            {"timestamp": "2025-10-15 08:25:01", "level": "INFO", "message": "Retrying connection..."},
            {"timestamp": "2025-10-15 08:25:03", "level": "ERROR", "message": "Connection failed after 3 retries"},
            {"timestamp": "2025-10-15 08:25:10", "level": "INFO", "message": "Fallback to local cache"},
        ],
        hidden_tests=[
            {
                "input": {
                    "pattern": r"(?P<word>\w+)",
                    "text": "hello world",
                },
                "expected": [{"word": "hello"}, {"word": "world"}],
            },
        ],
        adversarial_tests=[
            {"input": {"pattern": r"(?P<x>\d+)", "text": "no digits here"}},
            {"input": {"pattern": r"(no_named_group)", "text": "test"}},
            {"input": {"pattern": r"(?P<a>.*)*", "text": "x" * 100}},  # backtracking risk
        ],
    ),

    # Variant tasks — should REUSE tools from gap tasks
    Task(
        id="variant_1",
        description=(
            "Extract structured records from these Apache-style access logs. Each line should become "
            "a dict with keys: ip, timestamp, method, path, status, size.\n\n"
            f"{ACCESS_LOGS}"
        ),
        task_type=TaskType.VARIANT,
        reuses_task="gap_2",
        expected=[
            {"ip": "192.168.1.10", "timestamp": "15/Oct/2025:08:23:01 +0000", "method": "GET", "path": "/api/users", "status": "200", "size": "1234"},
            {"ip": "192.168.1.11", "timestamp": "15/Oct/2025:08:23:02 +0000", "method": "POST", "path": "/api/login", "status": "200", "size": "567"},
            {"ip": "10.0.0.5", "timestamp": "15/Oct/2025:08:24:00 +0000", "method": "GET", "path": "/api/products", "status": "404", "size": "89"},
            {"ip": "192.168.1.10", "timestamp": "15/Oct/2025:08:25:00 +0000", "method": "DELETE", "path": "/api/users/5", "status": "403", "size": "45"},
        ],
    ),
    Task(
        id="variant_2",
        description=(
            "Extract all IP:port pairs from this text using regex with named groups.\n\n"
            "Pattern: `(?P<ip>\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}):(?P<port>\\d+)`\n\n"
            f"Text:\n{IP_PORT_TEXT}"
        ),
        task_type=TaskType.VARIANT,
        reuses_task="gap_2",
        expected=[
            {"ip": "192.168.1.100", "port": "8080"},
            {"ip": "10.0.0.1", "port": "3000"},
            {"ip": "172.16.0.1", "port": "443"},
            {"ip": "192.168.1.100", "port": "8080"},
            {"ip": "10.0.0.1", "port": "3000"},
            {"ip": "172.16.0.1", "port": "443"},
        ],
    ),

    # Compose task — extract then aggregate
    Task(
        id="compose_1",
        description=(
            "Extract all log entries from the server logs using regex, then count the number of "
            "entries per log level. Return a JSON object mapping level -> count.\n\n"
            f"Text:\n{SERVER_LOGS}"
        ),
        task_type=TaskType.COMPOSE,
        composes_tasks=["gap_2", "seed_3"],
        expected={"INFO": 4, "WARN": 1, "ERROR": 2},
    ),

    # Regress task — re-test extraction
    Task(
        id="regress_1",
        description=(
            "Extract structured records from these log lines using regex with named groups:\n\n"
            "Pattern: `(?P<timestamp>\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) \\[(?P<level>\\w+)\\] (?P<message>.+)`\n\n"
            "Text:\n"
            "2025-11-01 10:00:00 [INFO] Startup complete\n"
            "2025-11-01 10:01:00 [WARN] Disk usage at 85%"
        ),
        task_type=TaskType.REGRESS,
        reuses_task="gap_2",
        expected=[
            {"timestamp": "2025-11-01 10:00:00", "level": "INFO", "message": "Startup complete"},
            {"timestamp": "2025-11-01 10:01:00", "level": "WARN", "message": "Disk usage at 85%"},
        ],
    ),

    # Adversarial tasks — break naive implementations
    Task(
        id="adversarial_1",
        description=(
            "Extract log entries from this text that contains unicode characters, empty level fields, "
            "missing date prefixes, and multiline messages. Handle gracefully — skip malformed lines "
            "or include partial data.\n\n"
            "Pattern: `(?P<timestamp>\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) \\[(?P<level>\\w+)\\] (?P<message>.+)`\n\n"
            f"Text:\n{MALFORMED_LOGS}"
        ),
        task_type=TaskType.ADVERSARIAL,
        breaks_task="gap_2",
        expected=[
            {"timestamp": "2025-10-15 08:23:01", "level": "INFO", "message": "Server started \u2014 ready to accept connections"},
            {"timestamp": "2025-10-15 08:25:00", "level": "ERROR", "message": "Stack trace:"},
            {"timestamp": "2025-10-15 08:25:10", "level": "INFO", "message": "Caf\u00e9 service resumed with \u00fcber-fast mode"},
        ],
    ),
    Task(
        id="adversarial_2",
        description=(
            "Extract matches using a regex pattern that has overlapping match potential. "
            "Use non-overlapping leftmost matches only.\n\n"
            "Pattern: `(?P<num>\\d{2,4})`\n\n"
            "Text: 'Order 12345 has 67 items weighing 8 kg, reference A99B'"
        ),
        task_type=TaskType.ADVERSARIAL,
        breaks_task="gap_2",
        expected=[
            {"num": "1234"},
            {"num": "67"},
            {"num": "99"},
        ],
    ),
]


def create_session() -> Session:
    return Session(
        id="data_transform_s3",
        name="Text Processing & Extraction",
        domain="data_transform",
        tasks=TASKS,
        seed_tools=SEED_TOOLS,
        description="Tests tool creation for structured extraction from unstructured text "
                    "and regex pattern matching, with composition, reuse, and adversarial edge cases.",
    )
