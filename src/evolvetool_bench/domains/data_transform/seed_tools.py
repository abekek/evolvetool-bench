"""Shared seed tools for all Data Transformation sessions.

These three tools are provided to the agent at the start of every session.
"""

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
