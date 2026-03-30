"""Human Oracle baseline — agent gets perfect hand-written tools for every gap task.

This measures the CEILING: how well would the agent perform with ideal tools?
The gap between Human Oracle and ARISE quantifies synthesis quality loss.
"""

from __future__ import annotations

import json
from typing import Any

from .no_evolution import NoEvolutionSystem


# Hand-written reference tools for each domain's gap tasks
HUMAN_TOOLS = [
    # ── Domain A: Data Transform ─────────────────────────────────
    {
        "name": "decode_abr",
        "description": "Decode ARISE Binary Record (ABR) format from base64.",
        "implementation": '''
def decode_abr(data: str) -> list:
    """Decode ABR format: base64 → binary records with field_count, name/value pairs, 0xFF separators."""
    import base64, struct
    raw = base64.b64decode(data)
    records = []
    i = 0
    while i < len(raw):
        if raw[i] == 0xFF:
            i += 1
            continue
        field_count = raw[i]; i += 1
        rec = {}
        for _ in range(field_count):
            name_len = raw[i]; i += 1
            name = raw[i:i+name_len].decode("utf-8"); i += name_len
            val_len = struct.unpack(">H", raw[i:i+2])[0]; i += 2
            val = raw[i:i+val_len].decode("utf-8", errors="replace"); i += val_len
            rec[name] = val
        records.append(rec)
    return records
''',
    },
    {
        "name": "decode_rle_matrix",
        "description": "Parse RLE matrix format: rows,cols;val:count,...",
        "implementation": '''
def decode_rle_matrix(rle_string: str) -> list:
    """Parse 'rows,cols;val:count,...' into list of lists."""
    header, data = rle_string.split(";", 1)
    rows, cols = map(int, header.split(","))
    flat = []
    if data.strip():
        for run in data.split(","):
            val, count = run.split(":")
            flat.extend([int(val)] * int(count))
    matrix = []
    for r in range(rows):
        matrix.append(flat[r*cols:(r+1)*cols])
    return matrix
''',
    },
    {
        "name": "hmac_timestamp_auth",
        "description": "Generate HMAC-timestamp auth header for the mock API.",
        "implementation": '''
def hmac_timestamp_auth(secret: str) -> str:
    """Generate X-Auth header: timestamp:hmac_sha256(secret, timestamp)."""
    import hmac, hashlib, time
    ts = str(int(time.time()))
    sig = hmac.new(secret.encode(), ts.encode(), hashlib.sha256).hexdigest()
    return ts + ":" + sig
''',
    },
    {
        "name": "fetch_all_pages",
        "description": "Fetch all pages from a cursor-paginated API endpoint.",
        "implementation": '''
def fetch_all_pages(url: str, auth_header: str) -> list:
    """Follow cursor pagination until has_more=false. Returns all items."""
    import urllib.request, json
    all_items = []
    cursor = None
    while True:
        full_url = url + (f"?cursor={cursor}" if cursor else "")
        req = urllib.request.Request(full_url, headers={"X-Auth": auth_header})
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
        all_items.extend(resp.get("data", []))
        pag = resp.get("pagination", {})
        if not pag.get("has_more"):
            break
        cursor = pag.get("next_cursor")
    return all_items
''',
    },
    {
        "name": "parse_vdl_schema",
        "description": "Parse VDL (Validation Definition Language) schema format.",
        "implementation": '''
def parse_vdl_schema(schema_text: str) -> dict:
    """Parse @schema Name\\nfield_name:type[flags] into structured dict."""
    import re
    lines = [l.strip() for l in schema_text.strip().split("\\n") if l.strip()]
    result = {"name": "", "fields": []}
    for line in lines:
        if line.startswith("@schema"):
            result["name"] = line.split(None, 1)[1] if " " in line else ""
        else:
            m = re.match(r"(>?\\*?)(\\w+):(\\w+)(.*)", line)
            if m:
                prefix, name, typ, flags_str = m.groups()
                field = {"name": name, "type": typ, "required": "[R]" in flags_str,
                         "nested": prefix.startswith(">"), "array": "*" in prefix}
                result["fields"].append(field)
    return result
''',
    },
]


class HumanOracleSystem(NoEvolutionSystem):
    """Agent with seed tools PLUS hand-written reference tools for every gap task."""

    def setup(self, seed_tools: list[dict]) -> None:
        # Add human tools on top of seed tools
        all_tools = seed_tools + HUMAN_TOOLS
        super().setup(all_tools)
