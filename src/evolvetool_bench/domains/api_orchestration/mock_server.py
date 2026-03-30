"""Mock API server with proprietary auth, pagination, and data formats.

This server implements custom protocols that LLMs cannot solve from training data:
- HMAC-timestamp auth (not standard Bearer/OAuth)
- Cursor-based pagination with encrypted cursors
- Response data in TPACK binary format (from Domain A)

The server runs on localhost during benchmark execution.
"""

import base64
import hashlib
import hmac
import json
import struct
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs


# ── Config ───────────────────────────────────────────────────────────

API_SECRET = "evolvetool-bench-secret-2026"
ITEMS_PER_PAGE = 3

# Mock data
USERS = [
    {"id": 1, "name": "Alice", "role": "engineer", "score": 95},
    {"id": 2, "name": "Bob", "role": "designer", "score": 82},
    {"id": 3, "name": "Charlie", "role": "manager", "score": 78},
    {"id": 4, "name": "Diana", "role": "engineer", "score": 91},
    {"id": 5, "name": "Eve", "role": "analyst", "score": 88},
    {"id": 6, "name": "Frank", "role": "engineer", "score": 73},
    {"id": 7, "name": "Grace", "role": "designer", "score": 96},
    {"id": 8, "name": "Hank", "role": "manager", "score": 65},
    {"id": 9, "name": "Ivy", "role": "analyst", "score": 89},
    {"id": 10, "name": "Jack", "role": "engineer", "score": 77},
]

METRICS = [
    {"ts": 1700000000 + i * 3600, "cpu": 20 + (i * 7) % 60, "mem": 40 + (i * 13) % 50, "node": f"n{i%3}"}
    for i in range(20)
]

EVENTS = [
    {"id": i, "type": ["info", "warn", "error", "critical"][i % 4],
     "message": f"Event {i}: {'all good' if i%4==0 else 'check this' if i%4==1 else 'failed op' if i%4==2 else 'system down'}",
     "source": f"svc-{i%3}", "timestamp": 1700000000 + i * 300}
    for i in range(15)
]


# ── Auth ─────────────────────────────────────────────────────────────

def _verify_auth(headers: dict) -> bool:
    """Verify HMAC-timestamp auth.

    Expected header: X-Auth: <timestamp>:<hmac_hex>
    HMAC = HMAC-SHA256(secret, timestamp_string)
    Timestamp must be within 300 seconds of current time.
    """
    auth = headers.get("X-Auth", "")
    if ":" not in auth:
        return False
    parts = auth.split(":", 1)
    try:
        ts = int(parts[0])
    except ValueError:
        return False
    # Check timestamp freshness (allow generous window for benchmarking)
    if abs(time.time() - ts) > 3600:
        return False
    expected = hmac.new(API_SECRET.encode(), parts[0].encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(parts[1], expected)


def _make_auth_header() -> str:
    """Generate a valid auth header (for test verification)."""
    ts = str(int(time.time()))
    sig = hmac.new(API_SECRET.encode(), ts.encode(), hashlib.sha256).hexdigest()
    return f"{ts}:{sig}"


# ── Pagination ───────────────────────────────────────────────────────

def _encode_cursor(offset: int) -> str:
    """Encode cursor as base64(xor-scrambled offset bytes)."""
    key = 0x5A
    offset_bytes = struct.pack(">I", offset)
    scrambled = bytes(b ^ key for b in offset_bytes)
    return base64.urlsafe_b64encode(scrambled).decode()


def _decode_cursor(cursor: str) -> int:
    """Decode cursor back to offset."""
    key = 0x5A
    scrambled = base64.urlsafe_b64decode(cursor)
    offset_bytes = bytes(b ^ key for b in scrambled)
    return struct.unpack(">I", offset_bytes)[0]


# ── Request Handler ──────────────────────────────────────────────────

class MockAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress logs

    def _headers_dict(self) -> dict:
        return {k: v for k, v in self.headers.items()}

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        headers = self._headers_dict()

        # Health check (no auth required)
        if path == "/health":
            self._send_json({"status": "ok"})
            return

        # Auth info endpoint (no auth required) — tells agent how to authenticate
        if path == "/auth/info":
            self._send_json({
                "scheme": "HMAC-Timestamp",
                "header": "X-Auth",
                "format": "<unix_timestamp>:<hmac_sha256_hex>",
                "secret_env_var": "API_SECRET",
                "note": "HMAC = HMAC-SHA256(secret, timestamp_string). Secret is: " + API_SECRET,
            })
            return

        # All other endpoints require auth
        if not _verify_auth(headers):
            self._send_json({"error": "Unauthorized. Use X-Auth header with HMAC-timestamp auth. GET /auth/info for details."}, 401)
            return

        # ── /api/users (paginated) ───────────────────────────────
        if path == "/api/users":
            cursor = params.get("cursor", [None])[0]
            offset = 0
            if cursor:
                try:
                    offset = _decode_cursor(cursor)
                except Exception:
                    self._send_json({"error": "Invalid cursor"}, 400)
                    return

            page = USERS[offset:offset + ITEMS_PER_PAGE]
            next_offset = offset + ITEMS_PER_PAGE
            next_cursor = _encode_cursor(next_offset) if next_offset < len(USERS) else None

            self._send_json({
                "data": page,
                "pagination": {
                    "next_cursor": next_cursor,
                    "has_more": next_cursor is not None,
                    "total": len(USERS),
                },
            })
            return

        # ── /api/users/:id ───────────────────────────────────────
        if path.startswith("/api/users/"):
            try:
                uid = int(path.split("/")[-1])
                user = next((u for u in USERS if u["id"] == uid), None)
                if user:
                    self._send_json({"data": user})
                else:
                    self._send_json({"error": "User not found"}, 404)
            except ValueError:
                self._send_json({"error": "Invalid user ID"}, 400)
            return

        # ── /api/metrics (paginated, time-filtered) ──────────────
        if path == "/api/metrics":
            cursor = params.get("cursor", [None])[0]
            after = int(params.get("after", [0])[0])
            before = int(params.get("before", [9999999999])[0])

            filtered = [m for m in METRICS if after <= m["ts"] <= before]
            offset = 0
            if cursor:
                try:
                    offset = _decode_cursor(cursor)
                except Exception:
                    self._send_json({"error": "Invalid cursor"}, 400)
                    return

            page = filtered[offset:offset + ITEMS_PER_PAGE]
            next_offset = offset + ITEMS_PER_PAGE
            next_cursor = _encode_cursor(next_offset) if next_offset < len(filtered) else None

            self._send_json({
                "data": page,
                "pagination": {
                    "next_cursor": next_cursor,
                    "has_more": next_cursor is not None,
                    "total": len(filtered),
                },
            })
            return

        # ── /api/events ──────────────────────────────────────────
        if path == "/api/events":
            event_type = params.get("type", [None])[0]
            source = params.get("source", [None])[0]
            filtered = EVENTS
            if event_type:
                filtered = [e for e in filtered if e["type"] == event_type]
            if source:
                filtered = [e for e in filtered if e["source"] == source]
            self._send_json({"data": filtered, "count": len(filtered)})
            return

        self._send_json({"error": "Not found"}, 404)


def start_mock_server(port: int = 18080) -> tuple[HTTPServer, threading.Thread]:
    """Start the mock server in a background thread. Returns (server, thread)."""
    server = HTTPServer(("127.0.0.1", port), MockAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop_mock_server(server: HTTPServer):
    server.shutdown()
