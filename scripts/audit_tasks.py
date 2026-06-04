#!/usr/bin/env python3
"""Audit all benchmark tasks for deterministic verification coverage.

Prints a per-session report and summary table. Exit code 0 if all tasks
have either `expected` or `verify`; exit code 1 otherwise.
"""
from __future__ import annotations
import importlib, sys, pathlib, ast

REPO_ROOT = pathlib.Path(__file__).parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

DOMAIN_SESSIONS = [
    ("data_transform", [1, 2, 3, 4, 5]),
    ("numerical", [1, 2, 3]),
    ("api_orchestration", [1]),
]


def audit_session(domain: str, session_num: int) -> tuple[int, int, int]:
    """Returns (total_tasks, has_verifier, unverified)."""
    mod_path = f"evolvetool_bench.domains.{domain}.session_{session_num}"
    try:
        mod = importlib.import_module(mod_path)
    except ImportError as e:
        print(f"  IMPORT ERROR: {mod_path}: {e}")
        return 0, 0, 0

    # Find SESSION or session variable (common patterns)
    session_obj = None
    for attr in ["SESSION", "session", f"session_{session_num}"]:
        if hasattr(mod, attr):
            session_obj = getattr(mod, attr)
            break

    if session_obj is None and hasattr(mod, "create_session"):
        session_obj = mod.create_session()

    if session_obj is None:
        print(f"  WARN: no SESSION in {mod_path}")
        return 0, 0, 0

    tasks = session_obj.tasks
    verified = sum(1 for t in tasks if t.expected is not None or t.verify is not None)
    unverified = len(tasks) - verified
    return len(tasks), verified, unverified


def main() -> int:
    total_tasks = 0
    total_verified = 0
    total_unverified = 0
    all_ok = True

    print("=" * 60)
    print(f"{'Session':<40} {'Total':>6} {'Verified':>9} {'Unverified':>11}")
    print("-" * 60)

    for domain, sessions in DOMAIN_SESSIONS:
        for s in sessions:
            label = f"{domain}/session_{s}"
            n, v, u = audit_session(domain, s)
            total_tasks += n
            total_verified += v
            total_unverified += u
            if u > 0:
                all_ok = False
            status = "OK" if u == 0 else f"WARN: {u} unverified"
            print(f"  {label:<38} {n:>6} {v:>9} {u:>11}  {status}")

    print("=" * 60)
    print(f"  {'TOTAL':<38} {total_tasks:>6} {total_verified:>9} {total_unverified:>11}")
    print()

    if all_ok:
        print(f"✓ {total_tasks}/{total_tasks} benchmark tasks deterministically verified")
        return 0
    else:
        print(f"✗ {total_unverified}/{total_tasks} tasks lack deterministic verification")
        print("  Tasks without expected or verify will be marked FAIL by the runner.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
