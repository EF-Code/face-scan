"""Verify the tamper-evident hash chain in an audit JSONL file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a face-scan audit log hash chain.")
    parser.add_argument("path", help="Path to the audit JSONL file.")
    return parser.parse_args()


def canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def verify(path: str) -> int:
    prev_hash = ""
    line_no = 0
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line_no += 1
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"Invalid JSON at line {line_no}", file=sys.stderr)
                    return 1

                if not isinstance(entry, dict):
                    print(f"Non-object entry at line {line_no}", file=sys.stderr)
                    return 1

                expected_prev = entry.get("prev_hash", "")
                entry_hash = entry.get("hash", "")
                if not isinstance(expected_prev, str) or not isinstance(entry_hash, str):
                    print(f"Missing hash fields at line {line_no}", file=sys.stderr)
                    return 1

                if expected_prev != prev_hash:
                    print(
                        f"prev_hash mismatch at line {line_no}: expected {prev_hash} got {expected_prev}",
                        file=sys.stderr,
                    )
                    return 1

                payload = dict(entry)
                payload.pop("hash", None)
                canonical = canonical_json(payload)
                calc = hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()
                if calc != entry_hash:
                    print(
                        f"hash mismatch at line {line_no}: expected {calc} got {entry_hash}",
                        file=sys.stderr,
                    )
                    return 1

                prev_hash = entry_hash
    except OSError as exc:
        print(f"Unable to read {path}: {exc}", file=sys.stderr)
        return 1

    print("OK")
    return 0


def main() -> int:
    args = parse_args()
    return verify(args.path)


if __name__ == "__main__":
    raise SystemExit(main())

