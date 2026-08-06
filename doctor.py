"""Check whether the local runtime can execute face-scan workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys

from face_scan.observability import DEFAULT_CASCADE_SHA256, sha256_file
from face_scan.runtime import inspect_opencv


DEFAULT_CASCADE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "haarcascade_frontalface_default.xml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check face-scan runtime dependencies and assets.")
    parser.add_argument("--cascade", default=DEFAULT_CASCADE_PATH, help="Cascade XML file to validate.")
    parser.add_argument("--json", action="store_true", help="Emit a machine-readable JSON report.")
    return parser.parse_args()


def build_report(cascade_path: str) -> dict:
    runtime = inspect_opencv()
    cascade_exists = os.path.isfile(cascade_path)
    cascade_sha256 = sha256_file(cascade_path) if cascade_exists else None
    bundled_cascade = os.path.abspath(cascade_path) == DEFAULT_CASCADE_PATH
    cascade_valid = cascade_exists and (
        not bundled_cascade or cascade_sha256 == DEFAULT_CASCADE_SHA256
    )
    return {
        "ok": runtime.ok and cascade_valid,
        "opencv": runtime.to_dict(),
        "cascade": {
            "path": os.path.abspath(cascade_path),
            "exists": cascade_exists,
            "sha256": cascade_sha256,
            "valid": cascade_valid,
        },
    }


def main() -> int:
    args = parse_args()
    report = build_report(args.cascade)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "OK" if report["ok"] else "FAILED"
        opencv = report["opencv"]
        cascade = report["cascade"]
        print(f"face-scan doctor: {status}")
        print(f"OpenCV module: {opencv['module_version']} ({opencv['module_path']})")
        if opencv["distributions"]:
            packages = ", ".join(
                f"{name}=={version}" for name, version in opencv["distributions"].items()
            )
            print(f"OpenCV distributions: {packages}")
        if opencv["missing_apis"]:
            print(f"Missing OpenCV APIs: {', '.join(opencv['missing_apis'])}")
        print(f"Cascade: {'valid' if cascade['valid'] else 'invalid'} ({cascade['path']})")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
