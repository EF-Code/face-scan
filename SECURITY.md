# Security Notes (CIA Triad)

This project is a local OpenCV-based face detector. "CIA-level" in this repo refers to the CIA triad:
Confidentiality, Integrity, and Availability.

## Confidentiality (Privacy)

- Use `--privacy blur|pixelate|black` to redact faces in display/recording/snapshots.
- Snapshots and outputs attempt to use restrictive permissions on POSIX systems:
  - directories: `0700`
  - files: `0600`
- Avoid sharing audit logs and snapshots if they contain sensitive context (timestamps, filenames, run metadata).

## Integrity (Tamper Evidence)

- Cascade integrity check: `--cascade-sha256 ...` (or the built-in check for the default cascade) verifies the XML file hash.
- Audit log: `--audit-log audit.jsonl` writes append-only JSONL entries with a hash chain.
- Verify audit logs with `python verify_audit.py audit.jsonl`.

## Availability (Resilience)

- `detectCapture.py` supports `--reconnect-attempts` / `--reconnect-delay` for transient camera failures.
- Use `--timeout` for bounded runtime in automation.

## Threat Model (What This Does Not Do)

- No network transmission is implemented by default.
- This is not an authentication system and does not provide identity verification.
- This repo does not attempt to bypass OS security controls or silently record without user intent.

