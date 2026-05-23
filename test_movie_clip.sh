#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /absolute/path/to/clip.mp4 [output-dir]" >&2
  exit 1
fi

CLIP_PATH="$1"
OUTPUT_DIR="${2:-/tmp/face-scan-movie-test}"
PYTHON_BIN="/home/wellington/env/bin/python"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIP_NAME="$(basename "${CLIP_PATH}")"
CLIP_STEM="${CLIP_NAME%.*}"
RUN_DIR="${OUTPUT_DIR}/${CLIP_STEM}"

mkdir -p "${RUN_DIR}"

ANNOTATED_VIDEO="${RUN_DIR}/annotated.mp4"
SUMMARY_JSON="${RUN_DIR}/summary.json"
SNAPSHOT_DIR="${RUN_DIR}/snapshots"
AUDIT_LOG="${RUN_DIR}/audit.jsonl"

echo "Input: ${CLIP_PATH}"
echo "Output directory: ${RUN_DIR}"
echo "Press q or Esc in the playback window to stop early."

"${PYTHON_BIN}" "${SCRIPT_DIR}/detect_video.py" "${CLIP_PATH}" \
  --output "${ANNOTATED_VIDEO}" \
  --summary-json "${SUMMARY_JSON}" \
  --snapshot-dir "${SNAPSHOT_DIR}" \
  --audit-log "${AUDIT_LOG}" \
  --show-metrics \
  --draw-labels

echo
echo "Done."
echo "Annotated video: ${ANNOTATED_VIDEO}"
echo "Summary JSON: ${SUMMARY_JSON}"
echo "Snapshots: ${SNAPSHOT_DIR}"
echo "Audit log: ${AUDIT_LOG}"
