#!/usr/bin/env bash
#
# Render Open3D videos for every experiment under infer_runs/simple_infer_results.
# Uses create_open3d_videos.py to load *_points.ply (plus *_pred/_gt) and export MP4 clips.
#
# Usage:
#   objectdetection/scripts/render_open3d_videos.sh
#   OPEN3D_RUNS_ROOT=/path/to/runs objectdetection/scripts/render_open3d_videos.sh
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
SCRIPT="${REPO_ROOT}/create_open3d_videos.py"
RUNS_ROOT="${OPEN3D_RUNS_ROOT:-${REPO_ROOT}/infer_runs/simple_infer_results}"

if [[ ! -d "${RUNS_ROOT}" ]]; then
    echo "[error] Runs root not found: ${RUNS_ROOT}" >&2
    exit 1
fi

echo "Rendering Open3D videos from ${RUNS_ROOT}"
"${PYTHON_BIN}" "${SCRIPT}" \
    --runs-root "${RUNS_ROOT}" \
    --png-suffix "" \
    --output-pattern "{exp}_open3d.mp4"
