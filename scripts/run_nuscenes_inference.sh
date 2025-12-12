#!/usr/bin/env bash
#
# Run simple_infer_main.py for NuScenes v1.0-mini using the PointPillars
# and CenterPoint models. Produces PNGs, PLYs, and NuScenes eval JSON.
#
# Usage:
#   chmod +x objectdetection/scripts/run_nuscenes_inference.sh
#   objectdetection/scripts/run_nuscenes_inference.sh
#   # (set env vars below to override defaults)
#
# Environment overrides:
#   NUSCENES_DATA_ROOT=/path/to/v1.0-mini
#   CHECKPOINT_DIR=/path/to/checkpoints
#   MMDET3D_CONFIG_ROOT=/path/to/mmdetection3d/configs
#   PYTHON_BIN=/path/to/venv/bin/python

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MAIN="${REPO_ROOT}/simple_infer_main.py"
NUSCENES_DATA_ROOT="${NUSCENES_DATA_ROOT:-/home/manav/workspaces/v1.0-mini}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/manav/workspaces/checkpoints}"
CONFIG_ROOT="${MMDET3D_CONFIG_ROOT:-/home/manav/workspaces/mmdetection3d/configs}"
OUT_ROOT="${REPO_ROOT}/infer_runs/simple_infer_results"

mkdir -p "${OUT_ROOT}"

declare -A CONFIGS
declare -A CHECKPOINTS

CONFIGS["pointpillars"]="${CONFIG_ROOT}/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py"
CHECKPOINTS["pointpillars"]="${CHECKPOINT_DIR}/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"

CONFIGS["centerpoint"]="${CONFIG_ROOT}/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
CHECKPOINTS["centerpoint"]="${CHECKPOINT_DIR}/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth"

run_model() {
    local name="$1"
    local config="${CONFIGS[$name]}"
    local ckpt="${CHECKPOINTS[$name]}"
    local out_dir="${OUT_ROOT}/nuscenes_${name}_v1.0-mini_manual"

    echo "==> Running ${name} (output -> ${out_dir})"
    "${PYTHON_BIN}" "${MAIN}" \
        --config "${config}" \
        --checkpoint "${ckpt}" \
        --dataroot "${NUSCENES_DATA_ROOT}" \
        --out-dir "${out_dir}" \
        --nus-version v1.0-mini \
        --data-source custom \
        --device cpu \
        --eval \
        --eval-backend manual
}

run_model "pointpillars"
run_model "centerpoint"
