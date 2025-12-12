#!/usr/bin/env bash
#
# Run simple_infer_main.py on the KITTI dataset for two baseline models:
#   1) PointPillars (official KITTI 3-class checkpoint)
#   2) SECOND (serves as the “SSN” substitute since no KITTI SSN weights exist)
#
# Usage:
#   ./objectdetection/scripts/run_kitti_inference.sh
#
# Environment overrides:
#   KITTI_DATA_ROOT   : path to KITTI detection data (default: /home/manav/workspaces/dataset)
#   KITTI_ANN_FILE    : path to kitti_infos_val.pkl (default: /home/manav/workspaces/v1.0-mini/kitti_infos_val.pkl)
#   KITTI_CHECKPOINT_DIR : directory with .pth checkpoints (default: /home/manav/workspaces/checkpoints)
#   KITTI_CONFIG_ROOT : mmdetection3d configs folder (default: /home/manav/workspaces/mmdetection3d/configs)
#   PYTHON_BIN        : python interpreter (default: objectdetection/.venv/bin/python)
#
# NOTE: Make sure KITTI_DATA_ROOT contains the converted KITTI detection dataset
# (e.g., with training/velodyne, ImageSets, etc.) and that KITTI_ANN_FILE points
# to the matching info pickle.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MAIN="${REPO_ROOT}/simple_infer_main.py"

DATA_ROOT="${KITTI_DATA_ROOT:-/home/manav/workspaces/dataset}"
ANN_FILE="${KITTI_ANN_FILE:-/home/manav/workspaces/v1.0-mini/kitti_infos_val.pkl}"
CKPT_DIR="${KITTI_CHECKPOINT_DIR:-/home/manav/workspaces/checkpoints}"
CONFIG_ROOT="${KITTI_CONFIG_ROOT:-/home/manav/workspaces/mmdetection3d/configs}"
OUT_ROOT="${REPO_ROOT}/infer_runs/simple_infer_results"

mkdir -p "${OUT_ROOT}"

declare -A CONFIGS
declare -A CHECKPOINTS
declare -A OUTPUTS

CONFIGS["pointpillars"]="${CONFIG_ROOT}/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py"
CHECKPOINTS["pointpillars"]="${CKPT_DIR}/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
OUTPUTS["pointpillars"]="${OUT_ROOT}/kitti_pointpillars_manual"

# SSN does not ship with KITTI configs/weights, so we use SECOND as a substitute baseline.
CONFIGS["second"]="${CONFIG_ROOT}/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py"
CHECKPOINTS["second"]="${CKPT_DIR}/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth"
OUTPUTS["second"]="${OUT_ROOT}/kitti_second_manual"

check_requirements() {
    local ckpt="$1"
    if [[ ! -f "${ckpt}" ]]; then
        echo "[error] Missing checkpoint: ${ckpt}"
        echo "        Download it from the MMDetection3D model zoo and place it here."
        exit 1
    fi
    if [[ ! -f "${ANN_FILE}" ]]; then
        echo "[error] KITTI ann_file not found: ${ANN_FILE}"
        exit 1
    fi
    if [[ ! -d "${DATA_ROOT}" ]]; then
        echo "[error] KITTI data_root not found: ${DATA_ROOT}"
        exit 1
    fi
}

run_model() {
    local name="$1"
    local config="${CONFIGS[$name]}"
    local ckpt="${CHECKPOINTS[$name]}"
    local out_dir="${OUTPUTS[$name]}"

    check_requirements "${ckpt}"
    echo "==> Running ${name} (output -> ${out_dir})"

    "${PYTHON_BIN}" "${MAIN}" \
        --config "${config}" \
        --checkpoint "${ckpt}" \
        --dataroot "${DATA_ROOT}" \
        --ann-file "${ANN_FILE}" \
        --out-dir "${out_dir}" \
        --dataset kitti \
        --data-source cfg \
        --device cpu \
        --eval \
        --eval-backend manual \
        --max-samples -1
}

run_model "pointpillars"
run_model "second"
