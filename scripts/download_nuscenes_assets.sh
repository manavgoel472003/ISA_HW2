#!/usr/bin/env bash
set -euo pipefail

#
# Helper script to download the NuScenes v1.0-mini dataset and the checkpoints
# for the PointPillars and CenterPoint models used by simple_infer_main.py.
# Usage:
#   NUSCENES_DATA_ROOT=/path/to/v1.0-mini \
#   NUSCENES_CKPT_DIR=/path/to/checkpoints \
#   ./objectdetection/scripts/download_nuscenes_assets.sh
#
# Both environment variables are optional. Defaults match this workspace.
#

DATA_ROOT=${NUSCENES_DATA_ROOT:-/home/manav/workspaces/v1.0-mini}
CKPT_DIR=${NUSCENES_CKPT_DIR:-/home/manav/workspaces/checkpoints}
ARCHIVE_PATH="${DATA_ROOT%/}.tgz"
NUSC_DATA_URL="https://www.nuscenes.org/data/v1.0-mini.tgz"

POINTPILLARS_URL="https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
POINTPILLARS_CKPT="${CKPT_DIR}/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"

CENTERPOINT_URL="https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth"
CENTERPOINT_CKPT="${CKPT_DIR}/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth"

download_file() {
    local url="$1"
    local dest="$2"
    local label="$3"

    if [[ -f "${dest}" ]]; then
        echo "[skip] ${label} already exists at ${dest}"
        return
    fi

    echo "[download] Fetching ${label} ..."
    curl -L --progress-bar "${url}" -o "${dest}"
    echo "[done] Saved ${label} to ${dest}"
}

prepare_dataset() {
    if [[ -d "${DATA_ROOT}/samples" && -d "${DATA_ROOT}/sweeps" ]]; then
        echo "[skip] NuScenes v1.0-mini already prepared at ${DATA_ROOT}"
        return
    fi

    local parent_dir
    parent_dir=$(dirname "${DATA_ROOT}")
    mkdir -p "${parent_dir}"

    download_file "${NUSC_DATA_URL}" "${ARCHIVE_PATH}" "NuScenes v1.0-mini archive"

    echo "[extract] Unpacking NuScenes v1.0-mini to ${parent_dir}"
    tar -xzf "${ARCHIVE_PATH}" -C "${parent_dir}"
    echo "[done] Dataset extracted to ${DATA_ROOT}"
}

prepare_checkpoints() {
    mkdir -p "${CKPT_DIR}"
    download_file "${POINTPILLARS_URL}" "${POINTPILLARS_CKPT}" "PointPillars checkpoint"
    download_file "${CENTERPOINT_URL}" "${CENTERPOINT_CKPT}" "CenterPoint checkpoint"
}

prepare_dataset
prepare_checkpoints

echo "[complete] Assets ready. Dataset root: ${DATA_ROOT}, checkpoints: ${CKPT_DIR}"
