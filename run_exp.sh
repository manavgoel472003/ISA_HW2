#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# USER TUNABLES (via env vars)
# -----------------------------
MAX_SAMPLES="${MAX_SAMPLES:-20}"   # visualization sample count
WORKERS="${WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"

# Where outputs go
RUNS_ROOT="${RUNS_ROOT:-$HOME/3d_det/infer_runs}"

# Data roots (NO symlinks needed)
KITTI_ROOT="${KITTI_ROOT:-$HOME/3d_det/mmdetection3d/data/kitti}"
KITTI_ANN="${KITTI_ANN:-$HOME/3d_det/mmdetection3d/data/kitti/kitti_infos_val.pkl}"

NUS_ROOT="${NUS_ROOT:-$HOME/3d_det/mmdetection3d/data/nuscenes}"
NUS_ANN="${NUS_ANN:-$HOME/3d_det/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl}"
NUS_VERSION="${NUS_VERSION:-v1.0-mini}"   # IMPORTANT for mini eval

# Model zoo dir (checkpoints + mim-dumped configs)
MODELZOO="${MODELZOO:-$HOME/3d_det/modelzoo_mmdetection3d}"

# Repo root (assumes you run this from mmdetection3d root)
REPO_ROOT="$(pwd)"

PYTHONUNBUFFERED=1

mkdir -p "$RUNS_ROOT"

# -----------------------------
# Patch: nuScenes mini eval_set
# -----------------------------
# Manual eval is called with eval_set="val" in simple_infer_main.py.
# For v1.0-mini it MUST be "mini_val". We patch it once, idempotently.
python - <<'PY'
from pathlib import Path
p = Path("simple_infer_main.py")
txt = p.read_text()

needle = 'eval_set="val",                    # You can change this if needed'
if needle in txt and "mini_val" not in txt:
    repl = 'eval_set=("mini_val" if args.nus_version=="v1.0-mini" else "val"),  # auto: mini->mini_val'
    txt = txt.replace(needle, repl)
    p.write_text(txt)
    print("[Patch] Updated simple_infer_main.py: val -> (mini_val if v1.0-mini else val)")
else:
    print("[Patch] No change needed (already patched or pattern not found).")
PY

# -----------------------------
# Helpers
# -----------------------------
exists_or_skip () {
  local f="$1"
  local what="$2"
  if [[ ! -f "$f" ]]; then
    echo "[SKIP] Missing $what: $f"
    return 1
  fi
  return 0
}

run_eval_kitti () {
  local name="$1"
  local cfg="$2"
  local ckpt="$3"
  local out="$RUNS_ROOT/${name}_eval"
  echo
  echo "============================================================"
  echo "EVAL RUN: $name"
  echo "OUT     : $out"
  echo "============================================================"
  mkdir -p "$out"

  python "$REPO_ROOT/simple_infer_main.py" \
    --config "$cfg" \
    --checkpoint "$ckpt" \
    --dataroot "$KITTI_ROOT" \
    --ann-file "$KITTI_ANN" \
    --out-dir "$out" \
    --dataset kitti \
    --data-source cfg \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --max-samples -1 \
    --eval \
    --eval-backend runner
}

run_vis_kitti () {
  local name="$1"
  local cfg="$2"
  local ckpt="$3"
  local out="$RUNS_ROOT/${name}"
  echo
  echo "============================================================"
  echo "VIS RUN: $name"
  echo "OUT    : $out"
  echo "============================================================"
  mkdir -p "$out"

  # KITTI image overlays can be flaky depending on your kitti folder layout,
  # so we default to no 2D images (still writes metrics.json, and PLY if enabled).
  python "$REPO_ROOT/simple_infer_main.py" \
    --config "$cfg" \
    --checkpoint "$ckpt" \
    --dataroot "$KITTI_ROOT" \
    --ann-file "$KITTI_ANN" \
    --out-dir "$out" \
    --dataset kitti \
    --data-source cfg \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --max-samples "$MAX_SAMPLES" \
    --no-open3d \
    --no-save-images
}

run_eval_nus_mini () {
  local name="$1"
  local cfg="$2"
  local ckpt="$3"
  local out="$RUNS_ROOT/${name}_eval"
  echo
  echo "============================================================"
  echo "EVAL RUN: $name (nuScenes mini)"
  echo "OUT     : $out"
  echo "============================================================"
  mkdir -p "$out"

  # manual backend => exports nuscenes_results.json + runs NuScenesEval + writes benchmark_perf.json
  python "$REPO_ROOT/simple_infer_main.py" \
    --config "$cfg" \
    --checkpoint "$ckpt" \
    --dataroot "$NUS_ROOT" \
    --ann-file "$NUS_ANN" \
    --nus-version "$NUS_VERSION" \
    --out-dir "$out" \
    --dataset nuscenes \
    --data-source cfg \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --max-samples -1 \
    --eval \
    --eval-backend manual
}

run_vis_nus () {
  local name="$1"
  local cfg="$2"
  local ckpt="$3"
  local out="$RUNS_ROOT/${name}"
  echo
  echo "============================================================"
  echo "VIS RUN: $name (nuScenes)"
  echo "OUT    : $out"
  echo "============================================================"
  mkdir -p "$out"

  python "$REPO_ROOT/simple_infer_main.py" \
    --config "$cfg" \
    --checkpoint "$ckpt" \
    --dataroot "$NUS_ROOT" \
    --ann-file "$NUS_ANN" \
    --nus-version "$NUS_VERSION" \
    --out-dir "$out" \
    --dataset nuscenes \
    --data-source custom \
    --nus-cams all \
    --crop-policy center \
    --device "$DEVICE" \
    --workers "$WORKERS" \
    --max-samples "$MAX_SAMPLES" \
    --no-open3d
}

# -----------------------------
# Model Paths
# -----------------------------
# KITTI (configs are in repo)
KITTI_PP_CFG="$REPO_ROOT/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py"
KITTI_PP_CKPT="$MODELZOO/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth"

KITTI_SECOND_CFG="$REPO_ROOT/configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-car.py"
KITTI_SECOND_CKPT="$MODELZOO/second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth"

# nuScenes (configs are dumped into MODELZOO by mim download)
NUS_PP_CFG="$MODELZOO/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py"
NUS_PP_CKPT="$MODELZOO/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth"

NUS_SSN_CFG="$MODELZOO/hv_ssn_secfpn_sbn-all_16xb2-2x_nus-3d.py"
NUS_SSN_CKPT="$MODELZOO/hv_ssn_secfpn_sbn-all_2x16_2x_nus-3d_20210830_101351-51915986.pth"

# Optional extra: CenterPoint (pillar) for nuScenes
NUS_CP_PILLAR_CFG="$MODELZOO/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
NUS_CP_PILLAR_CKPT="$MODELZOO/centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth"

# -----------------------------
# RUNS (KITTI first, then nuScenes)
# -----------------------------
echo "[Info] Using KITTI dataroot    : $KITTI_ROOT"
echo "[Info] Using KITTI ann_file    : $KITTI_ANN"
echo "[Info] Using NuScenes dataroot : $NUS_ROOT"
echo "[Info] Using NuScenes ann_file : $NUS_ANN"
echo "[Info] Using NuScenes version  : $NUS_VERSION"
echo "[Info] Outputs under           : $RUNS_ROOT"

# KITTI: SECOND
if exists_or_skip "$KITTI_SECOND_CFG" "KITTI SECOND config" && exists_or_skip "$KITTI_SECOND_CKPT" "KITTI SECOND checkpoint"; then
  run_vis_kitti  "kitti_second" "$KITTI_SECOND_CFG" "$KITTI_SECOND_CKPT"
  run_eval_kitti "kitti_second" "$KITTI_SECOND_CFG" "$KITTI_SECOND_CKPT"
fi

# KITTI: PointPillars
if exists_or_skip "$KITTI_PP_CFG" "KITTI PointPillars config" && exists_or_skip "$KITTI_PP_CKPT" "KITTI PointPillars checkpoint"; then
  run_vis_kitti  "kitti_pointpillars" "$KITTI_PP_CFG" "$KITTI_PP_CKPT"
  run_eval_kitti "kitti_pointpillars" "$KITTI_PP_CFG" "$KITTI_PP_CKPT"
fi

# nuScenes: PointPillars
if exists_or_skip "$NUS_PP_CFG" "nuScenes PointPillars config" && exists_or_skip "$NUS_PP_CKPT" "nuScenes PointPillars checkpoint"; then
  run_vis_nus       "nus_pointpillars" "$NUS_PP_CFG" "$NUS_PP_CKPT"
  run_eval_nus_mini "nus_pointpillars" "$NUS_PP_CFG" "$NUS_PP_CKPT"
fi

# nuScenes: SSN
if exists_or_skip "$NUS_SSN_CFG" "nuScenes SSN config" && exists_or_skip "$NUS_SSN_CKPT" "nuScenes SSN checkpoint"; then
  run_vis_nus       "nus_ssn" "$NUS_SSN_CFG" "$NUS_SSN_CKPT"
  run_eval_nus_mini "nus_ssn" "$NUS_SSN_CFG" "$NUS_SSN_CKPT"
fi

# nuScenes: CenterPoint (pillar) OPTIONAL
if [[ -f "$NUS_CP_PILLAR_CFG" && -f "$NUS_CP_PILLAR_CKPT" ]]; then
  run_vis_nus       "nus_centerpoint_pillar" "$NUS_CP_PILLAR_CFG" "$NUS_CP_PILLAR_CKPT"
  run_eval_nus_mini "nus_centerpoint_pillar" "$NUS_CP_PILLAR_CFG" "$NUS_CP_PILLAR_CKPT"
else
  echo
  echo "[WARN] CenterPoint pillar not found yet."
  echo "       Download with:"
  echo "       mim download mmdet3d --config centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d --dest $MODELZOO"
fi

# -----------------------------
# Summary CSV for comparison
# -----------------------------
python - <<'PY'
import json, glob
from pathlib import Path

runs_root = Path.home() / "3d_det" / "infer_runs"
out_csv = runs_root / "summary_metrics.csv"

rows = []
# Runner eval outputs:
for p in runs_root.glob("*_eval/benchmark_results.json"):
    j = json.loads(p.read_text())
    perf = j.get("performance_metrics", {})
    acc  = j.get("accuracy_metrics", {})
    row = {
        "run": p.parent.name,
        "type": "runner",
        "latency_mean_ms": perf.get("latency_mean_ms") or perf.get("latency_mean") or "",
        "memory_max_mb": perf.get("memory_max_mb") or perf.get("memory_max") or "",
    }
    # Common KITTI keys (if present)
    for k in [
        "Kitti metric/pred_instances_3d/KITTI/Car_3D_AP40_moderate_strict",
        "Kitti metric/pred_instances_3d/KITTI/Car_BEV_AP40_moderate_strict",
        "Kitti metric/pred_instances_3d/KITTI/Car_2D_AP40_moderate_strict",
        "Kitti metric/pred_instances_3d/KITTI/Car_3D_AP11_moderate_strict",
    ]:
        if k in acc:
            row[k] = acc[k]
    rows.append(row)

# Manual nuScenes outputs:
for p in runs_root.glob("*_eval/benchmark_perf.json"):
    j = json.loads(p.read_text())
    row = {
        "run": p.parent.name,
        "type": "manual",
        "latency_mean_ms": j.get("latency_mean_ms") or j.get("latency_mean") or "",
        "memory_max_mb": j.get("memory_max_mb") or j.get("memory_max") or "",
    }

    # Try to find nuscenes metrics_summary.json anywhere under the eval dir
    cand = []
    cand += glob.glob(str(p.parent / "**/metrics_summary.json"), recursive=True)
    cand += glob.glob(str(p.parent / "**/metrics_summary*.json"), recursive=True)
    if cand:
        ms = json.loads(Path(cand[0]).read_text())
        # nuscenes devkit typically uses these fields:
        row["nusc_mAP"] = ms.get("mean_ap", ms.get("mAP", ""))
        row["nusc_NDS"] = ms.get("nd_score", ms.get("NDS", ""))
    rows.append(row)

# Write CSV (no pandas needed)
if rows:
    keys = sorted({k for r in rows for k in r.keys()})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")).replace(",", ";") for k in keys) + "\n")

print(f"[Summary] Wrote: {out_csv}")
PY

echo
echo "============================================================"
echo "All runs attempted. Outputs under: $RUNS_ROOT"
echo "Summary CSV: $RUNS_ROOT/summary_metrics.csv"
echo "============================================================"
