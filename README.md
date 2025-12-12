# NuScenes & KITTI Benchmark Report

## 1. Environment & Exact Commands
- Used my local system for nuscenes and used HPC for kitti.
- To setup HPC environment I followed this link provided by the professor: https://github.com/lkk688/DeepDataMiningLearning/blob/main/docs/source/mmdet3d_tutorial.md
- For Local 
Exact setup / execution commands:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel
pip install openmim && mim install "mmengine==0.10.4" "mmcv==2.1.0" "mmdet==3.3.0" "mmdet3d==1.4.0"
pip install nuscenes-devkit open3d imageio pillow tqdm psutil
pip install -e /home/manav/workspaces/mmdetection3d
./scripts/download_nuscenes_assets.sh
./scripts/run_nuscenes_inference.sh      # PointPillars + CenterPoint
KITTI_DATA_ROOT=/home/manav/workspaces/dataset \
  KITTI_ANN_FILE=/home/manav/workspaces/v1.0-mini/kitti_infos_val.pkl \
  ./scripts/run_kitti_inference.sh       # KITTI PointPillars + SECOND
./scripts/render_open3d_videos.sh        # stitch *_points.ply into MP4 demos
```

Run locations:
- **Local workstation (CPU)** handled both manual NuScenes jobs (`.../nuscenes_pointpillars_v1.0-mini_manual`, `.../nuscenes_centerpoint_v1.0-mini_manual`).
- **HPC cluster (NVIDIA H100 NVL, /fs/atipa paths)** produced the high-quality CenterPoint eval (`.../nuscenes_centerpoint_eval/`) and both KITTI baselines (`.../kitti_pointpillars_eval` and `.../kitti_second/eval_runner`), as seen in their `run.log` files.

## 2. Models & Datasets
| Dataset | Split | Model | Config | Checkpoint |
| --- | --- | --- | --- | --- |
| NuScenes v1.0-mini | `val` via manual custom loader | PointPillars | `pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py` | `hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth` |
| NuScenes v1.0-mini | `val` | CenterPoint pillar02 | `centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py` | `centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth` |
| KITTI 3D detection | `val` info splits | PointPillars (3-class) | `pointpillars_hv_secfpn_8xb6-160e_kitti-3d-3class.py` | `hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth` |
| KITTI 3D detection | `val` info splits | SECOND (SSN surrogate) | `second_hv_secfpn_8xb6-80e_kitti-3d-car.py` | `second_hv_secfpn_8xb6-80e_kitti-3d-car-75d9305e.pth` |

## 3. Metrics, Media, and Key Evidence
| Dataset | Model | Run site | Accuracy 1 | Accuracy 2 | Mean latency | Max mem |
| --- | --- | --- | --- | --- | --- | --- | 
| NuScenes v1.0-mini | CenterPoint | **HPC (NVIDIA H100 NVL)** | NDS 0.4494 | mAP 0.4796 | 219.3 ms (GPU) | 8 617 MB | 
| NuScenes v1.0-mini | CenterPoint (local) | Local workstation (CPU-only) | NDS 0.3963 | mAP 0.3883 | 682.9 ms (CPU) | 0 MB | 
| NuScenes v1.0-mini | PointPillars | Local workstation (CPU-only) | NDS 0.0000 | mAP 0.0000 | 915.7 ms (CPU) | 0 MB | 
| KITTI `val` | PointPillars | **HPC (NVIDIA H100 NVL)** | Car_3D_AP40_mod_strict 79.12 | Car_3D_AP11_mod_strict 77.61 | 13.7 ms | 17 096 MB |
| KITTI `val` | SECOND | **HPC (NVIDIA H100 NVL)** | Car_3D_AP40_mod_strict 0.00 | Car_3D_AP11_mod_strict 0.00 | 26.2 ms | 33 507 MB | 

Screenshots (see `results/01..04_*.png`) capture PointPillars NuScenes multi-view projections: front-view highway scene, dense urban scenario, truck close-up, and a pedestrian-heavy crossroads. Demo videos (Open3D renders) live in `results/*_open3d.mp4` for NuScenes PointPillars, NuScenes CenterPoint, and KITTI PointPillars.

## 4. Code Modifications (documented inline)
- `simple_infer_utils.py`: added `_boxes_to_sampled_points` and `save_ply_files` enrichment so every sample writes `{token}_points_with_boxes.ply`, plus `_update_experiment_summary` for `all_experiments_summary.json`. The new `set_all_seeds()` helper (with `mmengine_set_seed`) harmonizes RNG seeding across Python/NumPy/Torch/MMEngine.
- `simple_infer_main.py`: expanded CLI (`--seed`, `--eval-backend`, `--data-source` knobs) and plumbed metadata logging so each run records config/checkpoint/device/seed info before calling `run_manual_benchmark()` or `run_benchmark_evaluation()`.
- `create_open3d_videos.py` + `scripts/render_open3d_videos.sh`: added automated discovery of `_points.ply` + `_pred/_gt` line sets, uses Open3D’s off-screen renderer to write MP4 clips per experiment.
- Shell scripts (`download_nuscenes_assets.sh`, `run_nuscenes_inference.sh`, `run_kitti_inference.sh`) wrap the Python entrypoint with strict error checking and environment overrides, ensuring PNG/PLY/eval JSON land under `infer_runs/simple_infer_results/` for both datasets.

All touched sections carry descriptive comments to make future edits self-explanatory (see the environment helpers near the top of `simple_infer_utils.py`, the PLY utilities around `save_ply_files`, and the CLI docs in `simple_infer_main.py`).

## 5. Takeaways & Limitations
1. **CenterPoint dominates NuScenes**: With native support for sweep stacking and circle-NMS, it reaches ~0.40 NDS on the local CPU pass and climbs to 0.45 NDS / 0.48 mAP when the same config/checkpoint is replayed on the H100 cluster. PointPillars collapses (mAP=0) because its checkpoint expects lidar features that the simplified dataloader omits. Hence, random boxes are generated. I am currently working on that.
2. **KITTI PointPillars remains a strong baseline**: 77–79 AP at 13.7 ms inference proves the runner path is correctly configured; the PNG/PLY diagnostics show tight car boxes even on long-range LiDAR frames.
3. **SECOND as “SSN” substitute failed catastrophically**: All KITTI AP metrics drop to zero. Logs confirm predictions exist but confidence is ~0, suggesting a mismatch between checkpoint class heads and the KITTI 3-class evaluation script. Need to re-export weights or swap to a true SSN model for fairness.
4. **CPU vs GPU caveat**: NuScenes manual runs executed on CPU to simplify deployment, so latency metrics (≈0.8 s/sample) are not comparable with the GPU KITTI timings (≈14 ms) or the H100-backed CenterPoint evaluation (219 ms). I ran nuscenes offline as i was facing issues with eval on HPC.
