# RQ2 Pipeline: YOLOv8 + BoT-SORT + GCN for Relational Modeling & FP Reduction

## Install
```bash
pip install -r requirements.txt
```

## File map (maps directly to the 9 steps)

| Step | File | Purpose |
|---|---|---|
| 1 | `detect.py` | YOLOv8 detection (standalone; mainly for inspection/debugging) |
| 2 | `track.py` | YOLOv8 + BoT-SORT tracking, adds velocity features |
| 3 | `graph_utils.py` | Builds per-frame graphs (nodes=agents, edges=relations) |
| 4 | `model.py` | ST-GCN architecture (spatial GCN/GAT + temporal GRU, dual heads) |
| 5 | `train.py` | Training loop with sequence-level train/val/test split |
| 6/7 | `evaluate.py` | Baseline vs. GCN comparison, clutter-bucketed P/R/F1, significance test |
| 7 | `mot_metrics.py` | MOTA/IDF1 tracking-quality sanity check |
| 8/9 | `run_pipeline.py` | End-to-end inference: video in -> GCN-filtered tracks out |

## Typical workflow

**1. Prepare per-sequence graphs for training**
For each labeled training video/sequence:
```bash
python track.py --source seq01.mp4 --weights yolov8n.pt --out outputs/seq01_tracks.json
python graph_utils.py --tracks outputs/seq01_tracks.json --strategy knn --k 5 \
    --out outputs/graphs_by_sequence/seq01.pt
```
Repeat for every sequence, all landing in `outputs/graphs_by_sequence/`.

**2. Label the graphs**
Build `outputs/labels.json` per the schema documented at the top of `train.py`
(node_labels: 0=clutter/false-positive, 1=real object; edge_labels optional).

**3. Train**
```bash
python train.py --graphs_dir outputs/graphs_by_sequence --labels outputs/labels.json \
    --epochs 30 --use_temporal --out checkpoints/stgcn.pt
```

**4. Evaluate against the baseline**
```bash
python evaluate.py --graphs_dir outputs/graphs_by_sequence --labels outputs/labels.json \
    --checkpoint checkpoints/stgcn.pt --out outputs/eval_report.json
```
This gives you the core RQ2 evidence: F1/precision/recall/FP-rate split by
clutter density (low/medium/high) for baseline vs. GCN, plus a paired
t-test on per-sequence F1.

**5. Tracking-quality sanity check**
Export GT and post-GCN-filtered predictions in MOT format, then:
```bash
python mot_metrics.py --gt gt_tracks.txt --pred outputs/pred_tracks_post_gcn.txt
```

**6. Run the full pipeline on a new/unlabeled video**
```bash
python run_pipeline.py --source new_video.mp4 --weights yolov8n.pt \
    --checkpoint checkpoints/stgcn.pt --out outputs/pred_tracks_post_gcn.txt
```

## Ablations for the paper (Step 8)
- `--use_temporal` on/off in `train.py` -> static GCN vs. spatio-temporal GCN
- `--conv_type gcn` vs `gat` -> plain vs. attention-weighted edges
- `--strategy knn|distance|full` in `graph_utils.py` -> edge construction strategy
- Vary `--k` / `--dist_thresh` -> sensitivity to neighborhood definition
- Re-run `evaluate.py`'s clutter-bucketed breakdown for each variant to see
  where the GCN's advantage grows/shrinks as clutter increases — this is the
  strongest direct evidence for the RQ2 claim.

## Notes / things to adapt to your dataset
- `NUM_CLASSES` in `graph_utils.py` — set to your actual class count.
- `clutter_bucket()` in `evaluate.py` uses node-count-per-frame as a clutter
  proxy; swap for a bbox-IoU-overlap-based definition if that fits your
  domain's notion of "clutter" better.
- The `AppearanceEmbedder` in `graph_utils.py` is a ResNet18 fallback. If you
  patch Ultralytics' BoT-SORT to expose its own re-ID embeddings, wire those
  in instead for consistency with the tracker's association metric.
