# RQ2 on UCF-Crime — Adapted Pipeline

## Why the original pipeline needed adapting

The original `train.py`/`evaluate.py` assumed supervised **per-object**
labels (each tracked agent labeled "real" vs. "clutter/false-positive").
**UCF-Crime has no such labels** — only:
- video-level labels (which folder/split-file a video is in), and
- for the official test set only, **frame-range** anomaly windows
  (`Temporal_Anomaly_Annotation_for_Testing_Videos.txt`).

So training had to move to the standard **weakly-supervised MIL** setup used
in the UCF-Crime literature (Sultani et al.), and evaluation to
**frame-level ROC-AUC** plus a **false-alarm rate on normal videos**, which
is the direct empirical stand-in for "reduces clutter-related false
positives."

## New / changed files

| File | Role |
|---|---|
| `dataset_utils.py` | Parses the actual nested folder structure + split/annotation files, returns `VideoRecord` lists for train/test |
| `video_graph_dataset.py` | Per video: run YOLOv8+BoT-SORT once, split into 32 snippets (Sultani-style), build one relational graph per snippet, cache to disk |
| `anomaly_model.py` | Two encoder variants sharing identical pooling/temporal/scoring heads: `IndependentEncoder` (no message passing — the RQ2 baseline) vs. `GCNEncoder` (reuses `model.py`'s GCN — the RQ2 "ours") |
| `train_mil.py` | Trains both variants with MIL ranking loss on video-level labels only (no frame labels used in training, by design) |
| `evaluate_ucf.py` | Computes frame-level ROC-AUC and false-alarm rate on `Testing_Normal_Videos`, for both variants — this directly answers RQ2 |

`detect.py`, `track.py`, `graph_utils.py`, `model.py` are reused as-is
(`video_graph_dataset.py` and `anomaly_model.py` import from them directly).
`train.py` / `evaluate.py` / `run_pipeline.py` from the original pipeline
assumed object-level labels and don't apply to UCF-Crime — kept in the repo
only as a reference for datasets that *do* have bbox/track ground truth.

## Workflow

```bash
# 1. Sanity-check dataset parsing
python dataset_utils.py --dataset_root /kaggle/input/datasets/minmints/ufc-crime-full-dataset

# 2. Build + cache snippet graphs (run once; this is the expensive step —
#    consider --limit N for a first smoke test before running on everything)
python video_graph_dataset.py \
    --dataset_root /kaggle/input/datasets/minmints/ufc-crime-full-dataset \
    --weights yolov8n.pt --split both --out_dir outputs/ucf_snippets

# 3. Train both variants (independent baseline + GCN) on identical data/splits
python train_mil.py --snippets_dir outputs/ucf_snippets/train --epochs 20 \
    --use_temporal --out_dir checkpoints

# 4. Evaluate — this produces the RQ2 answer
python evaluate_ucf.py --snippets_dir outputs/ucf_snippets/test \
    --checkpoint_gcn checkpoints/mil_gcn.pt \
    --checkpoint_baseline checkpoints/mil_independent.pt \
    --out outputs/ucf_eval_report.json
```

## How to read `ucf_eval_report.json` against RQ2

```json
{
  "gcn": {"frame_level_auc": ..., "false_alarm_rate_on_normal_videos": ..., ...},
  "baseline_independent": {"frame_level_auc": ..., "false_alarm_rate_on_normal_videos": ..., ...}
}
```

- **`frame_level_auc`**: higher = better overall discrimination between
  anomalous and normal frames. This is the metric most UCF-Crime papers
  report, so it's your comparability anchor to prior work.
- **`false_alarm_rate_on_normal_videos`**: fraction of `Testing_Normal_Videos`
  (crowded streets, traffic, pedestrians — genuinely cluttered but non-
  anomalous) where the model fired at least once. **This is the number that
  directly answers the "reduce clutter-related false positives" half of
  RQ2.** GCN < baseline here, at matched or better AUC, is the core result
  you're looking for.

## Important caveats to state in your methodology section

1. **No RQ2 answer exists yet** — I haven't run this (no GPU/dataset access
   in this environment). The pipeline is built to *produce* the answer once
   you run it on your Kaggle instance; treat the structure above as the
   experimental design, not a result.
2. **MIL, not full supervision** — because there's no frame-level training
   signal, the model only ever sees "this whole video contains an anomaly
   somewhere." Any frame-level precision therefore comes from the ranking
   loss's induced localization, not direct supervision — a real limitation
   worth naming explicitly (it's the same limitation the Sultani et al.
   baseline has, so it's comparable, not novel to your method).
3. **Snippet count (32) is a hyperparameter** you should ablate — finer
   snippets give better temporal localization for the AUC/false-alarm
   metrics but cost more compute per video.
4. **`clutter_bucket` from the original `evaluate.py` doesn't transfer**
   directly — there's no per-object clutter ground truth here. The
   false-alarm-rate-on-normal-videos metric above is the closest available
   proxy given what UCF-Crime actually annotates.
