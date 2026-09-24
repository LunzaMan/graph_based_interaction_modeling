"""
evaluate_ucf.py — answers RQ2 on UCF-Crime.

Two things computed, for both encoder_type in {"independent" (baseline),
"gcn" (ours)}, using the SAME trained checkpoints from train_mil.py:

  1. Frame-level ROC-AUC (the standard UCF-Crime anomaly-detection metric).
     Snippet scores are broadcast across the frame range they cover
     (snippet_ranges, saved by video_graph_dataset.py) and compared against
     frame-level ground truth built from gt_windows / Temporal_Anomaly_
     Annotation_for_Testing_Videos.txt.

  2. False-alarm rate on NORMAL test videos specifically, at a fixed
     decision threshold. This is the direct empirical answer to "reduce
     clutter-related false positives": Testing_Normal_Videos contains
     crowded/cluttered but non-anomalous scenes (traffic, pedestrians,
     etc.), so a lower false-alarm rate here for the GCN model vs. the
     independent-object baseline is the evidence RQ2 asks for.

Usage:
    python evaluate_ucf.py --snippets_dir outputs/ucf_snippets/test \
        --checkpoint_gcn checkpoints/mil_gcn.pt \
        --checkpoint_baseline checkpoints/mil_independent.pt \
        --out outputs/ucf_eval_report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from anomaly_model import MILAnomalyScorer


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    model = MILAnomalyScorer(
        node_in_dim=ckpt["node_in_dim"], hidden_dim=cfg["hidden_dim"],
        encoder_type=ckpt["encoder_type"], gcn_layers=cfg["gcn_layers"],
        conv_type=cfg["conv_type"], use_temporal=cfg["use_temporal"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def frame_level_scores_and_labels(obj, scores):
    """
    Broadcasts per-snippet scores to per-frame scores using snippet_ranges,
    and builds frame-level binary GT from gt_windows. Video length is taken
    as the max frame index covered by any snippet range.
    """
    ranges = obj["snippet_ranges"]
    valid = [(r, s) for r, s in zip(ranges, scores.tolist()) if r is not None]
    if not valid:
        return np.array([]), np.array([])

    total_frames = max(r[1] for r, _ in valid) + 1
    frame_scores = np.zeros(total_frames, dtype=np.float32)
    for (start, end), s in valid:
        frame_scores[start:end + 1] = s

    frame_labels = np.zeros(total_frames, dtype=np.int32)
    for start, end in obj["gt_windows"]:
        s = max(start, 0)
        e = min(end, total_frames - 1)
        if s <= e:
            frame_labels[s:e + 1] = 1

    return frame_scores, frame_labels


def evaluate(model, snippets_dir, device, decision_thresh=0.5):
    all_scores, all_labels = [], []
    normal_video_alarms = []  # 1 if ANY frame in a normal video crossed threshold

    for f in Path(snippets_dir).glob("*.pt"):
        obj = torch.load(f)
        with torch.no_grad():
            scores = model(obj["snippets"], device).cpu()

        frame_scores, frame_labels = frame_level_scores_and_labels(obj, scores)
        if frame_scores.size == 0:
            continue

        all_scores.append(frame_scores)
        all_labels.append(frame_labels)

        if obj["label"] == 0:  # normal video -> any positive prediction is a false alarm
            fired = bool((frame_scores >= decision_thresh).any())
            normal_video_alarms.append(fired)

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    auc = roc_auc_score(all_labels, all_scores) if len(set(all_labels.tolist())) > 1 else float("nan")
    false_alarm_rate = float(np.mean(normal_video_alarms)) if normal_video_alarms else float("nan")

    return {
        "frame_level_auc": float(auc),
        "false_alarm_rate_on_normal_videos": false_alarm_rate,
        "n_normal_test_videos": len(normal_video_alarms),
        "n_frames_evaluated": int(all_labels.size),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GCN vs. baseline on UCF-Crime (answers RQ2)")
    parser.add_argument("--snippets_dir", required=True, help="Cached test snippets dir")
    parser.add_argument("--checkpoint_gcn", required=True)
    parser.add_argument("--checkpoint_baseline", required=True)
    parser.add_argument("--decision_thresh", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="outputs/ucf_eval_report.json")
    args = parser.parse_args()

    gcn_model = load_model(args.checkpoint_gcn, args.device)
    baseline_model = load_model(args.checkpoint_baseline, args.device)

    report = {
        "gcn": evaluate(gcn_model, args.snippets_dir, args.device, args.decision_thresh),
        "baseline_independent": evaluate(baseline_model, args.snippets_dir, args.device, args.decision_thresh),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
