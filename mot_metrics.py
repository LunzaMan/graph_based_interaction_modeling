"""
Step 7 (tracking-quality sanity check) — MOTA / IDF1

Confirms the GCN false-positive filtering step isn't damaging tracking
continuity (e.g. by dropping real objects it misclassifies as clutter,
which would fragment tracks).

Requires ground-truth tracks in MOT-Challenge format:
    frame, track_id, x, y, w, h, conf, class, visibility
and predicted tracks in the same format (post-GCN-filtering).

Usage:
    python mot_metrics.py --gt gt_tracks.txt --pred pred_tracks_post_gcn.txt
"""

import argparse

import motmetrics as mm
import numpy as np
import pandas as pd


def load_mot_format(path):
    cols = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]
    df = pd.read_csv(path, header=None, names=cols, usecols=range(9))
    return df


def compute_mot_metrics(gt_path, pred_path):
    gt = load_mot_format(gt_path)
    pred = load_mot_format(pred_path)

    acc = mm.MOTAccumulator(auto_id=True)

    frames = sorted(set(gt.frame.unique()) | set(pred.frame.unique()))
    for frame in frames:
        gt_frame = gt[gt.frame == frame]
        pred_frame = pred[pred.frame == frame]

        gt_ids = gt_frame.id.values
        pred_ids = pred_frame.id.values

        gt_boxes = gt_frame[["x", "y", "w", "h"]].values
        pred_boxes = pred_frame[["x", "y", "w", "h"]].values

        dist_matrix = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, dist_matrix)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["mota", "motp", "idf1", "idp", "idr", "num_switches",
                 "num_false_positives", "num_misses", "num_fragmentations"],
        name="overall",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Step 7: MOTA/IDF1 tracking metrics")
    parser.add_argument("--gt", required=True, help="Ground truth tracks (MOT format .txt)")
    parser.add_argument("--pred", required=True, help="Predicted tracks (MOT format .txt)")
    args = parser.parse_args()

    summary = compute_mot_metrics(args.gt, args.pred)
    print(summary.to_string())


if __name__ == "__main__":
    main()
