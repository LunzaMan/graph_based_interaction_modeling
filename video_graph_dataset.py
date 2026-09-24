"""
video_graph_dataset.py — turns one UCF-Crime video into a sequence of
per-snippet relational graphs (Sultani-style: each video -> fixed number of
snippets, standard is 32).

Why snippets and not every frame: UCF-Crime has ~1900 videos; per-frame
graphs for all of them is not tractable for a weakly-labeled MIL setup where
supervision only exists at (roughly) the snippet/video level anyway.

Pipeline per video:
  1. Run YOLOv8 + BoT-SORT once over the whole video (track.py's run_tracking)
  2. Split frame indices into NUM_SNIPPETS contiguous, equal-length bins
  3. For each bin, pick the frame with the most detections (best signal for
     relational structure) and build its graph via graph_utils.build_frame_graph
  4. Bins with <2 detected agents get an empty-graph placeholder (handled
     downstream by the pooling layer as a zero vector)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from track import run_tracking, compute_velocities
from graph_utils import build_frame_graph, AppearanceEmbedder

NUM_SNIPPETS = 32  # standard choice in the UCF-Crime MIL literature (Sultani et al.)


def build_video_snippets(video_path, weights="yolov8n.pt", num_snippets=NUM_SNIPPETS,
                          strategy="knn", k=5, use_appearance=True, device="cpu",
                          conf_thres=0.25):
    tracks, per_frame = run_tracking(source=str(video_path), weights=weights,
                                      conf_thres=conf_thres, device=device)
    tracks = compute_velocities(tracks)

    frame_indices = sorted(per_frame.keys())
    if len(frame_indices) == 0:
        return [None] * num_snippets

    bins = np.array_split(frame_indices, num_snippets)
    embedder = AppearanceEmbedder(device=device) if use_appearance else None

    snippet_graphs = []
    snippet_ranges = []  # (start_frame, end_frame) inclusive, for mapping scores back to frame-level GT
    for b in bins:
        if len(b) == 0:
            snippet_graphs.append(None)
            snippet_ranges.append(None)
            continue
        snippet_ranges.append((int(b[0]), int(b[-1])))
        # pick the frame with the most detections in this bin (richest relational signal)
        best_frame = max(b, key=lambda fidx: len(per_frame.get(fidx, [])))
        entries = per_frame.get(best_frame, [])
        g = build_frame_graph(entries, frame_img=None, embedder=embedder,
                               strategy=strategy, k=k)
        snippet_graphs.append(g)

    return snippet_graphs, snippet_ranges


def build_and_cache(video_record, out_dir, weights="yolov8n.pt", **kwargs):
    """video_record: dataset_utils.VideoRecord. Caches snippets as a .pt file
    named after the video, so re-runs (training reload / different model
    variants) don't need to re-run detection+tracking."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_record.path.stem}.pt"

    if out_path.exists():
        return out_path

    snippets, ranges = build_video_snippets(video_record.path, weights=weights, **kwargs)
    torch.save({
        "snippets": snippets,
        "snippet_ranges": ranges,
        "label": video_record.label,
        "class_name": video_record.class_name,
        "gt_windows": video_record.gt_windows,
        "video_path": str(video_record.path),
    }, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build cached snippet graphs for all UCF-Crime videos")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--out_dir", default="outputs/ucf_snippets")
    parser.add_argument("--split", default="both", choices=["train", "test", "both"])
    parser.add_argument("--strategy", default="knn", choices=["knn", "distance", "full"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Cap videos per split (debug runs)")
    args = parser.parse_args()

    from dataset_utils import load_ucf_crime, summarize

    records = load_ucf_crime(args.dataset_root)
    summarize(records)

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        vids = records[split]
        if args.limit:
            vids = vids[: args.limit]
        out_dir = Path(args.out_dir) / split
        for i, rec in enumerate(vids):
            try:
                path = build_and_cache(
                    rec, out_dir, weights=args.weights,
                    strategy=args.strategy, k=args.k, device=args.device,
                )
                print(f"[{split}] ({i+1}/{len(vids)}) {rec.path.name} -> {path}")
            except Exception as e:
                print(f"[{split}] FAILED on {rec.path.name}: {e}")


if __name__ == "__main__":
    main()
