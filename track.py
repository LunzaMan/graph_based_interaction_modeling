"""
Step 2 — Tracking (BoT-SORT)

Ultralytics ships BoT-SORT as a built-in tracker, so the simplest path is to
call YOLO's .track() directly (it fuses detection + BoT-SORT tracking in one
pass, including the appearance re-ID embeddings we want as GCN node features
later). This module wraps that call and produces per-track trajectories.

Usage:
    python track.py --source path/to/video.mp4 --weights yolov8n.pt --out outputs/tracks.json
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO


def run_tracking(source: str, weights: str = "yolov8n.pt", tracker_cfg: str = "botsort.yaml",
                  conf_thres: float = 0.25, iou_thres: float = 0.45, device: str = "0",
                  imgsz: int = 640):
    """
    Runs YOLOv8 + BoT-SORT tracking. Returns:
      - tracks: dict[track_id] -> list of {frame, bbox, confidence, class}
      - per_frame: dict[frame_idx] -> list of {track_id, bbox, confidence, class}
    """
    model = YOLO(weights)

    results = model.track(
        source=source,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        imgsz=imgsz,
        tracker=tracker_cfg,   # 'botsort.yaml' ships with ultralytics
        persist=True,
        stream=True,
        verbose=False,
    )

    tracks = defaultdict(list)
    per_frame = {}

    for frame_idx, r in enumerate(results):
        frame_entries = []
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            xyxy = r.boxes.xyxy.tolist()
            confs = r.boxes.conf.tolist()
            clses = r.boxes.cls.int().tolist()

            for tid, box, conf, cls_id in zip(ids, xyxy, confs, clses):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                entry = {
                    "frame": frame_idx,
                    "bbox": [x1, y1, w, h],
                    "confidence": conf,
                    "class": cls_id,
                    "class_name": r.names.get(cls_id, str(cls_id)),
                }
                tracks[tid].append(entry)
                frame_entries.append({"track_id": tid, **entry})

        per_frame[frame_idx] = frame_entries

    return dict(tracks), per_frame


def compute_velocities(tracks: dict):
    """
    Adds velocity (dx, dy per frame-step) to each track entry based on
    bbox-center deltas from the previous frame in that track. First frame
    of each track gets velocity [0, 0].
    """
    for tid, entries in tracks.items():
        entries.sort(key=lambda e: e["frame"])
        prev_center = None
        prev_frame = None
        for e in entries:
            x, y, w, h = e["bbox"]
            cx, cy = x + w / 2, y + h / 2
            if prev_center is None:
                e["velocity"] = [0.0, 0.0]
            else:
                dt = max(e["frame"] - prev_frame, 1)
                e["velocity"] = [(cx - prev_center[0]) / dt, (cy - prev_center[1]) / dt]
            prev_center = (cx, cy)
            prev_frame = e["frame"]
    return tracks


def main():
    parser = argparse.ArgumentParser(description="Step 2: BoT-SORT tracking")
    parser.add_argument("--source", required=True, help="Path to video file")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 weights")
    parser.add_argument("--tracker", default="botsort.yaml", help="Tracker config (botsort.yaml)")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default="outputs/tracks.json")
    args = parser.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)

    tracks, per_frame = run_tracking(
        source=args.source,
        weights=args.weights,
        tracker_cfg=args.tracker,
        conf_thres=args.conf,
        iou_thres=args.iou,
        device=args.device,
        imgsz=args.imgsz,
    )
    tracks = compute_velocities(tracks)

    out_obj = {
        "tracks": tracks,
        "per_frame": per_frame,
    }
    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=2)

    print(f"Done. {len(tracks)} unique tracks across {len(per_frame)} frames.")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
