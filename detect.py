"""
Step 1 — Detection (YOLOv8)

Runs YOLOv8 on a video (or directory of frames) and outputs per-frame
detections: [x, y, w, h, class, confidence].

Usage:
    python detect.py --source path/to/video.mp4 --weights yolov8n.pt --out outputs/detections.json
    python detect.py --source path/to/frames_dir --weights runs/train/exp/weights/best.pt --out outputs/detections.json
"""

import argparse
import json
import os
from pathlib import Path

from ultralytics import YOLO


def run_detection(source: str, weights: str = "yolov8n.pt", conf_thres: float = 0.25,
                   iou_thres: float = 0.45, device: str = "0", imgsz: int = 640):
    """
    Runs YOLOv8 inference over a video or frame sequence and returns
    a dict keyed by frame index, each value a list of detections.

    Each detection dict:
        {
            "bbox": [x, y, w, h],   # top-left x,y + width,height (pixel coords)
            "class": int,
            "class_name": str,
            "confidence": float
        }
    """
    model = YOLO(weights)

    results = model.predict(
        source=source,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        imgsz=imgsz,
        stream=True,      # generator -> memory-efficient for long videos
        verbose=False,
    )

    all_detections = {}
    for frame_idx, r in enumerate(results):
        frame_dets = []
        if r.boxes is not None:
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = r.names.get(cls_id, str(cls_id))
                frame_dets.append({
                    "bbox": [x1, y1, w, h],
                    "class": cls_id,
                    "class_name": cls_name,
                    "confidence": conf,
                })
        all_detections[frame_idx] = frame_dets

    return all_detections


def main():
    parser = argparse.ArgumentParser(description="Step 1: YOLOv8 detection")
    parser.add_argument("--source", required=True, help="Path to video file or frame directory")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLOv8 weights (pretrained or fine-tuned)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default="0", help="cuda device, e.g. '0' or 'cpu'")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--out", default="outputs/detections.json", help="Output JSON path")
    args = parser.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)

    detections = run_detection(
        source=args.source,
        weights=args.weights,
        conf_thres=args.conf,
        iou_thres=args.iou,
        device=args.device,
        imgsz=args.imgsz,
    )

    with open(args.out, "w") as f:
        json.dump(detections, f, indent=2)

    n_frames = len(detections)
    n_dets = sum(len(v) for v in detections.values())
    print(f"Done. {n_frames} frames processed, {n_dets} total detections.")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
