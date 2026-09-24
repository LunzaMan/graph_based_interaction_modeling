"""
dataset_utils.py — UCF-Crime dataset adapter

UCF-Crime has NO bounding-box or track-level ground truth. It only provides:
  - Video-level labels (which category folder a video sits in / which split
    file it's listed in: "Normal" vs. one of 13 anomaly classes)
  - For the OFFICIAL TEST split only: temporal start/end FRAME annotations
    of the anomalous segment(s), in
    Temporal_Anomaly_Annotation_for_Testing_Videos.txt

This means the RQ2 pipeline cannot be trained with the node-level
"real object / clutter-FP" labels used in train.py originally. Instead we
adopt the standard UCF-Crime weakly-supervised setup: MIL over video-level
labels (see anomaly_model.py / train_mil.py), and evaluate with frame-level
ROC-AUC using the temporal annotation file (see evaluate_ucf.py).

Layout handled here (note the doubled directory names, e.g.
".../Anomaly-Videos-Part-1/Anomaly-Videos-Part-1/Assault/..."):

DATASET/
  ReadMe-Anomaly-Detection.txt
  Anomaly_Train.txt                                  <- video-level train list (anomaly classes only)
  Temporal_Anomaly_Annotation_for_Testing_Videos.txt  <- test GT: name, class, start1,end1,start2,end2
  UCF_Crimes-Train-Test-Split/
    Anomaly_Detection_splits/Anomaly_Train.txt
    Anomaly_Detection_splits/Anomaly_Test.txt
  Anomaly-Videos-Part-{1..4}/Anomaly-Videos-Part-{1..4}/<Class>/<video>.mp4
  Training-Normal-Videos-Part-{1,2}/Training-Normal-Videos-Part-{1,2}/<video>.mp4
  Testing_Normal_Videos/Testing_Normal_Videos_Anomaly/<video>.mp4
  Normal_Videos_for_Event_Recognition/Normal_Videos_for_Event_Recognition/<video>.mp4
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


ANOMALY_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
    "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting",
    "Stealing", "Vandalism",
]


@dataclass
class VideoRecord:
    path: Path
    label: int              # 0 = normal, 1 = anomaly
    class_name: str         # e.g. "Fighting", "Normal"
    split: str               # "train" or "test"
    # test-only: list of (start_frame, end_frame) anomalous windows; empty if normal
    gt_windows: List[tuple] = field(default_factory=list)


def _index_videos_by_name(dataset_root: Path):
    """Walks the whole dataset once and maps filename -> full path, so the
    split/annotation files (which list bare filenames) can be resolved
    regardless of the nested/duplicated directory structure."""
    index = {}
    for root, _, files in os.walk(dataset_root):
        for f in files:
            if f.lower().endswith(".mp4"):
                index[f] = Path(root) / f
    return index


def _parse_temporal_annotations(ann_path: Path):
    """
    Each line: <video_name> <class> <start1> <end1> <start2> <end2>
    -1 means "no second window". Normal test videos have class "Normal" and
    start/end of -1 -1 -1 -1 (no anomaly anywhere in the video).
    """
    gt = {}
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            name, cls = parts[0], parts[1]
            nums = [int(x) for x in parts[2:6]]
            windows = [(nums[i], nums[i + 1]) for i in range(0, 4, 2) if nums[i] != -1]
            gt[name] = {"class": cls, "windows": windows}
    return gt


def load_ucf_crime(dataset_root: str, use_official_splits: bool = True):
    """
    Returns dict: {"train": [VideoRecord, ...], "test": [VideoRecord, ...]}
    """
    root = Path(dataset_root)
    name_index = _index_videos_by_name(root)

    splits_dir = root / "UCF_Crimes-Train-Test-Split" / "Anomaly_Detection_splits"
    train_list_path = splits_dir / "Anomaly_Train.txt" if use_official_splits and splits_dir.exists() else root / "Anomaly_Train.txt"
    test_list_path = splits_dir / "Anomaly_Test.txt" if use_official_splits and splits_dir.exists() else None

    temporal_ann_path = root / "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    temporal_gt = _parse_temporal_annotations(temporal_ann_path) if temporal_ann_path.exists() else {}

    records = {"train": [], "test": []}

    # ---- Train: anomaly videos (from Anomaly_Train.txt, video-level label only) ----
    if train_list_path.exists():
        with open(train_list_path) as f:
            for line in f:
                name = line.strip().split("/")[-1]
                if not name:
                    continue
                if name in name_index:
                    cls = next((c for c in ANOMALY_CLASSES if name.startswith(c)), "Unknown")
                    records["train"].append(VideoRecord(
                        path=name_index[name], label=1, class_name=cls, split="train"))

    # ---- Train: normal videos (Training-Normal-Videos-Part-1/2 folders) ----
    for part in ["Training-Normal-Videos-Part-1", "Training-Normal-Videos-Part-2"]:
        part_dir = root / part
        if part_dir.exists():
            for vid_path in part_dir.rglob("*.mp4"):
                records["train"].append(VideoRecord(
                    path=vid_path, label=0, class_name="Normal", split="train"))

    # ---- Test: use temporal annotation file as the source of truth (has both
    # anomaly and normal test videos with frame-level windows) ----
    if temporal_gt:
        for name, info in temporal_gt.items():
            if name not in name_index:
                continue
            is_anomaly = info["class"] != "Normal" and len(info["windows"]) > 0
            records["test"].append(VideoRecord(
                path=name_index[name],
                label=1 if is_anomaly else 0,
                class_name=info["class"],
                split="test",
                gt_windows=info["windows"],
            ))
    else:
        # fallback: Testing_Normal_Videos folder only, no anomaly test videos found
        test_normal_dir = root / "Testing_Normal_Videos"
        if test_normal_dir.exists():
            for vid_path in test_normal_dir.rglob("*.mp4"):
                records["test"].append(VideoRecord(
                    path=vid_path, label=0, class_name="Normal", split="test"))

    return records


def summarize(records):
    for split in ("train", "test"):
        vids = records[split]
        n_anom = sum(1 for v in vids if v.label == 1)
        n_norm = sum(1 for v in vids if v.label == 0)
        print(f"{split}: {len(vids)} videos ({n_anom} anomaly, {n_norm} normal)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    args = parser.parse_args()

    recs = load_ucf_crime(args.dataset_root)
    summarize(recs)
    print("\nExample train record:", recs["train"][0] if recs["train"] else None)
    print("Example test record:", recs["test"][0] if recs["test"] else None)
