"""
Step 3 — Graph construction

Builds one graph per frame (or sliding temporal window) from tracked agents.

Nodes  = agents present in the frame. Node features: bbox center (x, y),
         size (w, h), velocity (vx, vy), class one-hot, appearance embedding.
Edges  = relationships between agents, built via one of:
         - distance threshold
         - k-nearest neighbors
         - fully connected (for attention-weighted GCN variants)
Edge features: relative distance, relative velocity, angle of approach.

Output: a torch_geometric.data.Data object per frame, saved as a list.

Usage:
    python graph_utils.py --tracks outputs/tracks.json --frames_dir path/to/frames \
        --strategy knn --k 5 --out outputs/graphs.pt
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

try:
    import cv2
    import torchvision.models as tv_models
    import torchvision.transforms as T
    HAS_CV = True
except ImportError:
    HAS_CV = False


NUM_CLASSES = 10  # adjust to your dataset's class count


class AppearanceEmbedder:
    """
    Lightweight appearance feature extractor. If you already have BoT-SORT's
    internal re-ID embeddings available (e.g. by patching the tracker to
    expose them), swap this out — it's more consistent with the tracker's
    own association metric. This is a self-contained fallback using a
    pretrained ResNet18's penultimate layer.
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.enabled = HAS_CV
        if self.enabled:
            backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
            backbone.fc = torch.nn.Identity()
            self.model = backbone.to(device).eval()
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((128, 64)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    @torch.no_grad()
    def embed(self, frame_img, bbox):
        if not self.enabled or frame_img is None:
            return np.zeros(512, dtype=np.float32)
        x, y, w, h = [int(v) for v in bbox]
        x, y = max(x, 0), max(y, 0)
        crop = frame_img[y:y + max(h, 1), x:x + max(w, 1)]
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        feat = self.model(tensor).squeeze(0).cpu().numpy()
        return feat.astype(np.float32)


def class_one_hot(cls_id, num_classes=NUM_CLASSES):
    v = np.zeros(num_classes, dtype=np.float32)
    if 0 <= cls_id < num_classes:
        v[cls_id] = 1.0
    return v


def build_frame_graph(frame_entries, frame_img=None, embedder=None,
                       strategy="knn", k=5, dist_thresh=150.0):
    """
    frame_entries: list of dicts with keys track_id, bbox, class, confidence, velocity
    Returns a torch_geometric Data object, or None if <2 nodes.
    """
    n = len(frame_entries)
    if n < 2:
        return None

    node_feats = []
    positions = []
    track_ids = []

    for e in frame_entries:
        x, y, w, h = e["bbox"]
        cx, cy = x + w / 2, y + h / 2
        vx, vy = e.get("velocity", [0.0, 0.0])
        cls_oh = class_one_hot(e["class"])
        appearance = embedder.embed(frame_img, e["bbox"]) if embedder is not None else np.zeros(512, dtype=np.float32)

        feat = np.concatenate([
            [cx, cy, w, h, vx, vy, e.get("confidence", 1.0)],
            cls_oh,
            appearance,
        ])
        node_feats.append(feat)
        positions.append((cx, cy))
        track_ids.append(e["track_id"])

    node_feats = np.stack(node_feats).astype(np.float32)
    positions = np.array(positions, dtype=np.float32)

    # ---- Edge construction ----
    edge_index = []
    edge_attr = []

    def add_edge(i, j):
        dx, dy = positions[j] - positions[i]
        dist = float(np.hypot(dx, dy))
        angle = float(np.arctan2(dy, dx))
        rel_vel = node_feats[j][4:6] - node_feats[i][4:6]
        edge_index.append([i, j])
        edge_attr.append([dist, angle, rel_vel[0], rel_vel[1]])

    if strategy == "distance":
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(positions[i] - positions[j])
                if d <= dist_thresh:
                    add_edge(i, j)

    elif strategy == "knn":
        for i in range(n):
            dists = np.linalg.norm(positions - positions[i], axis=1)
            dists[i] = np.inf
            nearest = np.argsort(dists)[:k]
            for j in nearest:
                add_edge(i, int(j))

    elif strategy == "full":
        for i in range(n):
            for j in range(n):
                if i != j:
                    add_edge(i, j)

    else:
        raise ValueError(f"Unknown edge strategy: {strategy}")

    if len(edge_index) == 0:
        # fall back to at least connecting nearest neighbor to avoid empty graphs
        for i in range(n):
            j = (i + 1) % n
            add_edge(i, j)

    data = Data(
        x=torch.tensor(node_feats, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )
    data.track_ids = track_ids
    return data


def build_all_graphs(tracks_path, frames_dir=None, strategy="knn", k=5,
                      dist_thresh=150.0, use_appearance=True, device="cpu"):
    with open(tracks_path) as f:
        obj = json.load(f)
    per_frame = obj["per_frame"]

    embedder = AppearanceEmbedder(device=device) if use_appearance else None

    graphs = {}
    for frame_idx_str, entries in per_frame.items():
        frame_idx = int(frame_idx_str)
        frame_img = None
        if frames_dir and HAS_CV:
            candidate = Path(frames_dir) / f"{frame_idx:06d}.jpg"
            if candidate.exists():
                frame_img = cv2.imread(str(candidate))

        g = build_frame_graph(
            entries, frame_img=frame_img, embedder=embedder,
            strategy=strategy, k=k, dist_thresh=dist_thresh,
        )
        if g is not None:
            graphs[frame_idx] = g

    return graphs


def main():
    parser = argparse.ArgumentParser(description="Step 3: Graph construction")
    parser.add_argument("--tracks", required=True, help="Path to tracks.json from track.py")
    parser.add_argument("--frames_dir", default=None, help="Directory of extracted frames (for appearance features)")
    parser.add_argument("--strategy", default="knn", choices=["knn", "distance", "full"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--dist_thresh", type=float, default=150.0)
    parser.add_argument("--no_appearance", action="store_true", help="Skip appearance embedding (faster)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="outputs/graphs.pt")
    args = parser.parse_args()

    os.makedirs(Path(args.out).parent, exist_ok=True)

    graphs = build_all_graphs(
        tracks_path=args.tracks,
        frames_dir=args.frames_dir,
        strategy=args.strategy,
        k=args.k,
        dist_thresh=args.dist_thresh,
        use_appearance=not args.no_appearance,
        device=args.device,
    )

    torch.save(graphs, args.out)
    print(f"Built {len(graphs)} frame graphs.")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
