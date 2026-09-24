"""
train_mil.py — weakly-supervised MIL training on video-level labels only
(Sultani et al. 2018 style ranking loss), since UCF-Crime has no frame-level
training labels.

Per training step: sample one anomaly video and one normal video (a
positive/negative "bag" pair). Each video is scored snippet-by-snippet.
Loss pushes max(anomaly snippet scores) above max(normal snippet scores)
by a margin, plus temporal-smoothness and sparsity regularizers on the
anomaly video's scores (standard terms from the original MIL formulation —
they discourage erratic, all-snippets-flagged degenerate solutions).

Trains BOTH encoder_type variants back to back so the RQ2 comparison uses
identical data, splits, and hyperparameters end to end.

Usage:
    python train_mil.py --snippets_dir outputs/ucf_snippets/train --epochs 20 \
        --out_dir checkpoints
"""

import argparse
import random
from pathlib import Path

import torch
from torch.optim import Adagrad
from tqdm import tqdm

from anomaly_model import MILAnomalyScorer


def load_video_snippets(snippets_dir):
    anomaly_files, normal_files = [], []
    for f in Path(snippets_dir).glob("*.pt"):
        obj = torch.load(f)
        (anomaly_files if obj["label"] == 1 else normal_files).append(f)
    return anomaly_files, normal_files


def mil_ranking_loss(anomaly_scores, normal_scores, lambda_smooth=8e-5, lambda_sparse=8e-5):
    max_anom = anomaly_scores.max()
    max_norm = normal_scores.max()
    ranking = torch.clamp(1.0 - max_anom + max_norm, min=0.0)

    # temporal smoothness: adjacent snippet scores shouldn't jump around
    smooth = torch.sum((anomaly_scores[1:] - anomaly_scores[:-1]) ** 2)
    # sparsity: only a few snippets in an anomaly video should actually fire
    sparse = torch.sum(anomaly_scores)

    return ranking + lambda_smooth * smooth + lambda_sparse * sparse


def run_training(encoder_type, anomaly_files, normal_files, node_in_dim, args, device):
    model = MILAnomalyScorer(
        node_in_dim=node_in_dim, hidden_dim=args.hidden_dim, encoder_type=encoder_type,
        gcn_layers=args.gcn_layers, conv_type=args.conv_type, use_temporal=args.use_temporal,
    ).to(device)
    optimizer = Adagrad(model.parameters(), lr=args.lr)

    n_steps = min(len(anomaly_files), len(normal_files))

    for epoch in range(1, args.epochs + 1):
        random.shuffle(anomaly_files)
        random.shuffle(normal_files)
        model.train()
        epoch_loss = 0.0

        for i in tqdm(range(n_steps), desc=f"[{encoder_type}] epoch {epoch}", leave=False):
            anom_obj = torch.load(anomaly_files[i])
            norm_obj = torch.load(normal_files[i])

            optimizer.zero_grad()
            anom_scores = model(anom_obj["snippets"], device)
            norm_scores = model(norm_obj["snippets"], device)

            loss = mil_ranking_loss(anom_scores, norm_scores)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"[{encoder_type}] epoch {epoch:03d} | avg loss {epoch_loss / n_steps:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train MIL anomaly scorer (GCN vs. independent baseline)")
    parser.add_argument("--snippets_dir", required=True, help="Dir of cached train snippet .pt files")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gcn_layers", type=int, default=3)
    parser.add_argument("--conv_type", default="gcn", choices=["gcn", "gat"])
    parser.add_argument("--use_temporal", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    anomaly_files, normal_files = load_video_snippets(args.snippets_dir)
    print(f"Loaded {len(anomaly_files)} anomaly / {len(normal_files)} normal training videos")

    # infer node feature dim from the first non-empty snippet
    node_in_dim = None
    for f in anomaly_files + normal_files:
        obj = torch.load(f)
        for g in obj["snippets"]:
            if g is not None:
                node_in_dim = g.x.shape[1]
                break
        if node_in_dim:
            break
    if node_in_dim is None:
        raise RuntimeError("Could not infer node feature dim — all cached snippets are empty.")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for encoder_type in ["independent", "gcn"]:
        model = run_training(encoder_type, list(anomaly_files), list(normal_files), node_in_dim, args, args.device)
        out_path = Path(args.out_dir) / f"mil_{encoder_type}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "encoder_type": encoder_type,
            "node_in_dim": node_in_dim,
            "config": vars(args),
        }, out_path)
        print(f"Saved {encoder_type} model to {out_path}")


if __name__ == "__main__":
    main()
