"""
anomaly_model.py — weakly-supervised video anomaly scorer, in two variants
that isolate exactly what RQ2 is asking about:

  - IndependentEncoder : per-node MLP, no message passing between agents
                          (= "treats objects independently", the baseline)
  - GCNEncoder          : reuses model.py's SpatialGCNEncoder, agents pass
                           messages to their neighbors before pooling
                           (= "Can GCNs better model relationships")

Both feed into the same snippet-pooling + temporal GRU + MIL scoring head,
so any difference in results is attributable to the relational modeling
step itself, not to unrelated architecture changes.

Empty-graph snippets (bins with <2 detections in video_graph_dataset.py)
are pooled as a zero vector, so a snippet with almost no clutter to reason
about contributes near-nothing to the score — appropriate for the FP-vs
-clutter framing.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from model import SpatialGCNEncoder


class IndependentEncoder(nn.Module):
    """Baseline: each detected agent is embedded independently; no edges used
    at all, even if the graph has them. This is the fair 'objects treated
    independently' counterpart to GCNEncoder."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, edge_index=None):
        return self.mlp(x)  # edge_index intentionally ignored


class SnippetPooler(nn.Module):
    """Pools node embeddings in a snippet's graph into one vector.
    Handles the None (empty-graph) case as a zero vector."""

    def __init__(self, dim):
        self.dim = dim
        super().__init__()

    def forward(self, encoder, data, device):
        if data is None:
            return torch.zeros(self.dim, device=device)
        data = data.to(device)
        h = encoder(data.x, data.edge_index)
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        pooled = global_mean_pool(h, batch)  # [1, dim]
        return pooled.squeeze(0)


class MILAnomalyScorer(nn.Module):
    """
    encoder_type: "gcn" or "independent"
    Produces one anomaly score in [0, 1] per snippet, for a full video
    (sequence of snippet graphs).
    """

    def __init__(self, node_in_dim, hidden_dim=128, encoder_type="gcn",
                 gcn_layers=3, conv_type="gcn", use_temporal=True, dropout=0.2):
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.use_temporal = use_temporal

        if encoder_type == "gcn":
            self.encoder = SpatialGCNEncoder(
                in_dim=node_in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
                num_layers=gcn_layers, conv_type=conv_type, dropout=dropout,
            )
        elif encoder_type == "independent":
            self.encoder = IndependentEncoder(
                in_dim=node_in_dim, hidden_dim=hidden_dim, num_layers=gcn_layers, dropout=dropout,
            )
        else:
            raise ValueError(encoder_type)

        self.pooler = SnippetPooler(hidden_dim)

        if use_temporal:
            self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, snippet_graphs, device):
        """snippet_graphs: list of Data-or-None, length = num_snippets.
        Returns: scores tensor [num_snippets]"""
        vecs = [self.pooler(self.encoder, g, device) for g in snippet_graphs]
        seq = torch.stack(vecs).unsqueeze(0)  # [1, T, hidden]

        if self.use_temporal:
            seq, _ = self.temporal(seq)

        scores = self.score_head(seq.squeeze(0)).squeeze(-1)  # [T]
        return scores
