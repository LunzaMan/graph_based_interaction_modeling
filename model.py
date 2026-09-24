"""
Step 4 — Spatio-Temporal GCN architecture

Two heads, trainable jointly or separately:
  - Node classification: "real object" vs. "clutter artifact" (false-positive suppression)
  - Edge classification: interaction type / strength (relational modeling)

Spatial reasoning: stacked GCNConv (or GATConv for attention-weighted edges)
Temporal reasoning: GRU applied across the sequence of per-frame node embeddings,
                     matched across frames via track_id.

This is intentionally modular: swap GCNConv <-> GATConv, or drop the GRU
for a "static GCN only" ablation (Step 8 in the write-up).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


class SpatialGCNEncoder(nn.Module):
    """Stacked graph conv layers producing a per-node embedding for one frame's graph."""

    def __init__(self, in_dim, hidden_dim=128, out_dim=128, num_layers=3,
                 conv_type="gcn", heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()

        def make_conv(i, o):
            if conv_type == "gat":
                return GATConv(i, o // heads, heads=heads, dropout=dropout)
            return GCNConv(i, o)

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.convs.append(make_conv(dims[i], dims[i + 1]))

        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in dims[1:]])

    def forward(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h  # [num_nodes, out_dim]


class TemporalAggregator(nn.Module):
    """
    GRU over the sequence of embeddings for a single tracked agent across frames.
    Expects input shaped [batch=num_tracks_in_window, seq_len, feat_dim].
    """

    def __init__(self, in_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, seq):
        out, _ = self.gru(seq)
        return out  # [batch, seq_len, hidden_dim] — take last step for a per-track summary


class STGCN(nn.Module):
    """
    Full spatio-temporal model.

    forward_frame(): spatial-only pass for a single frame's graph -> used
                      both for training the static-GCN ablation and as the
                      per-frame building block for the full spatio-temporal model.

    forward_sequence(): runs forward_frame() over a window of frames, then
                         aggregates each track's embeddings temporally.
    """

    def __init__(self, node_in_dim, edge_in_dim=4, hidden_dim=128,
                 gcn_layers=3, conv_type="gcn", use_temporal=True,
                 num_node_classes=2, num_edge_classes=3, dropout=0.2):
        super().__init__()
        self.use_temporal = use_temporal

        self.spatial_encoder = SpatialGCNEncoder(
            in_dim=node_in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
            num_layers=gcn_layers, conv_type=conv_type, dropout=dropout,
        )

        if use_temporal:
            self.temporal_agg = TemporalAggregator(in_dim=hidden_dim, hidden_dim=hidden_dim)

        # Node head: false-positive vs. real object
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_node_classes),
        )

        # Edge head: interaction type, takes concat of endpoint embeddings + edge_attr
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_edge_classes),
        )

    def forward_frame(self, data):
        """Single-frame spatial pass. data: torch_geometric.data.Data"""
        h = self.spatial_encoder(data.x, data.edge_index)
        node_logits = self.node_classifier(h)

        edge_logits = None
        if data.edge_index.numel() > 0:
            src, dst = data.edge_index
            edge_in = torch.cat([h[src], h[dst], data.edge_attr], dim=1)
            edge_logits = self.edge_classifier(edge_in)

        return node_logits, edge_logits, h

    def forward_sequence(self, graph_seq, track_id_maps):
        """
        graph_seq: list of Data objects, one per frame in the temporal window
        track_id_maps: list of lists — track_ids[i][n] = track id of node n in frame i

        Returns per-frame node logits (spatial-only) plus a temporally-refined
        node embedding for tracks present across the window (used for a
        temporally-smoothed FP-suppression decision).
        """
        frame_outputs = []
        embeddings_by_track = {}

        for t, data in enumerate(graph_seq):
            node_logits, edge_logits, h = self.forward_frame(data)
            frame_outputs.append((node_logits, edge_logits))
            for local_idx, tid in enumerate(track_id_maps[t]):
                embeddings_by_track.setdefault(tid, []).append(h[local_idx])

        temporal_logits = {}
        if self.use_temporal:
            for tid, emb_list in embeddings_by_track.items():
                seq = torch.stack(emb_list).unsqueeze(0)  # [1, seq_len, hidden]
                out = self.temporal_agg(seq)
                final_state = out[:, -1, :]  # last time step summary
                temporal_logits[tid] = self.node_classifier(final_state.squeeze(0))

        return frame_outputs, temporal_logits


if __name__ == "__main__":
    # quick shape sanity check with dummy data
    from torch_geometric.data import Data

    node_dim = 7 + 10 + 512  # matches graph_utils.py feature layout
    x = torch.randn(6, node_dim)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    edge_attr = torch.randn(4, 4)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    model = STGCN(node_in_dim=node_dim)
    node_logits, edge_logits, h = model.forward_frame(data)
    print("node_logits:", node_logits.shape)
    print("edge_logits:", edge_logits.shape)
    print("node embeddings:", h.shape)
