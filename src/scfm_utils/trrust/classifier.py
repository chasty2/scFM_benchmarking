from __future__ import annotations

import torch.nn as nn


class TRRClassifierModel(nn.Module):
    """Bilinear classifier for TF → target gene regulation prediction.

    Learns a (n_classes, embsize, embsize) weight tensor via nn.Bilinear.
    Output logits of shape (batch, n_classes).
    """

    def __init__(self, embsize: int, n_classes: int = 3):
        super().__init__()
        self.bilinear = nn.Bilinear(embsize, embsize, n_classes)

    def forward(self, tf_emb, target_emb):
        return self.bilinear(tf_emb, target_emb)
