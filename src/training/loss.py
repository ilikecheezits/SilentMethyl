import torch
import torch.nn as nn

class FocalLossWithLogits(nn.Module):
    """Handles classification mode-collapse in highly imbalanced genomic datasets."""
    def __init__(self, alpha=0.65, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        logits = logits.view(-1)
        targets = targets.view(-1)
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * ((1 - p_t) ** self.gamma)
        return torch.mean(focal_weight * bce_loss)