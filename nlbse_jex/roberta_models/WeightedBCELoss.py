import torch
from torch import nn as nn

from nlbse_jex.utils import get_device


class WeightedBCELoss(nn.Module):
    def __init__(self,custom_loss_weight):
        super(WeightedBCELoss, self).__init__()
        self.device = get_device()
        self.custom_loss_weight = torch.tensor(custom_loss_weight, device=self.device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, labels,update_loss_weight=None,scalar = False):
        if update_loss_weight is not None:
            self.custom_loss_weight = update_loss_weight
        logits = logits.to(self.device)
        labels = labels.to(self.device)

        loss = self.bce(logits, labels)  # Calculate unweighted BCE loss

        weighted_loss = loss * self.custom_loss_weight  # Apply weights to each label's loss

        if scalar:
            return weighted_loss.mean()
        return weighted_loss
