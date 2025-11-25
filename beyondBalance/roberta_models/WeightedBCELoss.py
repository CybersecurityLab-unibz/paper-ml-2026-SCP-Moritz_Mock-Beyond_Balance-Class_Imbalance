import torch
from torch import nn as nn

from nlbse_jex.utils import get_device


class WeightedBCELoss(nn.Module):
    def __init__(self, custom_loss_weight):
        """
        Initializes a custom weighted binary cross-entropy loss function for multi-label classification.

        This constructor:
            - Inherits from `nn.Module`
            - Stores the computation device
            - Converts and stores the provided class weights as a tensor on the correct device
            - Initializes `BCEWithLogitsLoss` with `reduction='none'` to allow manual weighting of each class

        Parameters:
            custom_loss_weight (Iterable[float] or Tensor): Class-specific weights to apply to the BCE loss.

        Returns:
            None
        """

        super(WeightedBCELoss, self).__init__()
        self.device = get_device()
        self.custom_loss_weight = torch.tensor(custom_loss_weight, device=self.device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, labels, update_loss_weight=None, scalar=False):
        """
        Computes a weighted binary cross-entropy (BCE) loss with optional support for dynamic weight updates.

        This method:
            - Updates internal class weights if `update_loss_weight` is provided
            - Moves logits and labels to the appropriate device
            - Computes element-wise BCE loss between logits and labels
            - Applies class-specific weights to the loss
            - Optionally returns a scalar mean of the loss for backpropagation

        Parameters:
            logits (Tensor): The raw model outputs (before sigmoid), shape (batch_size, num_classes).
            labels (Tensor): The ground truth labels, shape (batch_size, num_classes).
            update_loss_weight (Tensor, optional): A tensor of updated class weights.
            scalar (bool): If True, returns the mean loss; otherwise returns the weighted loss tensor.

        Returns:
            Tensor: The weighted loss, either per-sample or as a scalar mean.
        """

        if update_loss_weight is not None:
            self.custom_loss_weight = update_loss_weight
        logits = logits.to(self.device)
        labels = labels.to(self.device)

        loss = self.bce(logits, labels)  # Calculate unweighted BCE loss

        weighted_loss = (
            loss * self.custom_loss_weight
        )  # Apply weights to each label's loss

        if scalar:
            return weighted_loss.mean()
        return weighted_loss
