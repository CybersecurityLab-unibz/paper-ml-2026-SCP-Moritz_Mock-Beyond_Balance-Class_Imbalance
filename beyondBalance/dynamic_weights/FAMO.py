##########################################################
# original code from https://github.com/Cranial-XIX/FAMO #
# original paper https://arxiv.org/pdf/2306.03792        #
##########################################################

import torch
from torch.nn import functional as F
from typing import List, Union
from nlbse_jex.dynamic_weights.weight_method import WeightMethod


class FAMO(WeightMethod):
    """Fast Adaptive Multitask Optimization (FAMO).

    This class implements the FAMO algorithm for dynamic weight adjustment in multi-task learning.
    FAMO adapts task weights based on the relative improvement of task losses, using a softmax-based
    weighting scheme with gradient-based updates.

    ##########################################################
    #    original paper https://arxiv.org/pdf/2306.03792     #
    ##########################################################

    Attributes:
        min_losses (torch.Tensor): Tensor storing the minimum observed losses for each task.
        w (torch.Tensor): Learnable weight parameters for each task (requires gradient).
        w_opt (torch.optim.Optimizer): Optimizer for the task weights.
        max_norm (float): Maximum gradient norm for clipping.
        prev_loss (torch.Tensor): Tensor storing the previous losses for each task.
        gamma (float): Regularization coefficient for the weight optimizer.
        w_lr (float): Learning rate for the task weight optimizer.
    """

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 0.01,  # the regularization coefficient
        w_lr: float = 0.025,  # the learning rate of the task logits
        max_norm: float = 1.0,  # the maximum gradient norm
        task_weights: Union[List[float], torch.Tensor] = None,
    ):
        """
        Initializes the FAMO weight method.

        Parameters:
            n_tasks (int): The number of tasks.
            device (torch.device): The device (CPU/GPU) where tensors will be stored.
            gamma (float): The regularization coefficient for the weight optimizer (default: 0.01).
            w_lr (float): The learning rate for the task weight optimizer (default: 0.025).
            max_norm (float): The maximum gradient norm for clipping (default: 1.0).
            task_weights (Union[List[float], torch.Tensor, None]): Initial task weights.
                If None, weights are initialized to zeros (default: None).
        """
        super().__init__(n_tasks, device)
        self.min_losses = torch.zeros(n_tasks, device=self.device)
        self.w = torch.tensor([0.0] * n_tasks, device=self.device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm

    def set_min_losses(self, losses):
        """
        Sets the minimum observed losses for each task.

        Parameters:
            losses (torch.Tensor): A tensor of minimum losses for each task.
        """
        self.min_losses = losses

    def _get_weight(self):
        """
        Computes the softmax of the task weights.

        Returns:
            torch.Tensor: The softmax-normalized task weights.
        """
        return F.softmax(self.w, -1)

    def get_weighted_loss(self, losses):
        """
        Computes the FAMO-weighted loss.

        The weighted loss is computed as:
            loss = sum((log(D) * z) / c)
        where:
            - D = losses - min_losses + ε
            - z = softmax(w)
            - c = sum(z / D)

        Parameters:
            losses (torch.Tensor): A tensor of shape (n_tasks,) containing the current task losses.

        Returns:
            torch.Tensor: The computed FAMO-weighted loss.
        """
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss

    def update(self, curr_loss):
        """
        Updates the task weights based on the current and previous losses.

        The update rule is based on the gradient of the softmax weights with respect to the
        log difference between previous and current losses.

        Parameters:
            curr_loss (torch.Tensor): A tensor of shape (n_tasks,) containing the current task losses.
        """
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - (
            curr_loss - self.min_losses + 1e-8
        ).log()
        with torch.enable_grad():
            d = torch.autograd.grad(
                F.softmax(self.w, -1), self.w, grad_outputs=delta.detach()
            )[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()
