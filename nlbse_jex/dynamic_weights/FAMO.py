import torch
from torch.nn import functional as F

from nlbse_jex.dynamic_weights.weight_method import WeightMethod


##########################################################
# original code from https://github.com/Cranial-XIX/FAMO #
# original paper https://arxiv.org/pdf/2306.03792        #
##########################################################
class FAMO(WeightMethod):
    """
    Fast Adaptive Multitask Optimization.
    """
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        gamma: float = 0.01,   # the regularization coefficient
        w_lr: float = 0.025,   # the learning rate of the task logits
        max_norm: float = 1.0, # the maximum gradient norm
    ):
        super().__init__(n_tasks, device)
        self.min_losses = torch.zeros(n_tasks,device=self.device)
        self.w = torch.tensor([0.0] * n_tasks, device=self.device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm

    def set_min_losses(self, losses):
        self.min_losses = losses

    def _get_weight(self):
        return F.softmax(self.w, -1)

    def get_weighted_loss(self, losses):
        self.prev_loss = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        return loss

    def update(self, curr_loss):
        delta = (self.prev_loss - self.min_losses + 1e-8).log() - \
                (curr_loss      - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad()
        self.w.grad = d
        self.w_opt.step()
