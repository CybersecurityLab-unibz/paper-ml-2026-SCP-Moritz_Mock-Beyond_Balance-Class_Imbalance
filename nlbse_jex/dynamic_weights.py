import torch
import torch.nn.functional as F
from typing import Union
from typing import Dict, List, Tuple, Union
from abc import abstractmethod


class WeightMethod:
    def __init__(self, n_tasks: int, device: torch.device):
        super().__init__()
        self.n_tasks = n_tasks
        self.device = device

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        **kwargs,
    ):
        pass

    @abstractmethod
    def _get_weight(self, **kwargs):
        pass

    def update(self, **kwargs):
        pass

    def backward(
        self,
        losses: torch.Tensor,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """

        Parameters
        ----------
        losses :
        shared_parameters :
        task_specific_parameters :
        last_shared_parameters : parameters of last shared layer/block
        representation : shared representation
        kwargs :

        Returns
        -------
        Loss, extra outputs
        """
        loss = self.get_weighted_loss(
            losses=losses,
            **kwargs,
        )

        loss.backward()
        return loss

    def __call__(
        self,
        losses: torch.Tensor,
        **kwargs,
    ):
        return self.backward(
            losses=losses,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """return learnable parameters"""
        return []


class LinearScalarization(WeightMethod):
    """Linear scalarization baseline L = sum_j w_j * l_j where l_j is the loss for task j and w_h"""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        loss = torch.sum(losses * self.task_weights)
        return loss


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
        gamma: float = 0.01,  # the regularization coefficient
        w_lr: float = 0.025,  # the learning rate of the task logits
        max_norm: float = 1.0,  # the maximum gradient norm
    ):
        self.min_losses = torch.zeros(n_tasks, device=device)
        self.w = torch.tensor([0.0] * n_tasks, device=device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=w_lr, weight_decay=gamma)
        self.max_norm = max_norm
        self.n_tasks = n_tasks
        self.device = device

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


##############################################################
# coda adapted from https://github.com/codefuse-ai/MFTCoder/ #
# original paper https://arxiv.org/pdf/2410.06741            #
##############################################################


class COBA(WeightMethod):
    """
    Convergence Balancer
    """

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        coba_warmup_steps=100,
        coba_history_length=200,
        coba_tau=5,
        coba_update_interval=1,
        coba_sample_valid_num=1,
    ):
        self.coba_warmup_steps = coba_warmup_steps
        self.coba_history_length = coba_history_length
        self.coba_tau = coba_tau
        self.coba_update_interval = coba_update_interval
        self.coba_sample_valid_num = coba_sample_valid_num
        self.history_task_valid_loss = torch.empty((n_tasks, 0), device=device)
        self.w = (torch.ones(n_tasks, device=device),)
        self.n_tasks = n_tasks
        self.device = device
        self.completed_steps = 0
        self.minimum_weight = 1 / (n_tasks * 10)

    def compute_per_task_weight(self):
        task_num = self.n_tasks
        task_slope_fitting = torch.ones(
            task_num, dtype=torch.float64, device=self.device
        )
        start_step = max(
            0,
            self.completed_steps // self.coba_update_interval
            - self.coba_history_length,
        )
        history_steps = torch.arange(
            start_step, self.completed_steps, 1, device=self.device
        )
        for i in range(task_num):
            per_task_history_valid_loss = self.history_task_valid_loss[i][
                -len(history_steps) :
            ]
            task_slope_fitting[i] = self.fit_window_slope(
                history_steps, per_task_history_valid_loss, type="slope"
            )
        history_total_valid_loss, index = torch.max(
            self.history_task_valid_loss[:, -len(history_steps) :], dim=0
        )
        total_slope_fitting = self.fit_window_slope(
            history_steps, history_total_valid_loss, type="slope"
        )
        if self.completed_steps == self.coba_warmup_steps:
            self.per_task_slope_list = task_slope_fitting.unsqueeze(1)
            self.total_slope_list = total_slope_fitting.unsqueeze(0)
        else:
            self.per_task_slope_list = torch.cat(
                (self.per_task_slope_list, task_slope_fitting.unsqueeze(1)), dim=-1
            )
            self.total_slope_list = torch.cat(
                (self.total_slope_list, total_slope_fitting.unsqueeze(0)), dim=0
            )

        # Relative Convergence Score
        normalize_task_slope = (
            task_num * task_slope_fitting / task_slope_fitting.abs().sum()
        )
        rcs = F.softmax(normalize_task_slope, dim=-1)

        # Absolute Convergence Score
        history_per_task_slope_list = self.per_task_slope_list[:, start_step:]
        reverse_normailize_iter_slope = (
            -len(history_per_task_slope_list[0])
            * history_per_task_slope_list
            / history_per_task_slope_list.abs().sum(dim=-1, keepdim=True)
        )

        flatten_rn_iter_slope = reverse_normailize_iter_slope.T.reshape(-1)
        current_step_rn_slope = flatten_rn_iter_slope[-task_num:]
        acs = F.softmax(current_step_rn_slope, dim=-1)

        # Divergence Factor
        normalize_total_iter_slope = (
            -len(self.total_slope_list)
            * self.total_slope_list
            / self.total_slope_list.abs().sum()
        )
        divergence_factor = F.softmax(
            normalize_total_iter_slope * self.coba_tau, dim=-1
        )[-1] * len(self.total_slope_list)

        weight_logits = divergence_factor * rcs + (1 - divergence_factor) * acs
        per_task_weight = F.softmax(weight_logits * task_num, dim=-1)

        if len((per_task_weight < self.minimum_weight).nonzero().squeeze(0)) > 0:
            per_task_weight = per_task_weight * (1 - self.minimum_weight * task_num)
            per_task_weight += self.minimum_weight

        return per_task_weight

    def fit_window_slope(self, x, y, type="slope"):

        y = y[y != 0]
        x = x[: len(y)]

        nonzero_index = torch.squeeze(torch.nonzero(y), dim=1).to(self.device)
        y = torch.index_select(y, 0, nonzero_index)
        x = torch.index_select(x, 0, nonzero_index)

        ws = torch.flip(1 ** torch.arange(len(y)), dims=[0])
        ws = ws.double().to(self.device)

        if len(y) >= 2:
            if type == "slope":
                X = torch.stack(
                    (x, torch.ones_like(x, dtype=torch.float64, device=self.device))
                ).T
                X = X.double()
            else:
                X = torch.stack(
                    (
                        x**2,
                        x,
                        torch.ones_like(x, dtype=torch.float64, device=self.device),
                    )
                ).T

            w = torch.linalg.solve(X.T @ (ws[:, None] * X), X.T @ (ws * y))

            result = w[0]
        else:
            result = 0.0

        return result

    def _get_weight(self):
        return self.w

    def get_weighted_loss(self, losses):
        if self.completed_steps >= self.coba_warmup_steps:
            self.w = self.compute_per_task_weight()

        if isinstance(self.w, tuple):
            self.w = self.w[0]

        weighted_losses = losses * self.w
        return weighted_losses.sum()

    def update(self, curr_loss):
        self.completed_steps += 1
        self.history_task_valid_loss = torch.cat(
            [
                self.history_task_valid_loss,
                curr_loss.unsqueeze(1),  # shape (n_tasks, 1)
            ],
            dim=1,
        )  # shape history ,task


class WeightMethods:
    def __init__(self, method: str, n_tasks: int, device: torch.device, **kwargs):
        """
        :param method:
        """
        assert method in list(METHODS.keys()), f"unknown method {method}."

        self.method = METHODS[method](n_tasks=n_tasks, device=device, **kwargs)

    def get_weighted_loss(self, losses, **kwargs):
        return self.method.get_weighted_loss(losses, **kwargs)

    def _get_weight(self, **kwargs):
        return self.method._get_weight(**kwargs)

    def update(self, **kwargs):
        return self.method.update(**kwargs)

    def backward(
        self, losses, **kwargs
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        return self.method.backward(losses, **kwargs)

    def __call__(self, losses, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self):
        return self.method.parameters()


METHODS = dict(FAMO=FAMO, COBA=COBA, ln=LinearScalarization)
