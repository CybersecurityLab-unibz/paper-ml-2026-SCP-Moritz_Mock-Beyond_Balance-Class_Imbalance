##############################################################
# coda adapted from https://github.com/codefuse-ai/MFTCoder/ #
# original paper https://arxiv.org/pdf/2410.06741            #
##############################################################


import torch
from torch.nn import functional as F
from typing import List, Union
from beyondBalance.dynamic_weights.weight_method import WeightMethod


class COBA(WeightMethod):
    """Convergence Balancer (COBA) for multi-task learning.

    This class implements the COBA algorithm for dynamic weight adjustment in multi-task learning.
    COBA balances task convergence by computing relative and absolute convergence scores and
    adjusting task weights accordingly.

    ############################################################
    #     original paper https://arxiv.org/pdf/2410.06741      #
    ############################################################

    Attributes:
        coba_warmup_steps (int): Number of warmup steps before weight adjustment begins.
        coba_history_length (int): Length of history to consider for slope fitting.
        coba_tau (float): Temperature parameter for divergence factor calculation.
        coba_update_interval (int): Interval for updating weights.
        coba_sample_valid_num (int): Number of valid samples to consider.
        history_task_valid_loss (torch.Tensor): History of validation losses for each task.
        w (torch.Tensor): Current task weights.
        n_tasks (int): Number of tasks.
        device (torch.device): Device where tensors are stored.
        completed_steps (int): Number of completed training steps.
        minimum_weight (float): Minimum weight for any task.
        per_task_slope_list (torch.Tensor): History of per-task slopes.
        total_slope_list (torch.Tensor): History of total slopes.
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
        task_weights: Union[List[float], torch.Tensor] = None,
    ):
        """
        Initializes the COBA weight method.

        Parameters:
            n_tasks (int): Number of tasks.
            device (torch.device): Device where tensors will be stored.
            coba_warmup_steps (int): Number of warmup steps before weight adjustment begins (default: 100).
            coba_history_length (int): Length of history to consider for slope fitting (default: 200).
            coba_tau (float): Temperature parameter for divergence factor calculation (default: 5).
            coba_update_interval (int): Interval for updating weights (default: 1).
            coba_sample_valid_num (int): Number of valid samples to consider (default: 1).
            task_weights (Union[List[float], torch.Tensor, None]): Initial task weights.
                If None, weights are initialized to ones (default: None).
        """
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
        """
        Computes the per-task weights based on convergence scores.

        This method calculates relative convergence scores (RCS), absolute convergence scores (ACS),
        and divergence factors to determine the appropriate weights for each task.

        Returns:
            torch.Tensor: Computed weights for each task.
        """
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
        """
        Fits a slope to the given data points.

        Parameters:
            x (torch.Tensor): X-coordinates (steps).
            y (torch.Tensor): Y-coordinates (losses).
            type (str): Type of fitting ("slope" for linear, others for quadratic) (default: "slope").

        Returns:
            float: The fitted slope.
        """

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
        """
        Gets the current task weights.

        Returns:
            torch.Tensor: Current task weights.
        """
        return self.w

    def get_weighted_loss(self, losses):
        """
        Computes the weighted loss using COBA weights.

        Parameters:
            losses (torch.Tensor): A tensor of shape (n_tasks,) containing the current task losses.

        Returns:
            torch.Tensor: The weighted sum of losses.
        """
        if self.completed_steps >= self.coba_warmup_steps:
            self.w = self.compute_per_task_weight()

        if isinstance(self.w, tuple):
            self.w = self.w[0]

        weighted_losses = losses * self.w
        return weighted_losses.sum()

    def update(self, curr_loss):
        """
        Updates the COBA state with the current losses.

        Parameters:
            curr_loss (torch.Tensor): A tensor of shape (n_tasks,) containing the current task losses.
        """
        self.completed_steps += 1
        self.history_task_valid_loss = torch.cat(
            [
                self.history_task_valid_loss,
                curr_loss.unsqueeze(1),  # shape (n_tasks, 1)
            ],
            dim=1,
        )  # shape history ,task
