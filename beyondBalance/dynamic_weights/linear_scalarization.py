from typing import Union, List
import torch
from beyondBalance.dynamic_weights.weight_method import WeightMethod


class LinearScalarization(WeightMethod):
    """
    A linear scalarization method for multi-task learning.

    This class implements a linear scalarization approach to combine multiple task losses into a single loss.
    The combined loss is computed as a weighted sum of individual task losses:

        L = sum_j (w_j * l_j)

    where:
        - `l_j` is the loss for task `j`
        - `w_j` is the weight for task `j`

    If `task_weights` is not provided, the class defaults to using identical weights (i.e., all tasks are weighted equally).

    Attributes:
        n_tasks (int): The number of tasks.
        task_weights (torch.Tensor): A tensor of weights for each task, used to scale the individual task losses.
        device (torch.device): The device (CPU/GPU) where the weights and losses are stored.
    """

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        task_weights: Union[List[float], torch.Tensor] = None,
    ):
        super().__init__(n_tasks, device=device)
        """
        Initializes the LinearScalarization object.
    
        This constructor:
            - Calls the parent class constructor to set up the number of tasks and device.
            - Initializes task weights to ones if not provided.
            - Converts task weights to a PyTorch tensor if they are provided as a list.
            - Validates that the length of task weights matches the number of tasks.
            - Moves the task weights tensor to the specified device.
    
        Parameters:
            n_tasks (int): The number of tasks.
            device (torch.device): The device (MPU/CPU/GPU) where the weights and losses will be stored.
            task_weights (Union[List[float], torch.Tensor, None], optional): The weights for each task.
                If None, all tasks are weighted equally (default: None).
    
        Raises:
            AssertionError: If the length of `task_weights` does not match `n_tasks`.
        Returns:
            None
        """

        if task_weights is None:
            task_weights = torch.ones((n_tasks,))
        if not isinstance(task_weights, torch.Tensor):
            task_weights = torch.tensor(task_weights)
        assert len(task_weights) == n_tasks
        self.task_weights = task_weights.to(device)

    def get_weighted_loss(self, losses, **kwargs):
        """
        Computes the weighted sum of task losses.

        The combined loss is computed as:
            L = sum_j (w_j * l_j)
        where `l_j` is the loss for task `j` and `w_j` is the weight for task `j`.

        Parameters:
            losses (torch.Tensor): A tensor of shape (batch_size, n_tasks) containing the losses for each task.
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            torch.Tensor: The scalarized loss, averaged over the batch.
        """
        loss = torch.mean(losses * self.task_weights)
        return loss
