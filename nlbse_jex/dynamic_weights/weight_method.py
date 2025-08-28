from abc import abstractmethod
from typing import Tuple, Union, List

import torch


class WeightMethod:
    """Abstract base class for weight methods in multi-task learning.

    This class defines the interface for methods that compute weighted losses for multi-task learning.
    Subclasses must implement the `get_weighted_loss` and `_get_weight` methods to define how task weights
    are computed and applied to the losses.

    Attributes:
        n_tasks (int): The number of tasks.
        device (torch.device): The device (CPU/GPU) where the weights and losses are stored.
    """

    def __init__(self, n_tasks: int, device: torch.device):
        super().__init__()
        """
        Initializes the WeightMethod object.

        Parameters:
            n_tasks (int): The number of tasks.
            device (torch.device): The device (CPU/GPU) where the weights and losses will be stored.

        Return:
            None
        """
        self.n_tasks = n_tasks
        self.device = device

    @abstractmethod
    def get_weighted_loss(
        self,
        losses: torch.Tensor,
        **kwargs,
    ):
        """
        Computes the weighted loss for multi-task learning.

        This method should be implemented by subclasses to define how the weighted loss is computed.

        Parameters:
            losses (torch.Tensor): A tensor of shape (batch_size, n_tasks) containing the losses for each task.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            torch.Tensor: The computed weighted loss.
        """
        pass

    @abstractmethod
    def _get_weight(self, **kwargs):
        """
        Computes the weights for each task.

        This method should be implemented by subclasses to define how task weights are computed.

        Parameters:
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Union[torch.Tensor, None]: A tensor of weights for each task, or None if not applicable.
        """
        pass

    def update(self, **kwargs):
        """
        Updates the weight method (optional).

        This method can be overridden by subclasses to update the weights or other parameters.

        Parameters:
            **kwargs (Any): Additional keyword arguments.

        Returns:
            None
        """
        pass

    def backward(
        self,
        losses: torch.Tensor,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, None], Union[dict, None]]:
        """
        Computes the weighted loss and performs backpropagation.

        This method computes the weighted loss using `get_weighted_loss`, performs backpropagation,
        and returns the loss along with any additional outputs.

        Parameters:
            losses (torch.Tensor): A tensor of shape (batch_size, n_tasks) containing the losses for each task.
            **kwargs (Any): Additional keyword arguments, such as:
                - shared_parameters: Parameters of shared layers/blocks.
                - task_specific_parameters: Parameters of task-specific layers/blocks.
                - last_shared_parameters: Parameters of the last shared layer/block.

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
                - The computed loss..
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
        """
        Calls the backward method to compute the weighted loss and perform backpropagation.

        This method is a convenience wrapper around `backward`.

        Parameters:
            losses (torch.Tensor): A tensor of shape (batch_size, n_tasks) containing the losses for each task.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
                - The computed loss.
        """
        return self.backward(
            losses=losses,
            **kwargs,
        )

    def parameters(self) -> List[torch.Tensor]:
        """
        Returns the learnable parameters of the weight method.

        This method should be overridden by subclasses to return any learnable parameters.

        Returns:
            List[torch.Tensor]: A list of learnable parameters. Defaults to an empty list.
        """
        return []
