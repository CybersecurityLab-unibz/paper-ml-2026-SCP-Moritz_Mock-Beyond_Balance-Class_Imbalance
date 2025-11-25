import torch
from typing import Dict, Tuple, Union

from beyondBalance.dynamic_weights.COBA import COBA
from beyondBalance.dynamic_weights.FAMO import FAMO
from beyondBalance.dynamic_weights.linear_scalarization import LinearScalarization


class WeightMethods:
    """A factory class for weight methods in multi-task learning.

    This class provides a unified interface for different weight methods (e.g., COBA, FAMO, LinearScalarization).
    It dynamically selects the appropriate weight method based on the provided configuration.

    Attributes:
        method (WeightMethod): An instance of the selected weight method.
    """

    def __init__(
        self, method: str, n_tasks: int, device: torch.device, isDynamic: bool, **kwargs
    ):
        """
        Initializes the WeightMethods object with the specified weight method.

        Parameters:
            method (str): The name of the weight method to use. Supported methods are "FAMO", "COBA", and "ln".
            n_tasks (int): The number of tasks.
            device (torch.device): The device (CPU/GPU) where the weights and losses will be stored.
            isDynamic (bool): If False, the method defaults to "ln" (LinearScalarization).
            **kwargs (Any): Additional keyword arguments passed to the weight method constructor.

        Raises:
            KeyError: If the specified method is not found in the METHODS dictionary.
        """
        self.method = METHODS[method if isDynamic else "ln"](
            n_tasks=n_tasks, device=device, **kwargs
        )

    def get_weighted_loss(self, losses, **kwargs):
        """
        Computes the weighted loss for multi-task learning.

        Parameters:
            losses (torch.Tensor): A tensor of shape (batch_size, n_tasks) containing the losses for each task.
            **kwargs (Any): Additional keyword arguments passed to the weight method.

        Returns:
            torch.Tensor: The computed weighted loss.
        """
        return self.method.get_weighted_loss(losses, **kwargs)

    def _get_weight(self, **kwargs):
        """
        Computes the weights for each task.

        Parameters:
            **kwargs (Any): Additional keyword arguments passed to the weight method.

        Returns:
            Union[torch.Tensor, None]: A tensor of weights for each task, or None if not applicable.
        """
        return self.method._get_weight(**kwargs)

    def update(self, **kwargs):
        """
        Updates the weight method (optional).

        Parameters:
            **kwargs (Any): Additional keyword arguments passed to the weight method.
        """
        self.method.update(**kwargs)

    def backward(
        self, losses, **kwargs
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        """
        Computes the weighted loss and performs backpropagation.

        Parameters:
            losses (torch.Tensor): A tensor of shape (batch_size, n_tasks) containing the losses for each task.
            **kwargs (Any): Additional keyword arguments passed to the weight method.

        Returns:
            Tuple[Union[torch.Tensor, None], Union[Dict[str, Any], None]]:
                - The computed loss.
                - A dictionary of additional outputs, or None if no additional outputs are provided.
        """
        return self.method.backward(losses, **kwargs)

    def __call__(self, losses, **kwargs):
        """
        Calls the backward method to compute the weighted loss and perform backpropagation.

        Parameters:
            losses (torch.Tensor): A tensor of shape (batch_size, n_tasks) containing the losses for each task.
            **kwargs (Any): Additional keyword arguments passed to the weight method.

        Returns:
            Tuple[Union[torch.Tensor, None], Union[Dict[str, Any], None]]:
                - The computed loss.
                - A dictionary of additional outputs, or None if no additional outputs are provided.
        """
        return self.backward(losses, **kwargs)

    def parameters(self):
        """
        Returns the learnable parameters of the weight method.

        Returns:
            list: A list of learnable parameters.
        """
        return self.method.parameters()


METHODS = dict(FAMO=FAMO, COBA=COBA, ln=LinearScalarization)
