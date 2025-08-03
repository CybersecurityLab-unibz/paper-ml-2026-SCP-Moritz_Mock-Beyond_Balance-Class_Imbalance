from abc import abstractmethod
from typing import Tuple, Union, List

import torch


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
