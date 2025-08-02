import torch
from typing import Dict, Tuple, Union

from nlbse_jex.dynamic_weights.COBA import COBA
from nlbse_jex.dynamic_weights.FAMO import FAMO
from nlbse_jex.dynamic_weights.linear_scalarization import LinearScalarization


class WeightMethods:
    def __init__(self, method: str, n_tasks: int, device: torch.device, **kwargs):
        """
        :param method:
        """
        assert method in list(METHODS.keys()), f"unknown method {method}."

        self.method = METHODS[method](n_tasks=n_tasks, device=device, **kwargs)

    def get_weighted_loss(self, losses, **kwargs):
        return self.method.get_weighted_loss(losses, **kwargs)
    
    def _get_weight(self,**kwargs):
        return self.method._get_weight(**kwargs)
    
    def update(self,**kwargs):
        return self.method.update(**kwargs)

    def backward(
        self, losses, **kwargs
    ) -> Tuple[Union[torch.Tensor, None], Union[Dict, None]]:
        return self.method.backward(losses, **kwargs)

    def __call__(self, losses, **kwargs):
        return self.backward(losses, **kwargs)

    def parameters(self):
        return self.method.parameters()
    


METHODS = dict(
    FAMO =FAMO,
    COBA =COBA,
    ln =LinearScalarization
)