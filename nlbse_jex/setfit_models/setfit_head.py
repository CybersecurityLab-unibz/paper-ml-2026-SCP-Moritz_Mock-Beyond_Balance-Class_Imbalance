import setfit
from torch import nn


class SetFitHead(setfit.SetFitHead):
    def __init__(self, n_layer=2, **kwargs) -> None:
        super().__init__(**kwargs)

        if self.in_features is not None:
            assert (
                n_layer >= 1
            ), "Number of layers for setfit cannot be below 1; increase 'n_layer' to 1 or more. Exiting..."

            layers = []
            for _ in range(n_layer):
                layers.append(
                    nn.Linear(self.in_features, self.in_features, bias=self.bias)
                )
                layers.append(nn.ReLU())

            layers.append(
                nn.Linear(self.in_features, self.out_features, bias=self.bias)
            )

            self.linear = nn.Sequential(*layers)
