import setfit
from torch import nn


class SetFitHead(setfit.SetFitHead):
    def __init__(self, n_layer=2, **kwargs) -> None:
        """
        Initializes a multi-layer feedforward classification head for SetFit models.

        This constructor:
            - Passes all keyword arguments to the superclass
            - Validates that the number of layers is at least 1
            - Constructs a sequence of fully connected layers with ReLU activations
              mapping from input features to output features
            - Appends a final linear layer projecting to the target output dimension

        Parameters:
            n_layer (int, optional): Number of hidden linear layers with ReLU activations
                before the final output layer. Must be >= 1. Default is 2.
            **kwargs: Additional keyword arguments passed to the parent class
                (e.g., `in_features`, `out_features`, `bias`, `device`).

        Returns:
            None
        """
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
