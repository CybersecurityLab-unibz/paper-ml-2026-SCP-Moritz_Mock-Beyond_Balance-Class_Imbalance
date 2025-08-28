import warnings
from typing import Optional, Union, Dict, Any

import setfit


class SetFitTrainer(setfit.Trainer):
    def __init__(self, freeze_embeddings=False, **kwargs):
        """
        Initializes the model with support for optionally freezing embeddings during training.

        Parameters:
            freeze_embeddings (bool): If True, prevents updates to embedding weights during training.
            **kwargs: Additional keyword arguments passed to the parent class (setfit.Trainer).

        Returns:
            None
        """

        super().__init__(**kwargs)
        self.freeze_embeddings = freeze_embeddings

    def train(
        self,
        args: Optional[setfit.TrainingArguments] = None,
        trial: Optional[Union["optuna.Trial", Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:
        """
        Main training entry point.

        Args:
            args (`TrainingArguments`, *optional*):
                Temporarily change the training arguments for this training call.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        if len(kwargs):
            warnings.warn(
                f"`{self.__class__.__name__}.train` does not accept keyword arguments anymore. "
                f"Please provide training arguments via a `TrainingArguments` instance to the `{self.__class__.__name__}` "
                f"initialisation or the `{self.__class__.__name__}.train` method.",
                DeprecationWarning,
                stacklevel=2,
            )

        if trial:  # Trial and model initialization
            self._hp_search_setup(
                trial
            )  # sets trainer parameters and initializes model

        args = args or self.args or setfit.TrainingArguments()

        if self.train_dataset is None:
            raise ValueError(
                f"Training requires a `train_dataset` given to the `{self.__class__.__name__}` initialization."
            )

        train_parameters = self.dataset_to_parameters(self.train_dataset)

        # Added the following if statement
        if self.freeze_embeddings is False:
            full_parameters = (
                train_parameters + self.dataset_to_parameters(self.eval_dataset)
                if self.eval_dataset
                else train_parameters
            )

            self.train_embeddings(*full_parameters, args=args)

        self.train_classifier(*train_parameters, args=args)
