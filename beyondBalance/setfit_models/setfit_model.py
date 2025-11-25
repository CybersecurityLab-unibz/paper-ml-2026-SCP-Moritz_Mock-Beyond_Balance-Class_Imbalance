from typing import List, Union, Optional

import setfit
import torch
from torch import nn
from tqdm.asyncio import trange, tqdm

from beyondBalance.dynamic_weights import WeightMethods


class SetFitModel(setfit.SetFitModel):

    def __init__(
        self,
        *args,
        custom_loss_weight=None,
        weighted_loss_strategy="EW",
        num_classes=1,
        isDynamic=False,
        **kwargs,
    ):
        """
        Initializes a model class with support for static or dynamic weighted loss strategies.

        This constructor:
            - Passes all positional and keyword arguments to the superclass
            - Stores the loss strategy and whether it is dynamic
            - Initializes and moves the custom class weights tensor to the appropriate device
            - Stores the number of output classes

        Parameters:
            *args: Positional arguments passed to the parent class.
            custom_loss_weight (Iterable[float] or Tensor): Class-specific weights for the loss function.
            weighted_loss_strategy (str or Callable): The selected loss weighting strategy (e.g., "EW", "ICF", or a callable).
            num_classes (int): The number of output classes.
            isDynamic (bool): Whether the model uses a dynamic weighting strategy.
            **kwargs: Additional keyword arguments passed to the parent class.

        Returns:
            None
        """

        super().__init__(*args, **kwargs)
        self.weighted_loss_strategy = weighted_loss_strategy
        self.custom_loss_weight = torch.tensor(
            custom_loss_weight, device=self.device
        )  # take device from parent class
        self.num_classes = num_classes
        self.isDynamic = isDynamic

    def loss_selection(self):
        """
        Selects and returns the appropriate loss function based on whether dynamic weighting is used.

        This method:
            - Returns `BCEWithLogitsLoss` with no reduction if dynamic loss weighting is enabled (`isDynamic` is True)
            - Returns `CrossEntropyLoss` with predefined class weights if static weighting is used

        Returns:
            torch.nn.Module: The selected loss function instance.
        """

        if self.isDynamic:
            return torch.nn.BCEWithLogitsLoss(reduction="none")
        return nn.CrossEntropyLoss(weight=self.custom_loss_weight)

    def fit(
        self,
        x_train: List[str],
        y_train: Union[List[int], List[List[int]]],
        num_epochs: int,
        batch_size: Optional[int] = None,
        body_learning_rate: Optional[float] = None,
        head_learning_rate: Optional[float] = None,
        end_to_end: bool = False,
        l2_weight: Optional[float] = None,
        max_length: Optional[int] = None,
        show_progress_bar: bool = True,
    ) -> None:
        """Train the classifier head, only used if a differentiable PyTorch head is used.

        Args:
            x_train (`List[str]`): A list of training sentences.
            y_train (`Union[List[int], List[List[int]]]`): A list of labels corresponding to the training sentences.
            num_epochs (`int`): The number of epochs to train for.
            batch_size (`int`, *optional*): The batch size to use.
            body_learning_rate (`float`, *optional*): The learning rate for the `SentenceTransformer` body
                in the `AdamW` optimizer. Disregarded if `end_to_end=False`.
            head_learning_rate (`float`, *optional*): The learning rate for the differentiable torch head
                in the `AdamW` optimizer.
            end_to_end (`bool`, defaults to `False`): If True, train the entire model end-to-end.
                Otherwise, freeze the `SentenceTransformer` body and only train the head.
            l2_weight (`float`, *optional*): The l2 weight for both the model body and head
                in the `AdamW` optimizer.
            max_length (`int`, *optional*): The maximum token length a tokenizer can generate. If not provided,
                the maximum length for the `SentenceTransformer` body is used.
            show_progress_bar (`bool`, defaults to `True`): Whether to display a progress bar for the training
                epochs and iterations.
        """
        self.model_body.train()
        self.model_head.train()
        if not end_to_end:
            self.freeze("body")

        dataloader = self._prepare_dataloader(x_train, y_train, batch_size, max_length)
        criterion = torch.nn.BCEWithLogitsLoss(
            reduction="none"
        )  # changed from self.model_head.get_loss_fn()
        optimizer = self._prepare_optimizer(
            head_learning_rate, body_learning_rate, l2_weight
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        dynamic_weight_loss = WeightMethods(
            method=self.weighted_loss_strategy,
            n_tasks=self.num_classes,
            device=self.device,
            isDynamic=self.isDynamic,
            task_weights=self.custom_loss_weight,
        )

        for epoch_idx in trange(
            num_epochs, desc="Epoch", disable=not show_progress_bar
        ):
            for batch in tqdm(
                dataloader,
                desc="Iteration",
                disable=not show_progress_bar,
                leave=False,
            ):
                features, labels = batch
                optimizer.zero_grad()

                # to model's device
                features = {k: v.to(self.device) for k, v in features.items()}
                labels = labels.to(self.device)

                outputs = self.model_body(features)
                if self.normalize_embeddings:
                    outputs["sentence_embedding"] = nn.functional.normalize(
                        outputs["sentence_embedding"], p=2, dim=1
                    )
                outputs = self.model_head(outputs)
                logits: torch.Tensor = outputs["logits"]
                loss: torch.Tensor = criterion(logits, labels)
                dynamic_weight_loss.backward(loss.mean(0))

                optimizer.step()

                with torch.no_grad():
                    dynamic_weight_loss.update(
                        curr_loss=criterion(logits, labels).mean(0)
                    )

            scheduler.step()

        if not end_to_end:
            self.unfreeze("body")
