import torch
from transformers import Trainer

from nlbse_jex.utils import get_device


class RoBERTaTrainer(Trainer):
    def __init__(
        self,
        *args,
        custom_loss=None,
        dynamic_weight_loss=None,
        num_classes=None,
        isDynamic=False,
        **kwargs,
    ):
        """
        Initializes the RoBERTaTrainer with support for custom and dynamic loss strategies.

        This constructor:
            - Passes standard arguments to the base trainer class
            - Stores references to:
                - `custom_loss`: the loss function to be used during training
                - `dynamic_weight_loss`: optional dynamic loss weighting strategy
                - `isDynamic`: flag indicating whether dynamic loss adjustment is enabled
                - `num_classes`: number of target classes for classification
            - Initializes `update_loss_weight` to None
            - Sets the computation device (CPU, CUDA, or MPS)

        Parameters:
            *args: Positional arguments passed to the base trainer.
            custom_loss (Callable): A custom loss function for multi-label classification.
            dynamic_weight_loss (Optional[Callable]): A dynamic weighting strategy object (e.g., FAMO or COBA).
            num_classes (Optional[int]): The number of output classes.
            isDynamic (bool): Whether dynamic class weighting is enabled.
            **kwargs: Additional keyword arguments passed to the base trainer.

        Returns:
            None
        """

        super().__init__(*args, **kwargs)
        self.custom_loss = custom_loss
        self.dynamic_weight_loss = dynamic_weight_loss
        self.isDynamic = isDynamic
        self.num_classes = num_classes
        self.update_loss_weight = None
        self.device = get_device()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Computes the custom loss for a batch of inputs, with optional support for dynamic weighting strategies.

        This method:
            - Extracts and moves input tensors and labels to the appropriate device
            - Computes logits by forwarding inputs through the model
            - Calculates the initial loss using the custom loss function
            - If dynamic weighting is enabled:
                - Updates the dynamic weights based on the current loss
                - Recomputes the scalar loss using updated weights
                - Updates the dynamic weighting strategy's internal state
            - Returns the loss and optionally the model outputs

        Parameters:
            model (nn.Module): The model used to compute logits.
            inputs (dict): A dictionary of batched input tensors, including 'labels'.
            return_outputs (bool): If True, also returns the model outputs.
            num_items_in_batch (int or None): (Unused) Can be used for future extension if needed.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, Any]: The computed loss, and optionally the model outputs.
        """

        labels = inputs.pop("labels").float().to(self.device)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.custom_loss(logits, labels)

        if self.isDynamic:
            with torch.no_grad():
                self.dynamic_weight_loss.get_weighted_loss(loss.mean(0))
                update_loss_weight = self.dynamic_weight_loss._get_weight()

        loss = self.custom_loss(logits, labels, update_loss_weight, scalar=True)

        if self.isDynamic:
            with torch.no_grad():
                new_loss = self.custom_loss(logits, labels, update_loss_weight)
                self.dynamic_weight_loss.update(curr_loss=new_loss.mean(0))

        return (loss, outputs) if return_outputs else loss
