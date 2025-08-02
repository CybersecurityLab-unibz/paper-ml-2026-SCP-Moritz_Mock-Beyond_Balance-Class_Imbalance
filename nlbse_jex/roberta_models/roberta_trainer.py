import torch
from transformers import Trainer

from nlbse_jex.utils import get_device


class RoBERTaTrainer(Trainer):
    def __init__(self, *args, custom_loss=None,dynamic_weight_loss=None,num_classes=None, isDynamic=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss = custom_loss
        self.dynamic_weight_loss = dynamic_weight_loss
        self.isDynamic = isDynamic
        self.num_classes=num_classes
        self.update_loss_weight=None
        self.device = get_device()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float().to(self.device)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.custom_loss(logits, labels)

        if self.isDynamic:
            with torch.no_grad():
                self.dynamic_weight_loss.get_weighted_loss(loss.mean(0))
                update_loss_weight = self.dynamic_weight_loss._get_weight()

        loss = self.custom_loss(logits, labels,update_loss_weight,scalar= True)

        if self.isDynamic:
            with torch.no_grad():
                new_loss = self.custom_loss(logits, labels,update_loss_weight)
                self.dynamic_weight_loss.update(curr_loss=new_loss.mean(0))

        return (loss, outputs) if return_outputs else loss
