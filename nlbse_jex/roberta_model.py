from transformers import AutoTokenizer
from nlbse_jex.data_handler import DataHandler
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding
import torch, torch.nn as nn
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class WeightedBCELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedBCELoss, self).__init__()
        self.weights = torch.tensor(weights, device=device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, labels):
        logits = logits.to(device)
        labels = labels.to(device)

        loss = self.bce(logits, labels)  # Calculate unweighted BCE loss

        weighted_loss = loss * self.weights  # Apply weights to each label's loss
        return weighted_loss.mean()


class CustomTrainer(Trainer):
    def __init__(self, *args, custom_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss = custom_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float().to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.custom_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics_roberta(eval_pred):
    # Unpack the predictions and labels
    logits, labels = eval_pred

    # Convert logits to a PyTorch tensor if they are in NumPy format
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    # Apply sigmoid to logits to get probabilities (for multilabel)
    probabilities = torch.sigmoid(logits)  # Shape: (num_samples, num_classes)

    # Convert probabilities to binary predictions based on a threshold
    threshold = 0.5
    predictions = (probabilities >= threshold).int()  # Binarize predictions

    # Ensure predictions and labels are on the same device (CPU)
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    ## call here
    #return compute_metrics(predictions, labels)

    # Check if predictions and labels are compatible
    num_classes = labels.shape[1]
    class_metrics = {}

    # Calculate TP, FP, TN, FN per class
    for class_idx in range(num_classes):
        tp = np.sum((predictions[:, class_idx] == 1) & (labels[:, class_idx] == 1))
        fp = np.sum((predictions[:, class_idx] == 1) & (labels[:, class_idx] == 0))
        tn = np.sum((predictions[:, class_idx] == 0) & (labels[:, class_idx] == 0))
        fn = np.sum((predictions[:, class_idx] == 0) & (labels[:, class_idx] == 1))

        # Calculate precision, recall, and F1 score for the current class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Store metrics for the current class
        class_metrics[f"class_{class_idx}_true_positives"] = tp
        class_metrics[f"class_{class_idx}_false_positives"] = fp
        class_metrics[f"class_{class_idx}_true_negatives"] = tn
        class_metrics[f"class_{class_idx}_false_negatives"] = fn
        class_metrics[f"class_{class_idx}_precision"] = precision
        class_metrics[f"class_{class_idx}_recall"] = recall
        class_metrics[f"class_{class_idx}_f1"] = f1

    return class_metrics

class RoBERTaModels(DataHandler):

    def __init__(self, args):
        super().__init__(args)
        #if self.args.maximise_model: super().find_best_args()
        super().calculate_weights()
        self.model_id = self.args.model_path
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.custom_loss = None



    def tokenize(self, batch):
        return self.tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

    def modify_data(self, data):
        data = data.map(self.tokenize, batched=True)
        data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        data = data.remove_columns(['text'])
        # val.to(device)
        data = data.map(lambda x: {key: val for key, val in x.items()})

        return data


    def initialize(self):

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id) # TODO tokenizer is not loaded properly, if it is a local path; from huggingface directly all is fine

        label_weights = torch.tensor([weight[1] for weight in self.class_weight_dicts])

        self.custom_loss = WeightedBCELoss(weights=label_weights)

        self.model = RobertaForSequenceClassification.from_pretrained(self.model_id, num_labels=self.num_classes)
        self.model.config.problem_type = "multi_label_classification"
        self.model = self.model.to(device)

        self.logger.info("Model Defined!")

        self.data_train = self.modify_data(self.data_train)
        if self.data_val is not None: self.data_val = self.modify_data(self.data_val)
        if self.data_test is not None: self.data_test = self.modify_data(self.data_test)

        self.logger.info("Data Adjusted for RoBERTa!")

    def create_trainer(self):
        args = TrainingArguments(
            output_dir=self.output_path,
            logging_dir=self.logs_path,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            num_train_epochs=self.args.epoch,
            weight_decay=self.args.weight_decay,
            learning_rate=self.args.lr,
            save_strategy="epoch",
            save_total_limit=1
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.trainer = CustomTrainer(
            model=self.model,
            args=args,
            train_dataset=self.data_train.select(range(50)), #TODO remove range
            eval_dataset=self.data_test,
            custom_loss=self.custom_loss,
            data_collator=data_collator,
            compute_metrics=compute_metrics_roberta
        )

        self.logger.info("Trainer Created!")

    def train(self):
        self.trainer.train()
        self.trainer.save_model(os.path.join(self.output_path, "final_model"))

    def eval(self):
        metrics = self.trainer.evaluate()
        self.logger.info(metrics)
        self.summarize_metrics(metrics)
        self.update_csv(metrics)
