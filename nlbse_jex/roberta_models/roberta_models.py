from transformers import AutoTokenizer
from nlbse_jex.data_handler import DataHandler
from transformers import (
    RobertaForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
)
import torch
import os

from nlbse_jex.roberta_models.WeightedBCELoss import WeightedBCELoss
from nlbse_jex.roberta_models.roberta_trainer import RoBERTaTrainer
from nlbse_jex.utils import compute_metrics_roberta
from nlbse_jex.dynamic_weights.weight_methods import WeightMethods
from nlbse_jex.utils import get_device


class RoBERTaModels(DataHandler):

    def __init__(self, args):
        """
        Initializes the training wrapper for a RoBERTa-based model.

        This constructor:
            - Inherits initialization from the parent class with the given arguments
            - If `maximise_model` is enabled, finds and applies the best configuration from prior runs
            - Computes class weights using the specified strategy
            - Stores the model ID and initializes placeholders for:
                - Tokenizer
                - Model
                - Trainer
                - Custom loss function (if needed)

        Parameters:
            args (argparse.Namespace): Configuration and training parameters.

        Returns:
            None
        """

        super().__init__(args)
        if self.args.maximise_model:
            super().find_best_args()
        super().calculate_weights()
        self.model_id = self.args.model_path
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.custom_loss = None

    def tokenize(self, batch):
        """
        Tokenizes a batch of input text using the model's tokenizer.

        This method:
            - Applies padding and truncation to ensure consistent input lengths
            - Limits tokenized sequences to a maximum of 512 tokens

        Parameters:
            batch (dict): A dictionary containing a "text" key with a list of input strings.

        Returns:
            dict: A dictionary of tokenized inputs suitable for the model.
        """

        return self.tokenizer(
            batch["text"], padding=True, truncation=True, max_length=512
        )

    def modify_data(self, data):
        """
        Preprocesses and formats a dataset for use with a RoBERTa-based model.

        This method:
            - Applies tokenization to the dataset using `self.tokenize`, processing batches of examples
            - Formats the dataset for PyTorch with only the necessary columns:
                "input_ids", "attention_mask", and "label"
            - Removes the original "text" column
            - Applies a final mapping to ensure a consistent data structure

        Parameters:
            data (datasets.Dataset): The dataset to preprocess.

        Returns:
            datasets.Dataset: The modified and formatted dataset ready for model training or evaluation.
        """

        data = data.map(self.tokenize, batched=True)
        data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        data = data.remove_columns(["text"])
        data = data.map(lambda x: {key: val for key, val in x.items()})

        return data

    def initialize(self):
        """
        Initializes a RoBERTa-based model for multi-label classification, including tokenizer, model, loss, and data preprocessing.

        This method:
            - Loads the tokenizer using the provided model ID (note: local paths may require manual handling)
            - Initializes class weights:
                - Uses uniform weights and a dynamic weighting method if `isDynamic` is True
                - Otherwise, uses static class weights derived from `class_weight_dicts`
            - Initializes the custom binary cross-entropy loss with class weights
            - Loads a RoBERTa model for sequence classification with the specified number of labels
            - Sets the problem type to "multi_label_classification"
            - Moves the model to the appropriate device
            - Preprocesses training, validation, and test datasets with tokenization and formatting
            - Logs the completion of model and data initialization

        Returns:
            None
        """

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id
        )  # TODO tokenizer is not loaded properly, if it is a local path; from huggingface directly all is fine

        if self.isDynamic:
            self.custom_loss_weight = torch.ones(self.num_classes, device=self.device)
            self.dynamic_weight_loss = WeightMethods(
                method=self.args.weighted_loss_strategy,
                n_tasks=self.num_classes,
                device=self.device,
            )
        else:
            self.custom_loss_weight = torch.tensor(
                [weight[1] for weight in self.class_weight_dicts]
            )

        self.custom_loss = WeightedBCELoss(self.custom_loss_weight)

        self.model = RobertaForSequenceClassification.from_pretrained(
            self.model_id, num_labels=self.num_classes
        )
        self.model.config.problem_type = "multi_label_classification"
        self.model = self.model.to(get_device())

        self.logger.info("Model Defined!")

        self.data_train = self.modify_data(self.data_train)
        if self.data_val is not None:
            self.data_val = self.modify_data(self.data_val)
        if self.data_test is not None:
            self.data_test = self.modify_data(self.data_test)

        self.logger.info("Data Adjusted for RoBERTa!")

    def create_trainer(self):
        """
        Creates and configures a `RoBERTaTrainer` instance for training a RoBERTa-based multi-label classification model.

        This method:
            - Initializes `TrainingArguments` with training and optimization parameters
            - Sets up a `DataCollatorWithPadding` to handle dynamic padding during batching
            - Instantiates a custom `RoBERTaTrainer` with:
                - Model and datasets
                - Custom loss function
                - Metric computation method
                - Support for dynamic or static loss weighting
            - (Note: `train_dataset.select(range(50))` is used temporarily for testing/debugging)
            - Logs confirmation after trainer creation

        Returns:
            None
        """

        args = TrainingArguments(
            output_dir=self.output_path,
            logging_dir=self.logs_path,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            num_train_epochs=self.args.epoch,
            weight_decay=self.args.weight_decay,
            learning_rate=self.args.lr,
            save_strategy="epoch",
            save_total_limit=1,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.trainer = RoBERTaTrainer(
            model=self.model,
            args=args,
            train_dataset=self.data_train.select(range(50)),  # TODO remove range
            eval_dataset=self.data_test,
            custom_loss=self.custom_loss,
            data_collator=data_collator,
            compute_metrics=compute_metrics_roberta,
            dynamic_weight_loss=self.dynamic_weight_loss,
            num_classes=self.num_classes,
            isDynamic=self.isDynamic,
        )

        self.logger.info("Trainer Created!")

    def train(self):
        """
        Trains the RoBERTa-based model and saves the final trained model to disk.

        This method:
            - Starts the training process using the `RoBERTaTrainer`
            - Saves the fully trained model to the `final_model` directory under the configured output path

        Returns:
            None
        """

        self.trainer.train()
        self.trainer.save_model(os.path.join(self.output_path, "final_model"))

    def eval(self):
        """
        Evaluates the trained RoBERTa model and logs the results.

        This method:
            - Runs evaluation using the `RoBERTaTrainer`'s `evaluate()` method
            - Strips the "eval_" prefix from metric keys (if present) for cleaner logging and reporting
            - Logs the full metrics dictionary
            - Summarizes class-wise precision, recall, and F1 metrics
            - Updates the results CSV with the evaluation outcomes

        Returns:
            None
        """

        metrics = self.trainer.evaluate()

        metrics = {
            (k.replace("eval_", "") if k.startswith("eval_") else k): v
            for k, v in metrics.items()
        }

        self.logger.info(metrics)
        self.summarize_metrics(metrics)
        self.update_csv(metrics)
