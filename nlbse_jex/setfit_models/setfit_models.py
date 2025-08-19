import os
import setfit
from sentence_transformers import SentenceTransformer
from nlbse_jex.data_handler import DataHandler
from nlbse_jex.setfit_models.setfit_model import SetFitModel
from nlbse_jex.setfit_models.setfit_trainer import SetFitTrainer
from nlbse_jex.utils import compute_metrics_setfit
import torch
import yaml


class SetFitModels(DataHandler):

    def __init__(self, args):
        """
        Initializes a model wrapper that prepares arguments, computes weights, and optionally optimizes configuration.

        This constructor:
            - Calls the parent class initializer with the given arguments
            - If `maximise_model` is enabled, it finds the best configuration from past runs
            - Computes class weights based on the current strategy
            - Initializes placeholders for model ID, model instance, and trainer

        Parameters:
            args (argparse.Namespace): The configuration object containing model setup, training parameters, and optimization flags.

        Returns:
            None
        """

        super().__init__(args)
        if self.args.maximise_model:
            super().find_best_args()
        super().calculate_weights()
        self.model_id = self.args.model_path
        self.model = None
        self.trainer = None

    def initialize(self):
        """
        Initializes the SetFit model with the appropriate architecture and loss weighting configuration.

        This method:
            - Loads the base model (`SentenceTransformer`) using `self.model_id`
            - Creates a classification head (`SetFitHead`) with input dimensions from the base model and output dimensions matching the number of classes
            - Prepares class weights:
                - Uses uniform weights if the loss strategy is dynamic
                - Otherwise uses precomputed static weights from `class_weight_dicts`
            - Constructs the full `SetFitModel` with configuration for multi-label classification and weighted loss
            - Logs a message confirming model definition

        Returns:
            None
        """

        model_body = SentenceTransformer(self.model_id)

        model_head = setfit.SetFitHead(
            in_features=model_body.get_sentence_embedding_dimension(),
            out_features=self.num_classes,
            multitarget=True,
        )

        if self.isDynamic:
            self.custom_loss_weight = torch.ones(self.num_classes)
        else:
            self.custom_loss_weight = torch.tensor(
                [weight[1] for weight in self.class_weight_dicts]
            )

        self.model = SetFitModel(
            model_body=model_body,
            model_head=model_head,
            multi_target_strategy="multi-output",
            use_differentiable_head=True,
            use_contrastive_head=False,
            custom_loss_weight=self.custom_loss_weight,
            weighted_loss_strategy=self.args.weighted_loss_strategy,
            num_classes=self.num_classes,
            isDynamic=self.isDynamic,
        )

        self.logger.info("Model Defined!")

    def create_trainer(self):
        """
        Creates and configures a `SetFitTrainer` instance for training the SetFit model.

        This method:
            - Initializes `setfit.TrainingArguments` with training hyperparameters and file paths
            - Constructs a `SetFitTrainer` with the model, training and evaluation datasets, metric function, and other settings
            - Stores the trainer instance and logs a confirmation message

        Trainer settings include:
            - Output and logging directories
            - Batch size, number of epochs, learning rate, weight decay
            - Save strategy and iteration count
            - Whether to freeze embeddings during training

        Returns:
            None
        """

        args = setfit.TrainingArguments(
            output_dir=self.output_path,
            logging_dir=self.logs_path,
            batch_size=self.args.batch_size,
            num_epochs=self.args.epoch,
            head_learning_rate=self.args.lr,
            l2_weight=self.args.weight_decay,  # https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
            save_strategy="epoch",
            save_total_limit=1,
            num_iterations=20,  # as default in setfit 1.0.0
        )

        self.trainer = SetFitTrainer(
            model=self.model,
            args=args,
            train_dataset=self.data_train,  # .select(range(50)), #TODO remove range
            eval_dataset=self.data_test,
            metric=compute_metrics_setfit,
            freeze_embeddings=self.args.freeze_embeddings,
        )

        self.logger.info("Trainer Created!")

    def train(self):
        """
        Trains the SetFit model and saves the resulting components and training configuration.

        This method:
            - Starts the training process using the `SetFitTrainer`
            - Saves the trained sentence embedding model (`model_body`) to the output path
            - Saves the trained classification head's state dictionary (`model_head.pt`)
            - Writes the training arguments to a YAML file (`training_args.yaml`) for reproducibility

        Returns:
            None
        """

        self.trainer.train()
        self.model.model_body.save(os.path.join(self.output_path, "model_body"))
        torch.save(
            self.model.model_head.state_dict(),
            os.path.join(self.output_path, "model_head.pt"),
        )
        with open(os.path.join(self.output_path, "training_args.yaml"), "w") as f:
            yaml.dump(vars(self.args), f)

    def eval(self):
        """
        Evaluates the trained model and logs performance metrics.

        This method:
            - Calls the trainer's `evaluate()` method to compute metrics on the evaluation dataset
            - Logs the full metrics dictionary
            - Summarizes and logs precision, recall, and F1 scores per class
            - Updates the results CSV with the current evaluation metrics

        Returns:
            None
        """

        metrics = self.trainer.evaluate()
        self.logger.info(metrics)
        self.summarize_metrics(metrics)
        self.update_csv(metrics)
