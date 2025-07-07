import os
from setfit import SetFitModel, Trainer, TrainingArguments, SetFitHead
from sentence_transformers import SentenceTransformer
from nlbse_jex.data_handler import DataHandler
from nlbse_jex.utils import compute_metrics
import warnings
from typing import Dict, List, Optional, Union, Any
import torch
from torch import nn
from tqdm.auto import tqdm, trange
import yaml


class CustomTrainer(Trainer):
    def __init__(self, freeze_embeddings=False, **kwargs):
        super().__init__(**kwargs)
        self.freeze_embeddings = freeze_embeddings

    def train(
        self,
        args: Optional[TrainingArguments] = None,
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
            self._hp_search_setup(trial)  # sets trainer parameters and initializes model

        args = args or self.args or TrainingArguments()

        if self.train_dataset is None:
            raise ValueError(
                f"Training requires a `train_dataset` given to the `{self.__class__.__name__}` initialization."
            )

        train_parameters = self.dataset_to_parameters(self.train_dataset)

        # Added the following if statement
        if self.freeze_embeddings is False:
            full_parameters = (
                train_parameters + self.dataset_to_parameters(self.eval_dataset) if self.eval_dataset else train_parameters
            )

            self.train_embeddings(*full_parameters, args=args)

        self.train_classifier(*train_parameters, args=args)


class CustomSetFitModel(SetFitModel):
    def __init__(self, *args, custom_loss=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss = torch.tensor(custom_loss, device=self.device) # take device from parent class

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
        criterion = nn.BCEWithLogitsLoss(reduction="none") # changed from self.model_head.get_loss_fn()
        optimizer = self._prepare_optimizer(head_learning_rate, body_learning_rate, l2_weight)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        for epoch_idx in trange(num_epochs, desc="Epoch", disable=not show_progress_bar):
            for batch in tqdm(dataloader, desc="Iteration", disable=not show_progress_bar, leave=False):
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
                logits = outputs["logits"]
                loss: torch.Tensor = criterion(logits, labels).mean(dim=0) * self.custom_loss # added the multiplication
                loss = loss.mean() # added reduction
                loss.backward()
                optimizer.step()

            scheduler.step()

        if not end_to_end:
            self.unfreeze("body")


class SetFitModels(DataHandler):

    def __init__(self, args):
        super().__init__(args)
        if self.args.maximise_model: super().find_best_args()
        super().calculate_weights()
        self.model_id = self.args.model_path
        self.model = None
        self.trainer = None

    def initialize(self):

        model_body = SentenceTransformer(self.model_id)

        model_head = SetFitHead(
            in_features=model_body.get_sentence_embedding_dimension(),
            out_features=self.num_classes,
            multitarget=True
        )

        self.model = CustomSetFitModel(
            model_body=model_body,
            model_head=model_head,
            multi_target_strategy="multi-output",
            use_differentiable_head=True,
            use_contrastive_head=False,
            custom_loss=[weight[1] for weight in self.class_weight_dicts]
        )

        self.logger.info("Model Defined!")

    def create_trainer(self):
        args = TrainingArguments(
            output_dir=self.output_path,
            logging_dir=self.logs_path,
            batch_size=self.args.batch_size,
            num_epochs=self.args.epoch,
            head_learning_rate=self.args.lr,
            #l2_weight=self.args.weight_decay, # https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
            save_strategy="epoch",
            save_total_limit=1
        )

        self.trainer = CustomTrainer(
            model=self.model,
            args=args,
            train_dataset=self.data_train, #.select(range(50)), #TODO remove range
            eval_dataset=self.data_test,
            metric=compute_metrics,
            freeze_embeddings=self.args.freeze_embeddings
        )

        self.logger.info("Trainer Created!")

    def train(self):
        self.trainer.train()
        self.model.model_body.save(os.path.join(self.output_path, "model_body"))
        torch.save(self.model.model_head.state_dict(), os.path.join(self.output_path, "model_head.pt"))
        with open(os.path.join(self.output_path, "training_args.yaml"), "w") as f:
            yaml.dump(vars(self.args), f)

    def eval(self):
        metrics = self.trainer.evaluate()
        self.logger.info(metrics)
        self.summarize_metrics(metrics)
        self.update_csv(metrics)

