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
        super().__init__(args)
        if self.args.maximise_model: super().find_best_args()
        super().calculate_weights()
        self.model_id = self.args.model_path
        self.model = None
        self.trainer = None

    def initialize(self):

        model_body = SentenceTransformer(self.model_id)

        model_head = setfit.SetFitHead(
            in_features=model_body.get_sentence_embedding_dimension(),
            out_features=self.num_classes,
            multitarget=False
        )

        if self.isDynamic:
            self.custom_loss_weight = torch.ones(self.num_classes)
        else:
            self.custom_loss_weight = torch.tensor([weight[1] for weight in self.class_weight_dicts])


        self.model = SetFitModel(
            model_body=model_body,
            model_head=model_head,
            multi_target_strategy="multi-output",
            use_differentiable_head=True,
            use_contrastive_head=False,
            custom_loss_weight=self.custom_loss_weight,
            weighted_loss_strategy=self.args.weighted_loss_strategy,
            num_classes=self.num_classes,
            isDynamic=self.isDynamic
        )

        self.logger.info("Model Defined!")

    def create_trainer(self):
        args = setfit.TrainingArguments(
            output_dir=self.output_path,
            logging_dir=self.logs_path,
            batch_size=self.args.batch_size,
            num_epochs=self.args.epoch,
            head_learning_rate=self.args.lr,
            l2_weight=self.args.weight_decay, # https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html
            save_strategy="epoch",
            save_total_limit=1,
            num_iterations=20 # as default in setfit 1.0.0
        )

        self.trainer = SetFitTrainer(
            model=self.model,
            args=args,
            train_dataset=self.data_train, #.select(range(50)), #TODO remove range
            eval_dataset=self.data_test,
            metric=compute_metrics_setfit,
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

