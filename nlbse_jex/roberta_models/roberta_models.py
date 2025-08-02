from transformers import AutoTokenizer
from nlbse_jex.data_handler import DataHandler
from transformers import RobertaForSequenceClassification, TrainingArguments, \
    DataCollatorWithPadding
import torch
import os

from nlbse_jex.roberta_models.WeightedBCELoss import WeightedBCELoss
from nlbse_jex.roberta_models.roberta_trainer import RoBERTaTrainer
from nlbse_jex.utils import compute_metrics_roberta
from nlbse_jex.dynamic_weights.weight_methods import WeightMethods
from nlbse_jex.utils import get_device


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

        if self.isDynamic:
            self.custom_loss_weight = torch.ones(self.num_classes,device=self.device)
            self.dynamic_weight_loss = WeightMethods(
                method=self.args.weighted_loss_strategy,
                n_tasks=self.num_classes,
                device=self.device
            )
        else:
            self.custom_loss_weight = torch.tensor([weight[1] for weight in self.class_weight_dicts])

        self.custom_loss = WeightedBCELoss(self.custom_loss_weight)

        self.model = RobertaForSequenceClassification.from_pretrained(self.model_id, num_labels=self.num_classes)
        self.model.config.problem_type = "multi_label_classification"
        self.model = self.model.to(get_device())

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

        self.trainer = RoBERTaTrainer(
            model=self.model,
            args=args,
            train_dataset=self.data_train.select(range(50)), #TODO remove range
            eval_dataset=self.data_test,
            custom_loss=self.custom_loss,
            data_collator=data_collator,
            compute_metrics=compute_metrics_roberta,
            dynamic_weight_loss=self.dynamic_weight_loss,
            num_classes=self.num_classes,
            isDynamic=self.isDynamic
        )

        self.logger.info("Trainer Created!")

    def train(self):
        self.trainer.train()
        self.trainer.save_model(os.path.join(self.output_path, "final_model"))

    def eval(self):
        metrics = self.trainer.evaluate()

        metrics = {(k.replace("eval_", "") if k.startswith("eval_") else k): v for k, v in metrics.items()}

        self.logger.info(metrics)
        self.summarize_metrics(metrics)
        self.update_csv(metrics)
