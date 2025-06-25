from setfit import SetFitModel, Trainer, TrainingArguments
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from nlbse_jex.data_handler import DataHandler
from nlbse_jex.utils import compute_metrics

class MultiOutputClassifier:
    def __init__(self, estimators):
        self.estimators_ = estimators

    def fit(self, X, Y):
        Y = np.array(Y)
        for i, clf in enumerate(self.estimators_):
            clf.fit(X, Y[:, i])
        return self

    def predict(self, X):
        return np.column_stack([clf.predict(X) for clf in self.estimators_])


class SetFitModels(DataHandler):

    def __init__(self, args):
        super().__init__(args)
        super().calculate_weights()
        self.model_id = self.args.model_path
        self.model = None
        self.trainer = None

    def initialize(self):

        model_body = SentenceTransformer(self.model_id)
        # https://stackoverflow.com/questions/30972029/how-does-the-class-weight-parameter-in-scikit-learn-work
        base_clfs = [
            LogisticRegression(class_weight=self.class_weight_dicts[i], max_iter=1000)
            for i in range(self.num_classes)
        ]

        model_head = MultiOutputClassifier(base_clfs)

        self.model = SetFitModel(
            model_body=model_body,
            model_head=model_head,
            multi_target_strategy="multi-output",
            use_contrastive_head=False
        )

        self.logger.info("Defined model!")

    def create_trainer(self):
        args = TrainingArguments(
            output_dir=".",
            logging_dir=".",
            batch_size=32,
            num_epochs=1,
            save_strategy="no"
        )

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.data_train.select(range(50)), #TODO remove range
            eval_dataset=self.data_test,
            metric=compute_metrics
        )

        self.logger.info("Created Trainer!")

    def train(self):
        self.trainer.train()

    def eval(self):
        zeroshot_metrics = self.trainer.evaluate()
        self.summarize_metrics(zeroshot_metrics)