import os
import json
from huggingface_hub import model_info
from huggingface_hub.utils import HfHubHTTPError
from nlbse_jex.roberta_model import RoBERTaModels
from nlbse_jex.setfit_model import SetFitModels
import logging

class Proxy():

    def __init__(self, args):
        self.logger = logging.getLogger()
        self.model_path = None
        self.model_type = None
        self.model = None
        self.args = args
        self.setfit_whitelist = ["bert", "mpnet"]
        self.roberta_whitelist = ["roberta"]

    def _type_of_model(self) -> str|None:
        if os.path.isdir(self.model_path):
            snapshots_dir = os.path.join(self.model_path, "snapshots")
            if os.path.isdir(snapshots_dir):
                snapshots = [d for d in os.listdir(snapshots_dir)
                             if not d.startswith(".") and os.path.isdir(os.path.join(snapshots_dir, d))]
                if snapshots:
                    config_path = os.path.join(snapshots_dir, snapshots[0], "config.json")
                    if os.path.isfile(config_path):
                        try:
                            with open(config_path, "r") as f:
                                config = json.load(f)
                                self.logger.info(self.args)
                                self.args.model_path = os.path.join(snapshots_dir, snapshots[0])
                                self.logger.info(self.args)
                                return config.get("model_type", "unknown")
                        except (json.JSONDecodeError, OSError):
                            return None

        try:
            info = model_info(self.model_path)
            if info.config:
                return info.config["model_type"]
        except HfHubHTTPError:
            pass

        return None

    def create_model(self):
        self.model_path = self.args.model_path
        self.model_type = self._type_of_model()

        self.logger.info(self.model_path)
        self.logger.info(self.model_type)

        if self.model_type in self.roberta_whitelist:
            self.model = RoBERTaModels(self.args)
        elif self.model_type in self.setfit_whitelist:
            self.model = SetFitModels(self.args)

        if self.model: self.model.initialize()

    def create_trainer(self):
        if self.model: self.model.create_trainer()

    def train(self):
        if self.model: self.model.train()

    def eval(self):
        if self.model: self.model.eval()

    def get_model_type(self):
        return self.model_type

    def run_already_exists(self):
        return self.model.run_already_exists() if self.model is not None else None