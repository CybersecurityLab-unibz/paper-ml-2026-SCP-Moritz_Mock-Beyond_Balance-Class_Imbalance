import os
import json
from huggingface_hub import model_info
from huggingface_hub.utils import HfHubHTTPError
from nlbse_jex.roberta_models import RoBERTaModels
from nlbse_jex.setfit_models import SetFitModels
import logging


class Proxy:

    def __init__(self, args):
        """
        Initializes the Proxy class responsible for managing model-related metadata and configuration.

        This constructor:
            - Sets up logging for the proxy module
            - Initializes attributes for model path, model type, and model instance
            - Stores the provided `args` object
            - Defines whitelists for model types supported by SetFit and RoBERTa-specific logic

        Parameters:
            args (Namespace): An object containing configuration parameters such as model path and type.

        Returns:
            None
        """

        self.logger = logging.getLogger("nlbse_jsx.proxy")
        self.model_path = None
        self.model_type = None
        self.model = None
        self.args = args
        self.setfit_whitelist = ["bert", "mpnet"]
        self.roberta_whitelist = ["roberta"]

    def _type_of_model(self) -> str | None:
        """
        Determines the type of model based on the model path.

        This method:
            - Checks if the model path points to a directory and looks for a "snapshots" subdirectory
            - If found, attempts to read the `config.json` file inside the first valid snapshot to extract the model type
            - Updates `self.args.model_path` to the snapshot path if applicable
            - If local config extraction fails, tries to fetch model type information from the Hugging Face Hub
            - Handles errors gracefully and returns None if the model type cannot be determined

        Returns:
            str | None: The model type as a string if found (e.g., "bert", "roberta"), otherwise None.
        """

        if os.path.isdir(self.model_path):
            snapshots_dir = os.path.join(self.model_path, "snapshots")
            if os.path.isdir(snapshots_dir):
                snapshots = [
                    d
                    for d in os.listdir(snapshots_dir)
                    if not d.startswith(".")
                    and os.path.isdir(os.path.join(snapshots_dir, d))
                ]
                if snapshots:
                    config_path = os.path.join(
                        snapshots_dir, snapshots[0], "config.json"
                    )
                    if os.path.isfile(config_path):
                        try:
                            with open(config_path, "r") as f:
                                config = json.load(f)
                                self.logger.info(self.args)
                                self.args.model_path = os.path.join(
                                    snapshots_dir, snapshots[0]
                                )
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
        """
        Creates and initializes a model instance based on the detected model type.

        This method:
            - Sets the model path and determines the model type using `_type_of_model()`
            - Initializes the appropriate model class (`RoBERTaModels` or `SetFitModels`) if the model type is supported
            - Calls the `initialize()` method on the model instance to prepare it for training or inference

        Returns:
            None
        """

        self.model_path = self.args.model_path
        self.model_type = self._type_of_model()

        if self.model_type in self.roberta_whitelist:
            self.model = RoBERTaModels(self.args)
        elif self.model_type in self.setfit_whitelist:
            self.model = SetFitModels(self.args)

        if self.model:
            self.model.initialize()

    def create_trainer(self):
        """
        Creates a trainer for the current model, if the model has been initialized.

        This method:
            - Checks if a model instance exists
            - Invokes the `create_trainer()` method on the model to set up training utilities

        Returns:
            None
        """

        if self.model:
            self.model.create_trainer()

    def train(self):
        """
        Starts the training process for the initialized model.

        This method:
            - Checks if a model instance exists
            - Invokes the `train()` method on the model to begin training

        Returns:
            None
        """

        if self.model:
            self.model.train()

    def eval(self):
        """
        Evaluates the initialized model on the validation or test set.

        This method:
            - Checks if a model instance exists
            - Calls the `eval()` method on the model to perform evaluation

        Returns:
            None
        """

        if self.model:
            self.model.eval()

    def run_already_exists(self):
        """
        Checks whether a training run with the current model configuration already exists.

        This method:
            - Delegates the check to the `run_already_exists()` method of the model instance
            - Returns None if the model has not been initialized

        Returns:
            bool or None: True if the run exists, False if it does not, or None if the model is uninitialized.
        """

        return self.model.run_already_exists() if self.model is not None else None
