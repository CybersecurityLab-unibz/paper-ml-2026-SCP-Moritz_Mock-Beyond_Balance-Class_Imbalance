import importlib.util
import os
import ast
from typing import get_origin, get_args, Union
import inspect
import yaml
import numpy as np
import shutil
import types
import logging
import argparse
import torch
import json


def read_files_weighted_loss_strategy(args):
    """
    Reads and imports any custom weighted loss strategy functions provided as file paths.

    This function:
        - Iterates over the `weighted_loss_strategy` entries in `args`
        - For entries ending with `.py`, imports the first function from the file
        - Validates that the function can accept a `labels` argument using `can_pass_args_with_types`
        - Replaces the file path with the actual function object in `args.weighted_loss_strategy`
        - Asserts that all imported functions are valid; raises an error with details if mismatches are found

    Parameters:
        args (argparse.Namespace): The parsed command-line or configuration arguments.

    Returns:
        argparse.Namespace: The updated `args` object with function references substituted for file paths.
    """

    args_ok = {"labels": "placeholder"}

    for index, wl in enumerate(args.weighted_loss_strategy):
        if wl.endswith(".py"):
            args.weighted_loss_strategy[index] = dynamic_import_code(wl, args_ok)

    return args


def read_file_compute_metric(args):
    """
    Reads and imports any custom weighted loss strategy functions provided as file paths.

    This function:
        - Iterates over the `weighted_loss_strategy` entries in `args`
        - For entries ending with `.py`, imports the first function from the file
        - Validates that the function can accept a `labels` argument using `can_pass_args_with_types`
        - Replaces the file path with the actual function object in `args.weighted_loss_strategy`
        - Asserts that all imported functions are valid; raises an error with details if mismatches are found

    Parameters:
        args (argparse.Namespace): The parsed command-line or configuration arguments.

    Returns:
        argparse.Namespace: The updated `args` object with function references substituted for file paths.
    """
    # preds, labels, num_classes

    args_ok = {
        "preds": "placeholder",
        "labels": "placeholder",
        "num_classes": "placeholder",
    }

    if args.costume_metrics is not None and args.costume_metrics.endswith(".py"):
        args.costume_metrics = dynamic_import_code(args.costume_metrics, args_ok)

    return args


def dynamic_import_code(wl, args_ok):
    """
    Dynamically imports and validates a function from a Python file path.

    This function:
        - Uses `import_function_from_path` to load the first function defined in the file specified by `wl`
        - Validates the imported function's signature using `can_pass_args_with_types`
        - Ensures that the function can accept the required arguments as defined in `args_ok`
        - Raises an AssertionError with mismatch details if the function signature is incompatible

    Parameters:
        wl (str): Path to the Python file containing the function to import.
        args_ok (dict): A mapping of argument names and placeholder values used to validate the function's signature.

    Returns:
        function: The imported and validated function object.
    """
    function = import_function_from_path(wl)
    ok, mismatches = can_pass_args_with_types(function, args_ok)
    assert ok is True, "\n".join(mismatches)
    return function


def can_pass_args_with_types(func, args_dict):
    """
    Checks whether a given dictionary of arguments can be passed to a function,
    validating both argument presence and type annotations.

    This function:
        - Compares the provided argument names and types with the function's signature
        - Handles standard arguments and accounts for **kwargs
        - Supports type checking for annotated arguments using `_is_instance_of`
        - Detects missing, unexpected, or type-mismatched arguments

    Parameters:
        func (Callable): The target function to inspect.
        args_dict (dict): A dictionary of argument names and values to validate.

    Returns:
        tuple:
            - bool: True if all arguments are valid, False otherwise.
            - list[str]: A list of mismatch error messages, empty if all arguments are valid.
    """

    sig = inspect.signature(func)
    params = sig.parameters
    mismatches = []

    for name, value in args_dict.items():
        param = params.get(name)

        if param is None:
            # Check for **kwargs
            if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
                mismatches.append(f"Missing required argument '{name}'")
            continue

        expected_type = param.annotation
        if expected_type is not inspect.Parameter.empty:
            if not _is_instance_of(value, expected_type):
                mismatches.append(
                    f"Argument '{name}' has wrong type: expected {type(value)}, got {expected_type}"
                )

    # Check for to many args
    for p in params.values():
        if (
            p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            and p.name not in args_dict
        ):
            mismatches.append(f"Unexpected argument '{p.name}'")

    return len(mismatches) == 0, mismatches


def _is_instance_of(value, expected_type):
    """
    Recursively checks whether a value is an instance of a complex typing type.

    Supports Python typing constructs such as:
        - Union
        - List[...]
        - Dict[..., ...]
        - Tuple[...]
        - Basic types

    Parameters:
        value (Any): The value to be checked.
        expected_type (type or typing.*): The expected type, possibly from the `typing` module.

    Returns:
        bool: True if the value matches the expected type structure, False otherwise.
    """

    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is None:
        return isinstance(value, expected_type)

    if origin is list:
        return isinstance(value, list) and all(
            _is_instance_of(v, args[0]) for v in value
        )

    if origin is dict:
        return isinstance(value, dict) and all(
            _is_instance_of(k, args[0]) and _is_instance_of(v, args[1])
            for k, v in value.items()
        )

    if origin is tuple:
        return isinstance(value, tuple) and all(
            _is_instance_of(v, t) for v, t in zip(value, args)
        )

    if origin is Union:
        return any(_is_instance_of(value, t) for t in args)

    return isinstance(value, expected_type)


def get_first_function_name(file_path):
    """
    Extracts the name of the first function defined in a Python source file.

    This function:
        - Parses the Python file into an abstract syntax tree (AST)
        - Scans the top-level body for the first function definition
        - Returns the name of that function

    Parameters:
        file_path (str): The path to the Python file to inspect.

    Returns:
        str: The name of the first function defined in the file.

    Raises:
        ValueError: If no function definitions are found in the file.
    """

    with open(file_path, "r") as f:
        node = ast.parse(f.read(), filename=file_path)

    for n in node.body:
        if isinstance(n, ast.FunctionDef):
            return n.name
    raise ValueError(f"No function definitions found in '{file_path}'")


def import_function_from_path(file_path):
    """
    Dynamically imports the first function defined in a Python file and returns it.

    This function:
        - Extracts the name of the first function using `get_first_function_name`
        - Loads the file as a module using `importlib.util`
        - Retrieves the function object from the loaded module
        - Validates that the object is callable

    Parameters:
        file_path (str): The path to the Python file containing the function.

    Returns:
        Callable: The first function defined in the specified file.

    Raises:
        ImportError: If the file cannot be loaded as a module.
        AttributeError: If no callable function is found with the expected name.
    """

    function_name = get_first_function_name(file_path)
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    func = getattr(module, function_name, None)
    if not callable(func):
        raise AttributeError(f"No callable '{function_name}' found in '{file_path}'")

    return func


def str2bool(v):
    """
    Converts a string representation of a boolean to a boolean value.

    This function:
        - Returns the input directly if it's already a boolean
        - Interprets various common string formats for True and False (case-insensitive)
        - Raises a ValueError if the string does not represent a valid boolean

    Parameters:
        v (str or bool): The input value to convert.

    Returns:
        bool: The corresponding boolean value.

    Raises:
        ValueError: If the input cannot be interpreted as a boolean.
    """

    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")


def str2json(value):
    """
    Parses a JSON string into a Python dictionary for use with argparse.

    This function:
        - Returns the input directly if it's already a dictionary
        - Attempts to parse the input string using `json.loads`
        - Raises an `ArgumentTypeError` if the string is not valid JSON

    Parameters:
        value (str or dict): The input JSON string or dictionary.

    Returns:
        dict: The parsed dictionary representation of the JSON string.

    Raises:
        argparse.ArgumentTypeError: If the input string is not valid JSON.
    """

    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON string: {e}")


def generate_base_config(path, data=None):
    """
    Generates and writes a base YAML configuration file to the specified path.

    This function:
        - Uses the provided `data` dictionary if available
        - Otherwise, defaults to a base configuration with sample values for "epoch" and "weighted_loss_strategy"
        - Writes the configuration in a readable block format (not inline)

    Parameters:
        path (str): The file path where the YAML config will be saved.
        data (dict, optional): A dictionary containing configuration keys and values to write.
                               If None, a default config is used.

    Returns:
        None
    """

    data = (
        data
        if data is not None
        else {"epoch": [1, 2, 5, 10], "weighted_loss_strategy": ["EW"]}
    )

    with open(path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def load_yaml_config(config_path):
    """
    Loads a YAML configuration file from the specified path.

    This function:
        - Opens the file at `config_path`
        - Parses its contents using `yaml.safe_load` to ensure safe deserialization

    Parameters:
        config_path (str): The path to the YAML configuration file.

    Returns:
        dict or None: The parsed YAML configuration as a dictionary, or None if the file is empty.

    Raises:
        OSError: If the file cannot be opened.
        yaml.YAMLError: If the YAML content is malformed.
    """

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_and_cast_config(yaml_config, parser):
    """
    Validates and casts a YAML configuration dictionary according to the argument parser's expected types.

    This function:
        - Checks that all keys in the YAML config are defined in the argument parser
        - For each config value:
            - Ensures list-type arguments match the expected list structure and element types (based on `nargs`)
            - Casts scalar values to the expected type
        - Raises detailed errors if any type mismatches or unknown keys are found

    Parameters:
        yaml_config (dict): The configuration dictionary loaded from a YAML file.
        parser (argparse.ArgumentParser): The argument parser defining expected argument names and types.

    Returns:
        dict: A validated and type-cast configuration dictionary ready for merging with CLI arguments.

    Raises:
        ValueError: If unknown keys are found, if types are incorrect, or if list structures don't match expectations.
    """

    validated_config = {}
    arg_dest_to_action = {action.dest: action for action in parser._actions}

    for key, value in yaml_config.items():
        if key not in arg_dest_to_action:
            raise ValueError(f"Unknown config key: '{key}' (not defined in argparse)")

        action = arg_dest_to_action[key]
        expected_type = action.type
        nargs = action.nargs

        # Handle nargs / list-type args
        if nargs is not None:
            if not isinstance(value, list):
                raise ValueError(
                    f"Expected a list for argument '{key}', but got {type(value).__name__}"
                )

            # Cast each element in the list
            if expected_type is not None:
                try:
                    validated_config[key] = [expected_type(v) for v in value]
                except Exception as e:
                    raise ValueError(
                        f"Type error in list for key '{key}': expected elements of type {expected_type.__name__}, got {value}"
                    )

        # Handle normal scalar args
        elif expected_type is not None:
            try:
                validated_config[key] = expected_type(value)
            except Exception as e:
                raise ValueError(
                    f"Type error for key '{key}': expected {expected_type.__name__}, got {value} ({type(value).__name__})"
                )

    return validated_config


def merge_config_with_args(yaml_config, cli_args, parser):
    """
    Merges a YAML configuration with command-line arguments, giving priority to CLI values.

    This function:
        - Converts CLI arguments (Namespace) to a dictionary
        - Iterates over YAML config entries
        - For each key, updates the CLI config only if:
            - The CLI argument is not explicitly set (i.e., it's None or the default value)

    Parameters:
        yaml_config (dict): The configuration dictionary loaded from a YAML file.
        cli_args (argparse.Namespace): The command-line arguments provided by the user.
        parser (argparse.ArgumentParser): The argument parser used to determine default values.

    Returns:
        dict: A merged dictionary of configuration values with CLI arguments taking precedence.
    """

    config = vars(cli_args).copy()

    for key, value in yaml_config.items():
        # CLI args take priority
        if getattr(cli_args, key, None) in [None, parser.get_default(key)]:
            config[key] = value

    return config


def read_config(args, parser):
    """
    Reads and merges configuration settings from a YAML file with command-line arguments.

    This function:
        - Loads a YAML config from the path specified in `args.config_path`
        - Logs and returns the original `args` if the YAML file is empty or not found
        - Validates and casts the raw YAML config using the provided argument parser
        - Merges the validated config with command-line arguments, giving precedence to command-line values
        - Returns a new `argparse.Namespace` object with the combined configuration

    Parameters:
        args (argparse.Namespace): The original command-line arguments.
        parser (argparse.ArgumentParser): The parser used to define expected argument types and defaults.

    Returns:
        argparse.Namespace: A namespace containing the merged configuration.
    """

    raw_yaml = load_yaml_config(args.config_path)
    if raw_yaml is None:
        logging.info(
            f"Provided file ({raw_yaml}) is empty, please specify a configuration!"
        )
        return args
    yaml_config = validate_and_cast_config(raw_yaml, parser)
    return argparse.Namespace(**merge_config_with_args(yaml_config, args, parser))


def compute_metrics_setfit(preds, labels, costume_metric=None):
    """
    Computes evaluation metrics for SetFit models in a multi-label classification setting.

    This function:
        - Converts predictions and labels to NumPy arrays if they are PyTorch tensors or lists
        - One-hot encodes the predicted class indices to match the shape of the true labels
        - Computes per-class metrics (precision, recall, F1 score, confusion matrix values) using `compute_metrics`

    Parameters:
        preds (Union[Tensor, list, np.ndarray]): Predicted class indices.
        labels (Union[Tensor, list, np.ndarray]): Ground truth binary label vectors.
        costume_metric: Costume metric, which can be passed by the user

    Returns:
        dict: A dictionary of computed evaluation metrics per class.
    """

    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    elif isinstance(preds, list):
        preds = np.array(preds)

    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    num_classes = labels.shape[1]

    metric_fn = costume_metric or compute_metrics
    return metric_fn(preds, labels, num_classes)


def compute_metrics_roberta(eval_pred, costume_metric=None):
    """
    Computes evaluation metrics for RoBERTa-based multilabel classification models.

    This function:
        - Unpacks logits and true labels from the evaluation prediction tuple
        - Converts logits and labels to PyTorch tensors if they are NumPy arrays
        - Applies the sigmoid function to logits to obtain class probabilities
        - Thresholds the probabilities at 0.5 to obtain binary predictions
        - Converts tensors back to NumPy arrays for metric computation
        - Delegates actual metric calculation to `compute_metrics`

    Parameters:
        eval_pred (tuple): A tuple containing:
            - logits (np.ndarray or torch.Tensor): Raw model outputs
            - labels (np.ndarray or torch.Tensor): Ground truth binary labels

    Returns:
        dict: A dictionary of computed evaluation metrics per class, including precision, recall, F1 score, and confusion matrix values.
    """

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

    metric_fn = costume_metric or compute_metrics
    return metric_fn(predictions, labels, labels.shape[1])


def compute_metrics(preds, labels, num_classes):
    """
    Computes precision, recall, F1 score, and confusion matrix components for each class in a multi-label classification setting.

    This function:
        - Iterates through each class index
        - Computes true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN)
        - Calculates precision, recall, and F1 score for each class
        - Stores all metrics in a dictionary keyed by class

    Parameters:
        preds (np.ndarray): A 2D NumPy array of predicted labels (binary values), shape (n_samples, n_classes).
        labels (np.ndarray): A 2D NumPy array of true labels (binary values), shape (n_samples, n_classes).
        num_classes (int): The total number of classes.

    Returns:
        dict: A dictionary containing precision, recall, F1 score, and confusion matrix values for each class.
    """

    class_metrics = {}
    for class_idx in range(num_classes):
        tp = np.sum((preds[:, class_idx] == 1) & (labels[:, class_idx] == 1))
        fp = np.sum((preds[:, class_idx] == 1) & (labels[:, class_idx] == 0))
        tn = np.sum((preds[:, class_idx] == 0) & (labels[:, class_idx] == 0))
        fn = np.sum((preds[:, class_idx] == 0) & (labels[:, class_idx] == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        class_metrics[f"class_{class_idx}_true_positives"] = tp
        class_metrics[f"class_{class_idx}_false_positives"] = fp
        class_metrics[f"class_{class_idx}_true_negatives"] = tn
        class_metrics[f"class_{class_idx}_false_negatives"] = fn
        class_metrics[f"class_{class_idx}_precision"] = precision
        class_metrics[f"class_{class_idx}_recall"] = recall
        class_metrics[f"class_{class_idx}_f1"] = f1

    return class_metrics


def copy_all_files(src_folder, dst_folder):
    """
    Recursively copies all files and directories from a source folder to a destination folder.

    This function:
        - Creates the destination folder if it doesn't exist
        - Traverses the source folder recursively
        - Recreates the directory structure in the destination
        - Copies each file from source to the corresponding destination path using `shutil.copy2` (preserving metadata)

    Parameters:
        src_folder (str): The source directory to copy files from.
        dst_folder (str): The destination directory to copy files to.

    Returns:
        None
    """

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for root, dirs, files in os.walk(src_folder):
        dest_path = os.path.join(dst_folder, os.path.relpath(root, src_folder))
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dest_path, file)
            shutil.copy2(src_file, dst_file)


def recursively_delete_all_files(path):
    """
    Recursively deletes all files and directories at the specified path.

    This function:
        - Checks if the given path exists
        - Attempts to delete the entire directory tree using `shutil.rmtree`
        - Logs a success message if deletion is successful
        - Logs an error message if an exception occurs during deletion

    Parameters:
        path (str): The path to the directory to delete.

    Returns:
        None
    """

    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            logging.info(f"Folder '{path}' deleted successfully.")
        except Exception as e:
            logging.info(f"Error while deleting folder '{path}' - error message -> {e}")


def find_function_callable(candidates, function_name):
    """
    Finds and returns a callable function from a list of candidates by matching its name.

    This function:
        - Searches through a list of candidate objects for a function with a name matching `function_name`
        - Returns the matching function if found
        - Raises an error if no matching function is found or if the candidate list is empty

    Parameters:
        candidates (list): A list of function objects to search through.
        function_name (str): The name of the function to find.

    Returns:
        Callable: The matched function object.

    Raises:
        ValueError: If the candidate list is empty or the specified function is not found in the list.
    """

    if not candidates:
        raise ValueError("No weighted loss strategies provided.")

    for candidate in candidates:
        if (
            isinstance(candidate, types.FunctionType)
            and candidate.__name__ == function_name
        ):
            return candidate

    raise ValueError(
        f"The costume function ({function_name}) was not passed! The path to the function needs to be added in for the argument 'weighted_loss_strategies'."
    )


def positive_value_int(v):
    """
    Validates that a given argument value is a positive integer.

    This function:
        - Attempts to convert the input to an integer
        - Raises an error if the value is negative

    Parameters:
        v (str): The input value to validate.

    Returns:
        int: The validated positive integer value.

    Raises:
        argparse.ArgumentTypeError: If the input value is negative.
    """

    casted_value = int(v)
    if casted_value < 0:
        raise argparse.ArgumentTypeError(f"{v} is not a positive integer")
    return casted_value


def positive_value_float(v):
    """
    Validates that a given argument value is a positive float.

    This function:
        - Attempts to convert the input to a float
        - Raises an error if the value is negative

    Parameters:
        v (str): The input value to validate.

    Returns:
        float: The validated positive float value.

    Raises:
        argparse.ArgumentTypeError: If the input value is negative.
    """

    casted_value = float(v)
    if casted_value < 0:
        raise argparse.ArgumentTypeError(f"{v} is not a positive float")
    return casted_value


def select_training_samples(dataset, sample_size=None, seed=None):
    """Return the full dataset or a random subset of its rows.

    Args:
        dataset: A Hugging Face dataset supporting ``len`` and ``select``.
        sample_size (int, optional): Maximum number of rows to return. If provided,
            rows are shuffled before selection. If omitted, the
            original dataset is returned.
        seed (int, optional): Shuffle seed. Pass a fixed value for reproducible subsets.

    Returns:
        The original dataset or a selected subset of it.
    """

    if sample_size is None:
        return dataset

    return dataset.shuffle(seed=seed).select(range(min(sample_size, len(dataset))))


device = torch.device(
    "mps"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built()
    else "cuda:0" if torch.cuda.is_available() else "cpu"
)


def get_device():
    """
    Detects and returns the most suitable computation device available.

    This function:
        - Prefers Apple's Metal Performance Shaders (MPS) if available and built
        - Falls back to CUDA (GPU) if available
        - Defaults to CPU if neither MPS nor CUDA is available

    Returns:
        torch.device: The selected device for computation (e.g., "mps", "cuda:0", or "cpu").
    """

    return device


DEFAULT_FAMO_CONFIG = {
    "gamma": 0.01,
    "w_lr": 0.025,
    "max_norm": 1.0,
}

DEFAULT_COBA_CONFIG = {
    "coba_warmup_steps": 100,
    "coba_tau": 5,
    "coba_history_length": 200,
    "coba_update_interval": 1,
    "coba_sample_valid_num": 1,
}


def validate_hs_parameters(costume_arguments, TA):
    dict_valid_parameters = TA.__init__.__annotations__
    for key in costume_arguments:
        if key not in dict_valid_parameters:
            raise ValueError(
                f"Type mismatch for '{key}': expected is not a valid parameter. Run stopped!"
            )


def validate_config(passed, default):
    """
    Validates a configuration dictionary by checking for type consistency and unexpected keys.

    This function:
        - Ensures each key in the `default` config exists in the `passed` config with the correct type
        - Uses default values for any missing keys in `passed`
        - Raises an error if a key in `passed` does not exist in `default`
        - Raises an error if a value in `passed` has a type mismatch with the corresponding value in `default`

    Parameters:
        passed (dict): The user-provided configuration to validate.
        default (dict): The reference configuration with default values and expected types.

    Returns:
        dict: A validated configuration dictionary with defaults filled in where necessary.

    Raises:
        ValueError: If there are unknown keys in `passed` or type mismatches between `passed` and `default`.
    """

    validated = {}

    for key, default_val in default.items():
        if key in passed:
            if not isinstance(passed[key], type(default_val)):
                raise ValueError(
                    f"Type mismatch for '{key}': expected {type(default_val).__name__}, got {type(passed[key]).__name__}. Run stopped!"
                )
            validated[key] = passed[key]
        else:
            validated[key] = default_val

    for key in passed:
        if key not in default:
            raise ValueError(f"Error: Unknown config key '{key}'. Run stopped!")

    return validated


def parse_args():
    """
    Parses command-line arguments for configuring training and evaluation of a model.

    This function:
        - Sets up an argument parser with a variety of options related to model training, data paths, loss weighting strategies, and runtime behavior.
        - Supports single or multiple values for strategies, hyperparameters, and configurations.
        - Includes support for boolean and JSON-style inputs using custom parsers (`str2bool`, `str2json`, etc.).

    Returns:
        argparse.ArgumentParser: The argument parser instance with all configured arguments.

    Expected Arguments:
        --generate-base-config (bool): Whether to generate a base config file. Default is False.
        --config-path (str): Optional path to a configuration file.
        --weighted-loss-strategy (list[str]): One or more loss strategies or callable paths. Default is ["EW", "test_function.py"].
        --batch-size (list[int]): One or more batch sizes to test. Default is [32].
        --epoch (list[int]): One or more epoch values. Default is [5].
        --lr (list[float]): One or more learning rates. Default is [1e-2].
        --weight-decay (list[float]): One or more weight decay values. Default is [0].
        --train (str): Path to the training dataset. (Required)
        --val (str): Path to the validation dataset. Optional.
        --test (str): Path to the test dataset. (Required)
        --column-key-text (str): Column name for text in the dataset. (Required)
        --column-key-label (str): Column name for labels in the dataset. (Required)
        --save-model-instances (str): Whether to save the last or all model instances. Choices: ["last", "all"]. Default is "last".
        --model-path (str): Path or identifier for the pre-trained model. Default is "sentence-transformers/paraphrase-MiniLM-L6-v2".
        --result-path (str): Path where results and logs will be saved. Default is ".".
        --sample-size (int): Optionally train on the first N samples instead of the full training set. Default is None.
        --maximise-model (bool): Whether to perform exhaustive search and save the best model. Default is False.
        --freeze-embeddings (bool): Whether to freeze embeddings during training. Default is False.
        --famo-config (dict): Configuration for the FAMO dynamic weighting strategy. Default is `DEFAULT_FAMO_CONFIG`.
        --coba-config (dict): Configuration for the COBA dynamic weighting strategy. Default is `DEFAULT_COBA_CONFIG`.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--generate-base-config", type=str2bool, default=False)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument(
        "--weighted-loss-strategy",
        nargs="*",
        type=str,
        default=["EW"],
    )
    parser.add_argument(
        "--batch-size", nargs="*", type=positive_value_int, default=[32]
    )
    parser.add_argument("--epoch", nargs="*", type=positive_value_int, default=[5])
    parser.add_argument("--lr", nargs="*", type=positive_value_float, default=[1e-2])
    parser.add_argument(
        "--weight-decay", nargs="*", type=positive_value_float, default=[0]
    )
    parser.add_argument("--train", required=True, type=str)
    parser.add_argument("--val", default=None, type=str)
    parser.add_argument("--test", required=True, type=str)
    parser.add_argument("--column-key-text", type=str, required=True)
    parser.add_argument("--column-key-label", type=str, required=True)
    parser.add_argument(
        "--save-model-instances", default="last", type=str, choices=["last", "all"]
    )
    parser.add_argument(
        "--model-path",
        default="sentence-transformers/paraphrase-MiniLM-L6-v2",
        type=str,
    )
    parser.add_argument("--costume-metrics", type=str, default=None)
    parser.add_argument("--costume-whitelist-setfit", nargs="*", type=str, default=None)
    parser.add_argument(
        "--costume-whitelist-roberta", nargs="*", type=str, default=None
    )
    parser.add_argument("--result-path", type=str, default=".")
    parser.add_argument(
        "--sample-size",
        type=positive_value_int,
        default=None,
        help="Optionally train on the first N samples instead of the full training set.",
    )
    parser.add_argument("--maximise-model", type=str2bool, default=False)
    parser.add_argument("--freeze-embeddings", type=str2bool, default=False)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--famo-config", type=str2json, default=DEFAULT_FAMO_CONFIG)
    parser.add_argument("--coba-config", type=str2json, default=DEFAULT_COBA_CONFIG)
    parser.add_argument("--setfit-parameters", type=str2json, default={})
    parser.add_argument("--roberta-parameters", type=str2json, default={})

    return parser


def isDynamic(v):
    """
    Checks whether a given weighting strategy is considered dynamic.

    Parameters:
        v (str): The name of the weighting strategy.

    Returns:
        bool: True if the strategy is dynamic (i.e., "FAMO" or "COBA"), otherwise False.
    """

    return v in ["FAMO", "COBA"]
