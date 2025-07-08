import argparse
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

def read_files_weighted_loss_strategy(args):
    args_ok = {
        "labels": "placeholder"
    }

    for index, wl in enumerate(args.weighted_loss_strategy):
        if wl.endswith(".py"):
            function = import_function_from_path(wl)
            ok, mismatches = can_pass_args_with_types(function, args_ok)
            assert ok is True, "\n".join(mismatches)
            args.weighted_loss_strategy[index] = function

    return args

def can_pass_args_with_types(func, args_dict):
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
            p.default is inspect.Parameter.empty and
            p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and
            p.name not in args_dict
        ):
            mismatches.append(f"Unexpected argument '{p.name}'")

    return len(mismatches) == 0, mismatches

def _is_instance_of(value, expected_type):
    """Support typing types like Union, List, etc."""
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
    with open(file_path, 'r') as f:
        node = ast.parse(f.read(), filename=file_path)

    for n in node.body:
        if isinstance(n, ast.FunctionDef):
            return n.name
    raise ValueError(f"No function definitions found in '{file_path}'")


def import_function_from_path(file_path):
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
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def generate_base_config(path):
    data = {
        'epoch': [1, 2, 5, 10],
        'weighted_loss_strategy': ['EW']
    }

    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


def load_yaml_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def validate_and_cast_config(yaml_config, parser):
    validated_config = {}
    arg_dest_to_action = {action.dest: action for action in parser._actions}

    for key, value in yaml_config.items():
        if key not in arg_dest_to_action:
            raise ValueError(f"Unknown config key: '{key}' (not defined in argparse)")

        action = arg_dest_to_action[key]
        expected_type = action.type
        nargs = action.nargs

        # Handle boolean flags (store_true / store_false)
        if isinstance(action, argparse._StoreTrueAction):
            if isinstance(value, bool):
                validated_config[key] = value
            elif isinstance(value, (int, str)):
                validated_config[key] = str(value).lower() in ['true', '1', 'yes', 'on']
            else:
                raise ValueError(f"Invalid type for key '{key}': expected boolean-like value")

        # Handle nargs / list-type args
        elif nargs is not None:
            if not isinstance(value, list):
                raise ValueError(f"Expected a list for argument '{key}', but got {type(value).__name__}")

            # Cast each element in the list
            if expected_type is not None:
                try:
                    validated_config[key] = [expected_type(v) for v in value]
                except Exception as e:
                    raise ValueError(f"Type error in list for key '{key}': expected elements of type {expected_type.__name__}, got {value}")
            else:
                validated_config[key] = value  # No type specified, accept as-is

        # Handle normal scalar args
        elif expected_type is not None:
            try:
                validated_config[key] = expected_type(value)
            except Exception as e:
                raise ValueError(f"Type error for key '{key}': expected {expected_type.__name__}, got {value} ({type(value).__name__})")
        else:
            validated_config[key] = value

    return validated_config


def merge_config_with_args(yaml_config, cli_args, parser):
    config = vars(cli_args).copy()

    for key, value in yaml_config.items():
        # CLI args take priority
        if getattr(cli_args, key, None) in [None, parser.get_default(key)]:
            config[key] = value

    return config


def read_config(args, parser):
    raw_yaml = load_yaml_config(args.config_path)
    logging.info(raw_yaml)
    yaml_config = validate_and_cast_config(raw_yaml, parser)
    logging.info(raw_yaml)
    return merge_config_with_args(yaml_config, args, parser)


def compute_metrics(preds, labels):
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    elif isinstance(preds, list):
        preds = np.array(preds)

    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)

    num_classes = labels.shape[1]
    class_metrics = {}

    for class_idx in range(num_classes):
        tp = np.sum((preds[:, class_idx] == 1) & (labels[:, class_idx] == 1))
        fp = np.sum((preds[:, class_idx] == 1) & (labels[:, class_idx] == 0))
        tn = np.sum((preds[:, class_idx] == 0) & (labels[:, class_idx] == 0))
        fn = np.sum((preds[:, class_idx] == 0) & (labels[:, class_idx] == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics[f"class_{class_idx}_true_positives"] = tp
        class_metrics[f"class_{class_idx}_false_positives"] = fp
        class_metrics[f"class_{class_idx}_true_negatives"] = tn
        class_metrics[f"class_{class_idx}_false_negatives"] = fn
        class_metrics[f"class_{class_idx}_precision"] = precision
        class_metrics[f"class_{class_idx}_recall"] = recall
        class_metrics[f"class_{class_idx}_f1"] = f1

    return class_metrics


def copy_all_files(src_folder, dst_folder):
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
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            logging.info(f"Folder '{path}' deleted successfully.")
        except Exception as e:
            logging.info(f"Error while deleting folder '{path}' - error message -> {e}")


def find_function_callable(candidates, function_name):
    if not candidates:
        raise ValueError("No weighted loss strategies provided.")

    for candidate in candidates:
        if isinstance(candidate, types.FunctionType) and candidate.__name__ == function_name:
            return candidate

    raise ValueError(f"The costum function ({function_name}) was not passed! The path to the function needs to be added in for the argument 'weighted_loss_strategies'.")




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--generate-base-config", type=str2bool, default=False)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--weighted-loss-strategy", nargs="*", type=str, default=["EW", "test_function.py"])
    parser.add_argument("--batch-size", nargs="*", type=int, default=[32])
    parser.add_argument("--epoch", nargs="*", type=int, default=[5])
    parser.add_argument("--lr", nargs="*", type=float, default=[1e-2])
    parser.add_argument("--weight-decay", nargs="*", type=float, default=[0])
    parser.add_argument("--train", required=True, type=str)
    parser.add_argument("--val", default=None, type=str)
    parser.add_argument("--test", required=True, type=str)
    parser.add_argument("--column-key-text", type=str, required=True)
    parser.add_argument("--column-key-label", type=str, required=True)
    parser.add_argument("--save-model-instances", default="last", type=str, choices=["last", "all"])
    parser.add_argument("--model-path", default="sentence-transformers/paraphrase-MiniLM-L6-v2", type=str)
    parser.add_argument("--result-path", type=str, default=".")
    parser.add_argument("--maximise-model", type=str2bool, default=False)
    parser.add_argument("--freeze-embeddings", type=str2bool, default=False)


    return parser
