from argparse import ArgumentParser
import importlib.util
import os
import ast
from typing import get_origin, get_args, Union
import inspect
import yaml

def read_files_weighted_loss(args):
    args_ok = {
        "name": "placeholder"
    }

    for index, wl in enumerate(args.weighted_loss):
        if wl.endswith(".py"):
            function = import_function_from_path(wl)
            ok, mismatches = can_pass_args_with_types(function, args_ok)
            assert ok is True, "\n".join(mismatches)
            args.weighted_loss[index] = function

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
                mismatches.append(f"Unexpected argument '{name}'")
            continue

        expected_type = param.annotation
        if expected_type is not inspect.Parameter.empty:
            if not _is_instance_of(value, expected_type):
                mismatches.append(
                    f"Argument '{name}' has wrong type: expected {expected_type}, got {type(value)}"
                )

    # Check for missing required args
    for p in params.values():
        if (
            p.default is inspect.Parameter.empty and
            p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and
            p.name not in args_dict
        ):
            mismatches.append(f"Missing required argument '{p.name}'")

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
        'epochs': [1, 2, 5, 10],
        'weighted_loss': ['EW', 'FAMO', 'test_function.py']
    }

    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def read_config(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    for key, value in config.items():
        setattr(args, key, value)

    return args


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--generate-config-hyper-parameter-search", default=False, type=str2bool)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--weighted-loss", nargs="*", type=str, default=["EW"])
    parser.add_argument("--train", default=".", type=str)
    parser.add_argument("--val", default=".", type=str)
    parser.add_argument("--test", default=".", type=str)
    parser.add_argument("--save-model-instances", default="all", type=str, choices=["best", "top-5", "top-10", "all"])
    parser.add_argument("--save-model-path", default=".", type=str)

    return parser.parse_args()
