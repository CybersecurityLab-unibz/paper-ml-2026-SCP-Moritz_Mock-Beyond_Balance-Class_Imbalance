import types
from nlbse_jex.utils import parse_args
from nlbse_jex.utils import read_files_weighted_loss
from nlbse_jex.utils import generate_base_config
from nlbse_jex.utils import read_config
from setfit import SetFitModel, Trainer, TrainingArguments, SetFitTrainer
from datasets import Dataset, load_dataset
import numpy as np
from nlbse_jex.proxy import Proxy
import logging

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

def main():
    args = parse_args()
    if args.generate_config_hyper_parameter_search is True:
        path = "config.yml"
        generate_base_config(path)
        print(f"Example configuration file was generated under the path {path}")
        quit()

    if args.config is not None:
        args = read_config(args)

    args = read_files_weighted_loss(args)

    for wl in args.weighted_loss:
        if isinstance(wl, str):
            print(wl)
        if isinstance(wl, types.FunctionType):
            wl("moritz")

    p = Proxy(args)
    p.create_model()
    #p.create_trainer()
    #p.train()
    #p.eval()

    #p.create_model("/leonardo/home/userexternal/mmock000/.cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/c9a2bfebc254878aee8c3aca9e6844d5bbb102d1")




if __name__ == "__main__":
    main()
