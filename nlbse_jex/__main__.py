import types
from itertools import product
import logging
from nlbse_jex.utils import parse_args
from nlbse_jex.utils import read_files_weighted_loss_strategy
from nlbse_jex.utils import read_file_compute_metric
from nlbse_jex.utils import generate_base_config
from nlbse_jex.utils import read_config
from nlbse_jex.proxy import Proxy
from nlbse_jex.utils import validate_config
from nlbse_jex.utils import DEFAULT_FAMO_CONFIG
from nlbse_jex.utils import DEFAULT_COBA_CONFIG

logging.basicConfig(
    format="%(asctime)s - %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)


def main():
    parser = parse_args()
    args = parser.parse_args()

    if args.generate_base_config is True:
        path = "config.yml"
        generate_base_config(path)
        print(f"Example configuration file was generated under the path {path}")
        quit()

    if args.config_path is not None:
        args = read_config(args, parser)
        logging.info(args)

    args.famo_config = validate_config(args.famo_config, DEFAULT_FAMO_CONFIG)
    args.coba_config = validate_config(args.coba_config, DEFAULT_COBA_CONFIG)

    args = read_files_weighted_loss_strategy(args)
    args = read_file_compute_metric(args)

    logging.info(args)

    if args.maximise_model:
        p = Proxy(args)
        p.create_model()
        p.model.merge_dataset()
        p.create_trainer()
        p.train()
        quit()

    epoch = args.epoch
    lr = args.lr
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    weighted_loss_strategy = args.weighted_loss_strategy
    arrays = [epoch, lr, batch_size, weight_decay, weighted_loss_strategy]
    combinations = list(product(*arrays))

    args.weighted_loss_strategy = args.weighted_loss_strategy[0]

    for index, combination in enumerate(combinations):
        args.epoch = combination[0]
        args.lr = combination[1]
        args.batch_size = combination[2]
        args.weight_decay = combination[3]
        args.weighted_loss_strategy = combination[4]

        logging.info(args)

        logging.info(f"Execution number {index + 1} out of {len(combinations)}")

        p = Proxy(args)
        p.create_model()

        if p.run_already_exists() is False:
            p.create_trainer()
            p.train()
            p.eval()
        elif p.run_already_exists() is True:
            p.model.delete_path()
            logging.warning("--------------------")
            logging.warning(f"Execution with the arguments {args} already exists!")
            logging.warning(f"")
            logging.warning(f"Run was skipped!")
            logging.warning("--------------------")
        else:
            logging.error("--------------------")
            logging.error("Model type is not whitelisted!")
            logging.error("Execution of the run was interrupted.")
            logging.error("--------------------")

        # p.create_model("/leonardo/home/userexternal/mmock000/.cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/c9a2bfebc254878aee8c3aca9e6844d5bbb102d1")


if __name__ == "__main__":
    main()
