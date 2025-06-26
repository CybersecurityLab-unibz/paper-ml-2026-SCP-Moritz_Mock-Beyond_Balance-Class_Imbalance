import types
from itertools import product
import logging
from nlbse_jex.utils import parse_args
from nlbse_jex.utils import read_files_weighted_loss_strategy
from nlbse_jex.utils import generate_base_config
from nlbse_jex.utils import read_config
from nlbse_jex.proxy import Proxy

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
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


    args = read_files_weighted_loss_strategy(args)

    logging.info(args)

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

        logging.info(f"Execution number {index + 1} out of {len(combinations)}")


        p = Proxy(args)
        p.create_model()

        if not p.run_already_exists():
            p.create_trainer()
            p.train()
            p.eval()
        else:
            logging.warning("--------------------")
            logging.warning(f"Execution with the arguments {args} already exists!")
            logging.warning(f"")
            logging.warning(f"Run was skipped!")
            logging.warning("--------------------")

        #p.create_model("/leonardo/home/userexternal/mmock000/.cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/c9a2bfebc254878aee8c3aca9e6844d5bbb102d1")




if __name__ == "__main__":
    main()
