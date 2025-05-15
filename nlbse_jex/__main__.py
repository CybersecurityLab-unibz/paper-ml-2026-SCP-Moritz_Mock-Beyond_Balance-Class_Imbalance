import types
from nlbse_jex.utils import parse_args
from nlbse_jex.utils import read_files_weighted_loss
from nlbse_jex.utils import generate_base_config
from nlbse_jex.utils import read_config

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

    print(args)
if __name__ == "__main__":
    main()
