import types
from nlbse_jex.utils import parse_args
from nlbse_jex.utils import read_files_weighted_loss

def main():
    args = parse_args()
    args = read_files_weighted_loss(args)


    for wl in args.weighted_loss:
        if isinstance(wl, str):
            print(wl)
        if isinstance(wl, types.FunctionType):
            wl("moritz")

    print(args.weighted_loss)
if __name__ == "__main__":
    main()
