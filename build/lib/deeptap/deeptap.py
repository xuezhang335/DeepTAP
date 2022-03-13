from .deeptap_main import deeptap_main
from .parse_args import commandLineParser


def main():
    args = commandLineParser()
    print(args)
    deeptap_main(args)


if __name__ == "__main__":
    main()
