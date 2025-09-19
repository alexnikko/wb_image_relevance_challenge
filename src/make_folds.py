import argparse
from .utils import make_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True, type=str)
    ap.add_argument("--n_splits", default=5, type=int)
    args = ap.parse_args()
    make_folds(args.train_csv, args.n_splits)


if __name__ == "__main__":
    main()