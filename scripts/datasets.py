#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_iris,
    load_wine,
    load_breast_cancer,
    load_digits,
    fetch_openml,
)

LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
    "mnist": lambda: fetch_openml("mnist_784", version=1, parser="auto"),
    "fashion_mnist": lambda: fetch_openml("Fashion-MNIST", version=1, parser="auto"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=LOADERS.keys())
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=".",
        help="Destination directory for files",
    )
    args = parser.parse_args()

    data = LOADERS[args.dataset]()
    X = (
        np.asarray(data.data.toarray())
        if hasattr(data.data, "toarray")
        else np.asarray(data.data)
    )
    y = pd.to_numeric(np.asarray(data.target))

    pd.DataFrame(X).to_csv(
        f"{args.directory}/{args.dataset}_data.csv", index=False, header=False
    )
    pd.Series(y, name="label").to_csv(
        f"{args.directory}/{args.dataset}_label.csv", index=False, header=False
    )


if __name__ == "__main__":
    main()
