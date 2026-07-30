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
    "regression_linear": lambda: _gen_regression_linear(),
    "regression_multitask": lambda: _gen_regression_multitask(),
}


def _gen_regression_linear(n=442, noise=0.1, seed=42):
    """Easy single-output regression: y = 3*x1 + 2*x2 - x3 + noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 10)
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] - 1.0 * X[:, 2] + 0.5 * X[:, 3]
    y += noise * rng.randn(n)
    return type('obj', (object,), {'data': X, 'target': y})


def _gen_regression_multitask(n=442, noise=0.05, seed=42):
    """Harder 3-output regression with nonlinear relationships."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 8)
    y = np.column_stack([
        2.0 * np.sin(X[:, 0]) + X[:, 1] * X[:, 2],
        X[:, 3] ** 2 - X[:, 4] + 0.5 * X[:, 5],
        1.5 * X[:, 6] + 3.0 * X[:, 7] + 0.1 * X[:, 0] * X[:, 3],
    ])
    y += noise * rng.randn(n, 3)
    return type('obj', (object,), {'data': X, 'target': y})


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
    y = np.asarray(data.target)

    pd.DataFrame(X).to_csv(
        f"{args.directory}/{args.dataset}_data.csv", index=False, header=False
    )
    # Handle single-output (1D) vs multi-output (2D) labels
    if y.ndim == 1:
        pd.Series(y, name="label").to_csv(
            f"{args.directory}/{args.dataset}_label.csv", index=False, header=False
        )
    else:
        pd.DataFrame(y).to_csv(
            f"{args.directory}/{args.dataset}_label.csv", index=False, header=False
        )


if __name__ == "__main__":
    main()
