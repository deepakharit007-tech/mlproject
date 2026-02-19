"""Run ModelTrainer with synthetic data and print only the results dict.

Usage:
    python3 run_trainer.py

This script tries to use scikit-learn to generate regression data. If not
available, it falls back to a simple numpy-generated linear dataset. The
script prints exactly one line: the result dictionary returned by
`ModelTrainer.initiate_model_trainer`.
"""

import sys
import numpy as np

try:
    from sklearn.datasets import make_regression
    SKLEARN_MAKE = True
except Exception:
    SKLEARN_MAKE = False

from src.components.model_trainer import ModelTrainer


def make_data(n_samples=500, n_features=5, noise=0.1):
    if SKLEARN_MAKE:
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, random_state=42)
    else:
        rng = np.random.RandomState(42)
        X = rng.randn(n_samples, n_features)
        coef = rng.randn(n_features)
        y = X.dot(coef) + noise * rng.randn(n_samples)
    return X, y


def to_train_test_arrays(X, y, test_size=0.2):
    n_test = int(len(X) * test_size)
    if n_test < 1:
        n_test = 1
    X_train, y_train = X[:-n_test], y[:-n_test]
    X_test, y_test = X[-n_test:], y[-n_test:]
    # train_array/test_array expected shape: (n_samples, n_features+1) with last col target
    train_array = np.hstack([X_train, y_train.reshape(-1, 1)])
    test_array = np.hstack([X_test, y_test.reshape(-1, 1)])
    return train_array, test_array


def main():
    X, y = make_data()
    train_array, test_array = to_train_test_arrays(X, y)

    trainer = ModelTrainer()
    try:
        result = trainer.initiate_model_trainer(train_array=train_array, test_array=test_array)
        # print only the results dict (single line)
        print(result)
    except Exception as e:
        # Print exception message to stderr and exit non-zero
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
