"""Compare the two local linear-regression implementations fairly.

Both runners use the same water_dataset training/test data and identical
learning rate, epoch count, initial weights, and no progress printing.
"""

import argparse
import statistics
import time

import numpy as np
import scipy.io as sio

from gradient_descent_for_linear_regression import LinearRegression as MatrixLR
from test import GradientDescent as TestGD
from test import MSELoss


DATA_PATH = "water_dataset.mat"
FEATURE_INDEX = 1


def load_water_data(path=DATA_PATH):
    data = sio.loadmat(path)

    X_tr = np.stack(
        [data["X_tr"][0, t] for t in range(data["X_tr"].shape[1])]
    )
    X_te = np.stack(
        [data["X_te"][0, t] for t in range(data["X_te"].shape[1])]
    )

    x_train = X_tr[:, :, FEATURE_INDEX].ravel()
    x_test = X_te[:, :, FEATURE_INDEX].ravel()
    y_train = data["Y_tr"].T.ravel()
    y_test = data["Y_te"].T.ravel()

    return x_train, y_train, x_test, y_test


def run_test(x, y, lr, epochs):
    loss_fn = MSELoss()
    optimizer = TestGD(lr=lr)
    X = x.reshape(-1, 1)
    w = np.zeros(X.shape[1])
    b = 0.0

    for _ in range(epochs):
        y_pred = X @ w + b
        dw, db = loss_fn.compute_grad(X, y_pred, y)
        w, b = optimizer.step(w, b, dw, db)

    return w, b


def run_matrix(x, y, lr, epochs):
    model = MatrixLR(x.reshape(-1, 1), y)
    w = np.zeros(model.d)
    b = 0.0

    for _ in range(epochs):
        dw, db = model.compute_gradient(w, b)
        w = w - lr * dw
        b = b - lr * db

    return w, b


def predict_test(w, b, x):
    return x.reshape(-1, 1) @ w + b


def predict_matrix(w, b, x):
    return x.reshape(-1, 1) @ w + b


def mse(pred, y):
    return float(np.mean(np.square(pred - y)))


def measure_runner(runner, predictor, x_train, y_train, lr, epochs, repeats):
    runner(x_train, y_train, lr, epochs)

    runtimes = []
    for _ in range(repeats):
        start = time.perf_counter()
        w, b = runner(x_train, y_train, lr, epochs)
        runtimes.append(time.perf_counter() - start)

    train_pred = predictor(w, b, x_train)
    return w, b, runtimes, mse(train_pred, y_train)


def print_report(
    name,
    runner,
    predictor,
    x_train,
    y_train,
    x_test,
    y_test,
    lr,
    epochs,
    repeats,
):
    w, b, runtimes, train_mse = measure_runner(
        runner, predictor, x_train, y_train, lr, epochs, repeats
    )

    test_mse = mse(predictor(w, b, x_test), y_test)
    best_time = min(runtimes)
    median_time = statistics.median(runtimes)

    print(f"--- {name} ---")
    print(f"  best time     : {best_time:.6f} s")
    print(f"  median time   : {median_time:.6f} s")
    print(f"  per epoch     : {best_time / epochs:.9f} s")
    print(f"  train MSE     : {train_mse:.9f}")
    print(f"  test MSE      : {test_mse:.9f}")
    print(f"  fitted w, b   : {float(np.asarray(w).item()):.9f}, "
          f"{float(np.asarray(b).item()):.9f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare the two vectorized linear-regression implementations."
    )
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    x_train, y_train, x_test, y_test = load_water_data()

    print(f"Data: {DATA_PATH}")
    print(f"Samples: train={x_train.size}, test={x_test.size}")
    print(f"Settings: lr={args.lr}, epochs={args.epochs}, "
          f"repeats={args.repeats}\n")

    print_report(
        "test.py vector gradient descent",
        run_test,
        predict_test,
        x_train,
        y_train,
        x_test,
        y_test,
        args.lr,
        args.epochs,
        args.repeats,
    )
    print()
    print_report(
        "gradient_descent_for_linear_regression.py matrix version",
        run_matrix,
        predict_matrix,
        x_train,
        y_train,
        x_test,
        y_test,
        args.lr,
        args.epochs,
        args.repeats,
    )


if __name__ == "__main__":
    main()
