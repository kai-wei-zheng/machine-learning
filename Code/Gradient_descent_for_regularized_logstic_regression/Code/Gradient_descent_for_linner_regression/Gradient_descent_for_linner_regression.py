import re
import numpy as np
import pandas as pd


class MSELoss:
    def compute_loss(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))

    def compute_grad(self, X, y_pred, y_true):
        n = X.shape[0]
        dw = 2.0 * X.T @ (y_pred - y_true) / n
        db = 2.0 * np.sum(y_pred - y_true) / n
        return dw, db


class GradientDescent:
    def __init__(self, lr=0.02):
        self.lr = lr

    def step(self, w, b, dw, db):
        w -= self.lr * dw
        b -= self.lr * db
        return w, b

 
if __name__ == "__main__":
    # Parse 128 column names
    names = []
    with open("communities.names", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r"@attribute\s+(\S+)", line)
            if m:
                names.append(m.group(1))

    df = pd.read_csv("communities.data", names=names, na_values="?")

    # Remove 5 metadata columns
    df = df.drop(columns=["state", "county", "community", "communityname", "fold"])

    # target column
    y = df["ViolentCrimesPerPop"].to_numpy()
    df = df.drop(columns=["ViolentCrimesPerPop"])


    df = df.dropna(axis=1)
    feature_names = df.columns.tolist()
    X = df.to_numpy()

    n, d = X.shape
    print(f"sample: n={n}, attribute: d={d}")

    # Feature Standardization
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    w = np.zeros(d)
    b = 0.0
    loss_function = MSELoss()
    opt = GradientDescent(lr=0.01)

    for i in range(5000):
        y_pred = X @ w + b
        loss = loss_function.compute_loss(y_pred, y)
        dw, db = loss_function.compute_grad(X, y_pred, y)
        w, b = opt.step(w, b, dw, db)

        if i % 50 == 0:
            print(f"iteration{i:3d} | loss={loss:.4f}")

    final_loss = loss_function.compute_loss(X @ w + b, y)
    print(f"finish: loss={final_loss:.4f}, b={b:.4f}")

    print("w:")
    for name, w in zip(feature_names, w):
        print(f"  {name:24s} = {w:+.4f}")
