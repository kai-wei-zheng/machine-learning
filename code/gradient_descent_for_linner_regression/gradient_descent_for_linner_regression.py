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


class OptLinearRegression:
    def __init__(self, lr=0.01, epochs=5000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        self.loss_function = MSELoss()
        self.opt = GradientDescent(lr=self.lr)

    def optimal(self, X, y):
        n, d = X.shape
        print(f"sample: n={n}, attribute: d={d}")

        # Feature Standardization
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        self.w = np.zeros(d)
        self.b = 0.0

        for i in range(self.epochs):
            y_pred = X @ self.w + self.b
            loss = self.loss_function.compute_loss(y_pred, y)
            dw, db = self.loss_function.compute_grad(X, y_pred, y)
            self.w, self.b = self.opt.step(self.w, self.b, dw, db)

            if i % 50 == 0:
                print(f"iteration{i:3d} | loss={loss:.4f}")

        final_loss = self.loss_function.compute_loss(X @ self.w + self.b, y)
        print(f"finish: loss={final_loss:.4f}, b={self.b:.4f}")

    def predict(self, X):
        return X @ self.w + self.b

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

    model = OptLinearRegression(lr=0.01, epochs=5000)
    model.optimal(X, y)