"""https://introml.mit.edu/notes/regression.html"""
import numpy as np
from ucimlrepo import fetch_ucirepo


class LinearRegression:
    def __init__(self, X, y):
        self.X = X  # design matrix
        self.y = y  # target class
        self.n = y.shape[0]  # number of datapoints
        self.d = X.shape[1]  # number of feature dimensions

    def compute_loss(self, w, b):
        y_pred = self.X @ w + b
        mse = np.mean(np.square(y_pred - self.y))
        return mse

    def compute_gradient(self, w, b):
        y_pred = self.X @ w + b
        d_w = (2.0 / self.n) * self.X.T @ (y_pred - self.y)  # derivatives of weights
        d_b = (2.0 / self.n) * np.sum(y_pred - self.y)  # derivative of the bias
        return d_w, d_b


class GradientDescent:
    def __init__(self, learning_rate=0.03, epochs=2000):
        self.lr = learning_rate
        self.epochs = epochs

    def optimize(self, loss_func):
        w = np.zeros(loss_func.d)  # weights of linear regression model
        b = 0.0  # bias of linear regression model

        for epoch in range(self.epochs):
            d_w, d_b = loss_func.compute_gradient(w, b)
            w = w - self.lr * d_w
            b = b - self.lr * d_b

            if epoch % 200 == 0:
                loss = loss_func.compute_loss(w, b)
                print(f"Epoch {epoch:5d} | Loss = {loss:.4f}")

        return w, b


if __name__ == "__main__":
    real_estate_valuation = fetch_ucirepo(id=477)

    X_raw = real_estate_valuation.data.features.values
    y_raw = real_estate_valuation.data.targets.values.ravel()

    print("Dataset name:", real_estate_valuation.metadata.name)
    print("Number of samples:", X_raw.shape[0])
    print("Number of features:", X_raw.shape[1])

    X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)

    loss = LinearRegression(X, y_raw)
    optimizer = GradientDescent(learning_rate=0.03, epochs=50000)
    w_opt, b_opt = optimizer.optimize(loss)

    print("=== Training Finished ===")
    print("Optimal weight w:", w_opt)
    print("Optimal bias b:", b_opt)
    final_loss = loss.compute_loss(w_opt, b_opt)
    print("Final Loss:", final_loss)

    # === 拟合结果输出 ===
    y_pred = loss.X @ w_opt + b_opt
    ss_res = np.sum(np.square(y_pred - loss.y))
    ss_tot = np.sum(np.square(loss.y - np.mean(loss.y)))
    r2 = 1.0 - ss_res / ss_tot
    print("R-squared :", r2)
