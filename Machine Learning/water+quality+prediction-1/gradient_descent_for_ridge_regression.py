import numpy as np
import scipy.io as sio


class RidgeLoss:
    def __init__(self, lam=0.1):
        self.lam = lam  # L2 regularization strength

    def compute_loss(self, y_pred, y_true, w):
        mse = np.mean((y_pred - y_true) ** 2)
        penalty = self.lam * np.sum(w ** 2)
        return mse + penalty

    def compute_grad(self, X, y_pred, y_true, w):
        n = X.shape[0]
        dw = 2 * X.T @ (y_pred - y_true) / n + 2 * self.lam * w
        db = 2 * np.sum(y_pred - y_true) / n
        return dw, db


class GradientDescent:
    def __init__(self, lr=0.02):
        self.lr = lr

    def step(self, w, b, dw, db):
        w = w - self.lr * dw
        b = b - self.lr * db
        return w, b


if __name__ == "__main__":
    data = sio.loadmat('water_dataset.mat')
    X_tr = data['X_tr']   # (1, 423) every element is (37, 11) matrix
    print(X_tr.shape)
    Y_tr = data['Y_tr']   # (37, 423)
    print(Y_tr.shape)

    Xs = np.stack([X_tr[0, t] for t in range(X_tr.shape[1])])  # (423, 37, 11)
    X = Xs[:, :, 1].reshape(-1, 1)   # design matrix (n, 1)
    y = Y_tr.T.ravel()

    w = np.zeros(X.shape[1])
    b = 0.0
    loss_function = RidgeLoss(lam=0.1)
    opt = GradientDescent(lr=0.02)

    for i in range(50):
        y_pred = X @ w + b
        loss = loss_function.compute_loss(y_pred, y, w)
        dw, db = loss_function.compute_grad(X, y_pred, y, w)
        w, b = opt.step(w, b, dw, db)

        if i % 10 == 0:
            print(f"iteration{i:3d} | loss={loss:.3f} | w={w[0]:.2f}, b={b:.2f}")

    print(f"\nfinish：w={w[0]:.2f}, b={b:.2f}")
