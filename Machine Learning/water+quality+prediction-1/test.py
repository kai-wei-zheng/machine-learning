import numpy as np
import scipy.io as sio

class MSELoss:
    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

    def compute_grad(self, X, y_pred, y_true):
        n = X.shape[0]
        dw = 2 * X.T @ (y_pred - y_true) / n
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
    Y_tr = data['Y_tr']   # (37, 423)

    Xs = np.stack([X_tr[0, t] for t in range(X_tr.shape[1])])  # (423, 37, 11)
    X = Xs[:, :, 1].reshape(-1, 1)   # design matrix (n, 1)
    y = Y_tr.T.ravel()

    w = np.zeros(X.shape[1])
    b = 0.0
    loss_function = MSELoss()
    opt = GradientDescent(lr=0.02)

    for i in range(50):
        y_pred = X @ w + b
        loss = loss_function.compute_loss(y_pred, y)
        dw, db = loss_function.compute_grad(X, y_pred, y)
        w, b = opt.step(w, b, dw, db)

        if i % 10 == 0:
            print(f"iteration{i:3d} | loss={loss:.3f} | w={w[0]:.2f}, b={b:.2f}")

    print(f"\nfinish：w={w[0]:.2f}, b={b:.2f}")
