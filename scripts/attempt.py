import numpy as np


class SNN_BPTT:
    def __init__(self, sizes, tau=100, dt=1.0, v_th=1.0, v_reset=0.0, lr=0.01):
        self.sizes, self.tau, self.dt, self.v_th, self.v_reset = (
            sizes,
            tau,
            dt,
            v_th,
            v_reset,
        )
        self.alpha = np.exp(-dt / tau)
        self.beta = 1 - self.alpha
        self.lr = lr
        self.W = [
            np.random.randn(sizes[i], sizes[i + 1]) * 0.1 for i in range(len(sizes) - 1)
        ]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def _surrogate(self, V):
        return np.exp(-np.abs(V - self.v_th))

    def forward(self, X):
        N, T, D_in = X.shape
        self.S = [np.zeros((N, T, self.sizes[l])) for l in range(len(self.sizes))]
        self.V = [np.zeros((N, T, self.sizes[l])) for l in range(len(self.sizes))]

        self.S[0][:] = X
        for l in range(len(self.W)):
            for t in range(T):
                self.S[l + 1][:, t] = self.S[l][:, t] @ self.W[l] + self.b[l]
                self.V[l + 1][:, t] = (
                    self.alpha * self.V[l + 1][:, t - 1]
                    + self.beta * self.S[l + 1][:, t]
                )
                self.S[l + 1][:, t] = (self.V[l + 1][:, t] >= self.v_th).astype(float)
                self.V[l + 1][:, t] = np.where(
                    self.S[l + 1][:, t] > 0, self.v_reset, self.V[l + 1][:, t]
                )
        return self.S[-1].sum(axis=1)

    def backward(self, y_true):
        N, T = y_true.shape[0], self.S[0].shape[1]
        y_pred = self.S[-1].sum(axis=1)

        dV = np.zeros((N, T, self.sizes[-1]))
        dV[:, -1] = ((y_pred - y_true) / N) * self._surrogate(self.V[-1][:, -1])

        gW = [np.zeros_like(w) for w in self.W]
        gb = [np.zeros_like(b) for b in self.b]

        for l in range(len(self.W) - 1, -1, -1):
            dI = np.zeros((N, T, self.sizes[l + 1]))
            dV_prev = np.zeros((N, T, self.sizes[l]))
            for t in range(T - 1, -1, -1):
                dI[:, t] = dV[:, t] * self.beta
                dV[:, t] *= self.alpha * self._surrogate(self.V[l][:, t])
                gW[l] += np.einsum("ni,nj->ij", self.S[l][:, t], dI[:, t])
                gb[l] += dI[:, t].sum(axis=0)
                dV_prev[:, t] = dI[:, t] @ self.W[l].T
            dV = dV_prev

        for l in range(len(self.W)):
            self.W[l] -= self.lr * gW[l]
            self.b[l] -= self.lr * gb[l]

    def train(self, X, y, epochs=100):
        for _ in range(epochs):
            self.forward(X)
            self.backward(y)
        return self.S[-1].sum(axis=1)


# Synthetic Dataset
np.random.seed(42)
N, T, D = 200, 20, 10
X = np.random.randint(0, 2, (N, T, D)).astype(float)
y = (
    np.mean(X[:, : T // 2], axis=(1, 2)) > np.mean(X[:, T // 2 :], axis=(1, 2))
).astype(float)

model = SNN_BPTT([D, 16, 8, 1], tau=50, dt=1.0, lr=0.005)
preds = model.train(X, y, epochs=300)
print(f"Accuracy: {np.mean((preds > 0.5).astype(int) == y):.2f}")
