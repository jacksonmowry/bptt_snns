#!/usr/bin/env python3

"""Per-neuron BPTT for a global leaky integrate-and-fire SNN."""

from __future__ import annotations

import argparse
from typing import Iterable, Tuple
from urllib.request import urlopen
import sys

import numpy as np

IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    probs = softmax(logits)
    eps = 1e-8
    loss = -np.mean(np.sum(targets * np.log(probs + eps), axis=1))
    grad = (probs - targets) / targets.shape[0]
    return float(loss), grad


def load_iris_raw() -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    with urlopen(IRIS_URL, timeout=30) as response:
        for raw_line in response.read().decode("utf-8").splitlines():
            line = raw_line.strip()
            if line:
                rows.append(line)

    features = []
    labels = []
    label_map = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2,
    }
    for row in rows:
        parts = row.split(",")
        if len(parts) != 5:
            continue
        features.append([float(value) for value in parts[:4]])
        labels.append(label_map[parts[4]])

    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def train_test_split_stratified(
    X: np.ndarray, y: np.ndarray, test_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        split = int(round(len(cls_idx) * (1.0 - test_fraction)))
        train_idx.extend(cls_idx[:split])
        test_idx.extend(cls_idx[split:])
    train_idx = np.array(train_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def min_max_normalize(
    train: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    min_v = train.min(axis=0)
    max_v = train.max(axis=0)
    scale = np.where(max_v > min_v, max_v - min_v, 1.0)
    return (train - min_v) / scale, (test - min_v) / scale


def rate_encode(
    X: np.ndarray, time_steps: int, max_rate: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.clip(X, 0.0, 1.0)
    rates = X[:, None, :] * max_rate
    samples = rng.random((X.shape[0], time_steps, X.shape[1]))
    return (samples < rates).astype(np.float32)


def leak_alpha_from_tau(tau: float, dt: float) -> float:
    if np.isinf(tau):
        return 1.0
    if tau <= 0.0:
        return 0.0
    return float(np.exp(-dt / tau))


class GlobalGraphSNN:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        tau: float = 8.0,
        dt: float = 1.0,
        threshold: float = 1.0,
        reset: float = 0.0,
        minimum_potential: float = -1.0,
        lr: float = 1e-2,
        seed: int = 0,
    ) -> None:
        if input_size <= 0 or hidden_size < 0 or output_size <= 0:
            raise ValueError("input_size, hidden_size, and output_size must be valid")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_nodes = input_size + hidden_size + output_size
        self.output_nodes = list(range(input_size + hidden_size, self.num_nodes))
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.step_count = 0
        self.reset = reset
        self.minimum_potential = minimum_potential
        self.alpha = leak_alpha_from_tau(tau, dt)
        self.delay_min = 1
        self.delay_max = 7
        self.rng = np.random.default_rng(seed)
        self.thresholds = np.clip(
            self.rng.normal(loc=threshold, scale=0.05, size=self.num_nodes),
            0.1,
            None,
        ).astype(np.float32)

        self.weights = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        self.delays = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int32)
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.incoming = [[] for _ in range(self.num_nodes)]

        self.L2 = 0.0

        self._initialize_global_graph(seed)

    def _initialize_global_graph(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        scale = 0.5 / max(1.0, np.sqrt(self.num_nodes))
        for target in range(self.num_nodes):
            for source in range(self.num_nodes):
                if rng.random() < 0.75:
                    self.weights[source, target] = rng.normal(0.0, scale)
                    self.delays[source, target] = rng.integers(
                        self.delay_min, self.delay_max + 1
                    )
                    self.incoming[target].append(source)

    def spike_surrogate(self, u: np.ndarray, threshold: float) -> np.ndarray:
        return 1.0 / np.square(1.0 + np.abs(u - threshold))

    def forward(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch_size, steps, _ = X.shape
        spikes = np.zeros((batch_size, steps, self.num_nodes), dtype=np.float32)
        v_pre = np.zeros_like(spikes)
        y = np.zeros_like(spikes)
        v = np.zeros((batch_size, self.num_nodes), dtype=np.float32)

        for t in range(steps):
            for node in range(self.num_nodes):
                if node < self.input_size:
                    spikes[:, t, node] = X[:, t, node]

                v = self.alpha * v
                v = np.maximum(v, self.minimum_potential)

                for source in self.incoming[node]:
                    delay = self.delays[source, node]
                    source_time = t - delay
                    if source_time >= 0:
                        v = (
                            v
                            + spikes[:, source_time, source]
                            * self.weights[source, node]
                        )

                v_pre[:, t, node] = v
                spike = (v >= self.threshold[node]).astype(np.float32)
                v = np.where(s > 0.0, self.reset, v)
                v = np.maximum(v, self.minimum_potential)
                y[:, t, node] = v
                spikes[:t, node] = spike

        logits = spikes[:, :, self.output_nodes].mean(axis=1)
        return logits, spikes, v_pre, y

    def backward(
        self,
        y_one_hot: np.ndarray,
        spikes: np.ndarray,
        v_pre: np.ndarray,
        y: np.ndarray,
        logits: np.ndarray,
        w_scale: float,
        tau_rho: float,
        scale_rho: float,
    ) -> float:
        batch_size, steps, _ = spikes.shape
        loss, dlogits = cross_entropy(logits, y_one_hot)
        dL_dy = dlogits / float(steps)

        grad_weights = np.zeros_like(self.weights)

        tau_rho_scaled = tau_rho * w_scale

        dL_dV_next = np.zeros((batch_size, self.num_nodes), dtype=np.float32)

        for t in range(steps - 1, -1, -1):
            for node in range(self.num_nodes - 1, -1, -1):
                dL_dV = dL_dy[node]
                dL_dV = dL_dV + dL_dV_next[node]

                v_pre_t = v_pre[t, idx]

                # Pipeline 1
                dy_t_dV_post = 1.0 / w_scale
                dV_post_dV_pre = 1.0 - (spikes[t, node])
                dV_pre_dx_t = w_scale
                dV_post_ds_t = (
                    ((self.min_potential - v_pre_t) * w_scale)
                    if (self.min_potential * w_scale > 0)
                    else (-v_pre_t * w_scale)
                )
                ds_t_dV_pre = (scale_rho / (2.0 * tau_rho_scaled)) * np.exp(
                    -np.abs(w_scale * (v_pre_t - self.threshold[node])) / tau_rho_scaled
                )

                # Pipeline 2
                dV_pre_dV_leak = 1.0
                dV_leak_dV_t1 = (
                    (1.0 - self.alpha)
                    if (v_pre_t * w_scale >= self.min_potential * w_scale)
                    else (0.0)
                )

        # spike_grad_history = np.zeros(
        #     (batch_size, steps, self.num_nodes), dtype=np.float32
        # )

        # for t in range(steps - 1, -1, -1):
        #     same_time_grad = spike_grad_history[:, t, :].copy()
        #     for output_offset, node in enumerate(self.output_nodes):
        #         same_time_grad[:, node] += dL_dy[:, output_offset]

        #     next_dL_dV_next = np.zeros_like(dL_dV_next)

        #     for node in range(self.num_nodes - 1, -1, -1):
        #         s = spikes[:, t, node]
        #         u = pre_acts[:, t, node]
        #         ds_du = self.spike_surrogate(u, self.thresholds[node])
        #         g_u = (
        #             dL_dV_next[:, node] * (1.0 - s)
        #             + same_time_grad[:, node] * ds_du
        #         )

        #         for source in self.incoming[node]:
        #             delay = self.delays[source, node]
        #             source_time = t - delay
        #             if source_time < 0:
        #                 continue
        #             source_spikes = spikes[:, source_time, source]
        #             grad_weights[source, node] += np.sum(source_spikes * g_u)
        #             spike_grad_history[:, source_time, source] += (
        #                 g_u * self.weights[source, node]
        #             )

        #         leaked_mem = self.alpha * mem_history[:, t, node]
        #         leak_mask = (leaked_mem > self.minimum_potential).astype(np.float32)
        #         next_dL_dV_next[:, node] = self.alpha * g_u * leak_mask

        #     dL_dV_next = next_dL_dV_next

        self._apply_gradients(grad_weights)
        return loss

    def _apply_gradients(self, grad_weights: np.ndarray) -> None:
        self.step_count += 1
        clipped_w = np.clip(grad_weights, -5.0, 5.0)

        self.m_weights = self.beta1 * self.m_weights + (1.0 - self.beta1) * clipped_w
        self.v_weights = self.beta2 * self.v_weights + (1.0 - self.beta2) * (
            clipped_w * clipped_w
        )

        mW_hat = self.m_weights / (1.0 - self.beta1**self.step_count)
        vW_hat = self.v_weights / (1.0 - self.beta2**self.step_count)

        self.L2 += np.sum((self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)) ** 2)
        self.weights -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)

    def train_batch(self, X: np.ndarray, y_one_hot: np.ndarray) -> float:
        logits, spikes, v_pre, y = self.forward(X)
        out = self.backward(y_one_hot, spikes, v_pre, y, logits)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits, _, _, _ = self.forward(X)
        return np.argmax(logits, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


def batch_iter(
    X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    indices = rng.permutation(X.shape[0])
    for start in range(0, X.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


def train_model(
    model: GlobalGraphSNN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batches = 0
        for xb, yb in batch_iter(X_train, y_train, batch_size, rng):
            total_loss += model.train_batch(xb, yb)
            batches += 1
        if True or epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            train_acc = model.score(X_train, np.argmax(y_train, axis=1))
            test_acc = model.score(X_test, np.argmax(y_test, axis=1))
            print(
                f"epoch={epoch:03d} loss={total_loss / max(1, batches):.4f} "
                f"train_acc={train_acc:.3f} test_acc={test_acc:.3f}"
            )
        print(f"L2: {model.L2}")
        model.L2 = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-neuron BPTT for a global-matrix spiking neural network"
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=8e-3)
    parser.add_argument(
        "--tau",
        type=float,
        default=8.0,
        help="Membrane time constant; use inf for no leak or 0 for full leak.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--time-steps", type=int, default=32)
    parser.add_argument("--max-rate", type=float, default=0.35)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument(
        "--minimum-potential",
        type=float,
        default=-1.0,
        help="Lowest membrane potential allowed after leak at the start of each timestep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    X_raw, y = load_iris_raw()
    X_train_raw, X_test_raw, y_train_int, y_test_int = train_test_split_stratified(
        X_raw, y, test_fraction=0.2, seed=args.seed
    )
    X_train_raw, X_test_raw = min_max_normalize(X_train_raw, X_test_raw)

    X_train = rate_encode(X_train_raw, args.time_steps, args.max_rate, seed=args.seed)
    X_test = rate_encode(X_test_raw, args.time_steps, args.max_rate, seed=args.seed + 1)
    y_train = one_hot(y_train_int, 3)
    y_test = one_hot(y_test_int, 3)

    model = GlobalGraphSNN(
        input_size=X_train.shape[2],
        hidden_size=args.hidden_size,
        output_size=3,
        tau=args.tau,
        minimum_potential=args.minimum_potential,
        lr=args.lr,
        seed=args.seed,
    )

    print(
        "dataset=iris "
        f"train={X_train.shape[0]} test={X_test.shape[0]} "
        f"steps={X_train.shape[1]} input_neurons={X_train.shape[2]} "
        f"hidden_neurons={args.hidden_size} output_neurons=3"
    )
    print(
        f"leak_alpha={model.alpha:.4f} tau={args.tau} "
        f"minimum_potential={model.minimum_potential:.3f}"
    )
    active_delays = model.delays[model.delays > 0]
    if active_delays.size:
        print(
            f"synaptic_delays=random_integer_range=[{active_delays.min()}, {active_delays.max()}]"
        )
    if np.isinf(args.tau):
        print("tau=inf corresponds to no leak (alpha=1.0)")
    elif args.tau <= 0.0:
        print("tau=0 corresponds to full leak (alpha=0.0)")
    print(f"initial_test_acc={model.score(X_test, y_test_int):.3f}")
    train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"final_test_acc={model.score(X_test, y_test_int):.3f}")


if __name__ == "__main__":
    main()
