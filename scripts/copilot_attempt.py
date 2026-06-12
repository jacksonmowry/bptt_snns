#!/usr/bin/env python3

"""NumPy BPTT for a multilayer leaky integrate-and-fire SNN."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, np.ndarray]:
    probs = softmax(logits)
    eps = 1e-8
    loss = -np.mean(np.log(probs[np.arange(labels.shape[0]), labels] + eps))
    grad = (probs - one_hot(labels, logits.shape[1])) / labels.shape[0]
    return float(loss), grad


@dataclass
class LIFLayer:
    in_features: int
    out_features: int
    tau: float
    dt: float = 1.0
    threshold: float = 1.0
    reset: float = 0.0
    weight_scale: float = 0.5
    seed: int = 0

    def __post_init__(self) -> None:
        self.alpha = float(np.exp(-self.dt / self.tau))
        self.rng = np.random.default_rng(self.seed)
        scale = self.weight_scale / max(1.0, np.sqrt(self.in_features))
        self.W = self.rng.normal(0.0, scale, size=(self.in_features, self.out_features)).astype(
            np.float32
        )
        self.b = np.zeros(self.out_features, dtype=np.float32)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def spike_surrogate(self, u: np.ndarray) -> np.ndarray:
        # Smooth derivative around the threshold for surrogate-gradient BPTT.
        return 1.0 / np.square(1.0 + np.abs(u - self.threshold))


class BPTTSNNClassifier:
    def __init__(
        self,
        sizes: Sequence[int],
        tau: float = 8.0,
        dt: float = 1.0,
        threshold: float = 1.0,
        reset: float = 0.0,
        lr: float = 1e-2,
        seed: int = 0,
    ) -> None:
        if len(sizes) < 2:
            raise ValueError("sizes must contain at least input and output dimensions")

        self.sizes = list(sizes)
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.step_count = 0
        self.layers: List[LIFLayer] = []
        for i in range(len(sizes) - 1):
            self.layers.append(
                LIFLayer(
                    in_features=sizes[i],
                    out_features=sizes[i + 1],
                    tau=tau,
                    dt=dt,
                    threshold=threshold,
                    reset=reset,
                    seed=seed + i,
                )
            )

    def forward(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        batch_size, steps, _ = X.shape
        layer_inputs: List[np.ndarray] = [
            np.zeros((batch_size, steps, layer.in_features), dtype=np.float32)
            for layer in self.layers
        ]
        pre_activations: List[np.ndarray] = [
            np.zeros((batch_size, steps, layer.out_features), dtype=np.float32)
            for layer in self.layers
        ]
        spike_traces: List[np.ndarray] = [
            np.zeros((batch_size, steps, layer.out_features), dtype=np.float32)
            for layer in self.layers
        ]
        membrane_state = [
            np.zeros((batch_size, layer.out_features), dtype=np.float32) for layer in self.layers
        ]

        for t in range(steps):
            current = X[:, t, :]
            for layer_index, layer in enumerate(self.layers):
                layer_inputs[layer_index][:, t, :] = current
                u = layer.alpha * membrane_state[layer_index] + current @ layer.W + layer.b
                s = (u >= layer.threshold).astype(np.float32)
                v = np.where(s > 0.0, layer.reset, u)
                pre_activations[layer_index][:, t, :] = u
                spike_traces[layer_index][:, t, :] = s
                membrane_state[layer_index] = v
                current = s

        logits = spike_traces[-1].mean(axis=1)
        return logits, layer_inputs, pre_activations, spike_traces

    def backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        layer_inputs: List[np.ndarray],
        pre_activations: List[np.ndarray],
        spike_traces: List[np.ndarray],
        logits: np.ndarray,
    ) -> float:
        batch_size, steps, _ = X.shape
        loss, dlogits = cross_entropy(logits, y)
        d_spike_out = dlogits / float(steps)

        grad_W = [np.zeros_like(layer.W) for layer in self.layers]
        grad_b = [np.zeros_like(layer.b) for layer in self.layers]
        carry_v = [
            np.zeros((batch_size, layer.out_features), dtype=np.float32) for layer in self.layers
        ]

        for t in range(steps - 1, -1, -1):
            same_time = [np.zeros_like(carry_v[idx]) for idx in range(len(self.layers))]
            same_time[-1] = d_spike_out

            for layer_index in range(len(self.layers) - 1, -1, -1):
                layer = self.layers[layer_index]
                u = pre_activations[layer_index][:, t, :]
                s = spike_traces[layer_index][:, t, :]
                ds_du = layer.spike_surrogate(u)

                g_u = carry_v[layer_index] * (1.0 - s) + same_time[layer_index] * ds_du
                grad_W[layer_index] += layer_inputs[layer_index][:, t, :].T @ g_u
                grad_b[layer_index] += np.sum(g_u, axis=0)
                carry_v[layer_index] = layer.alpha * g_u

                if layer_index > 0:
                    same_time[layer_index - 1] += g_u @ layer.W.T

        self._apply_gradients(grad_W, grad_b)
        return loss

    def _apply_gradients(self, grad_W: List[np.ndarray], grad_b: List[np.ndarray]) -> None:
        self.step_count += 1
        for idx, layer in enumerate(self.layers):
            gW = np.clip(grad_W[idx], -5.0, 5.0)
            gb = np.clip(grad_b[idx], -5.0, 5.0)

            layer.mW = self.beta1 * layer.mW + (1.0 - self.beta1) * gW
            layer.vW = self.beta2 * layer.vW + (1.0 - self.beta2) * (gW * gW)
            layer.mb = self.beta1 * layer.mb + (1.0 - self.beta1) * gb
            layer.vb = self.beta2 * layer.vb + (1.0 - self.beta2) * (gb * gb)

            mW_hat = layer.mW / (1.0 - self.beta1**self.step_count)
            vW_hat = layer.vW / (1.0 - self.beta2**self.step_count)
            mb_hat = layer.mb / (1.0 - self.beta1**self.step_count)
            vb_hat = layer.vb / (1.0 - self.beta2**self.step_count)

            layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

    def train_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        logits, layer_inputs, pre_activations, spike_traces = self.forward(X)
        return self.backward(X, y, layer_inputs, pre_activations, spike_traces, logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits, _, _, _ = self.forward(X)
        return np.argmax(logits, axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


def make_temporal_burst_dataset(
    samples: int,
    time_steps: int,
    features: int,
    noise_rate: float = 0.03,
    burst_len: int = 4,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = (rng.random((samples, time_steps, features)) < noise_rate).astype(np.float32)
    y = rng.integers(0, 2, size=samples, dtype=np.int64)

    pattern = np.array([1, 0, 1, 1, 0, 1], dtype=np.float32)
    if features > len(pattern):
        pattern = np.pad(pattern, (0, features - len(pattern)))
    else:
        pattern = pattern[:features]

    early_center = max(1, time_steps // 4)
    late_center = min(time_steps - burst_len - 1, 3 * time_steps // 4)

    for i, label in enumerate(y):
        center = early_center if label == 0 else late_center
        start = int(np.clip(center + rng.integers(-1, 2), 0, time_steps - burst_len))
        X[i, start : start + burst_len] = np.maximum(
            X[i, start : start + burst_len],
            pattern[None, :],
        )

    return X.astype(np.float32), y


def batch_iter(
    X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator
) -> Sequence[Tuple[np.ndarray, np.ndarray]]:
    indices = rng.permutation(X.shape[0])
    for start in range(0, X.shape[0], batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


def train_model(
    model: BPTTSNNClassifier,
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
        epoch_loss = 0.0
        batches = 0
        for xb, yb in batch_iter(X_train, y_train, batch_size, rng):
            epoch_loss += model.train_batch(xb, yb)
            batches += 1
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            print(
                f"epoch={epoch:03d} loss={epoch_loss / max(1, batches):.4f} "
                f"train_acc={train_acc:.3f} test_acc={test_acc:.3f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backpropagation through time for a NumPy spiking neural network"
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--tau", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--time-steps", type=int, default=24)
    parser.add_argument("--features", type=int, default=6)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[32, 16],
        help="Hidden layer sizes for the fully connected spiking network.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    X, y = make_temporal_burst_dataset(
        samples=args.samples,
        time_steps=args.time_steps,
        features=args.features,
        seed=args.seed,
    )

    split = int(0.8 * args.samples)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    sizes = [args.features, *args.hidden_sizes, 2]
    model = BPTTSNNClassifier(
        sizes=sizes,
        tau=args.tau,
        lr=args.lr,
        seed=args.seed,
    )

    print(
        "dataset="
        f"{X.shape[0]} samples, {X.shape[1]} steps, {X.shape[2]} features; "
        f"network={sizes}"
    )
    print(f"leak_alpha={model.layers[0].alpha:.4f} tau={args.tau}")
    print(f"initial_test_acc={model.score(X_test, y_test):.3f}")
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
    print(f"final_test_acc={model.score(X_test, y_test):.3f}")


if __name__ == "__main__":
    main()
