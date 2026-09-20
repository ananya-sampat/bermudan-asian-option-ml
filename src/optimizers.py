"""From-scratch optimizers for 0.5*MSE + 0.5*lambda*||w_nonintercept||^2."""

from time import perf_counter
import numpy as np


def objective(w, x, y, lam):
    r = x @ w - y
    return 0.5 * np.mean(r * r) + 0.5 * lam * np.dot(w[1:], w[1:])


def gradient(w, x, y, lam):
    g = x.T @ (x @ w - y) / len(y)
    g[1:] += lam * w[1:]
    return g


def closed_form(x, y, lam):
    penalty = np.eye(x.shape[1]) * lam
    penalty[0, 0] = 0
    return np.linalg.lstsq(x.T @ x / len(y) + penalty, x.T @ y / len(y), rcond=None)[0]


def optimize(x, y, method="gd", lr=0.03, lam=0.001, epochs=300, batch=256, seed=123):
    w = np.zeros(x.shape[1])
    velocity = w.copy()
    rng = np.random.default_rng(seed)
    history = []
    start = perf_counter()
    updates = 0
    for epoch in range(epochs):
        order = np.arange(len(y)) if method == "gd" else rng.permutation(len(y))
        size = len(y) if method == "gd" else batch
        for i in range(0, len(y), size):
            ix = order[i : i + size]
            point = w + 0.9 * velocity if method == "nesterov" else w
            g = gradient(point, x[ix], y[ix], lam)
            velocity = (
                0.9 * velocity if method in ("momentum", "nesterov") else 0
            ) - lr * g
            w += velocity
            updates += 1
            if not np.isfinite(w).all() or np.max(np.abs(w)) > 1e30:
                break
        loss = objective(w, x, y, lam)
        history.append(
            {
                "epoch": epoch + 1,
                "updates": updates,
                "seconds": perf_counter() - start,
                "objective": float(loss),
            }
        )
        if not np.isfinite(loss) or loss > 1e50:
            break
    return w, history
