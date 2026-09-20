"""Risk-neutral GBM and discrete-date optimal stopping; no market-data trading."""

from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np


@dataclass(frozen=True)
class Contract:
    spot: float = 100.0
    strike: float = 100.0
    rate: float = 0.05
    volatility: float = 0.20
    maturity: float = 1.0
    steps: int = 50
    kind: str = "asian"


# These are the user's original modules, imported directly rather than copied.
from simulation import simulate_paths
from asians_payoff import running_average, asian_put_payoffs
from asian_lsm import train_asian_put, asian_basis
from longstaff_schwartz import train_lsm_put, basis


def simulate(c, n, seed):
    return np.asarray(
        simulate_paths(
            stock_price=c.spot,
            rate=c.rate,
            volatility=c.volatility,
            maturity=c.maturity,
            steps=c.steps,
            paths=n,
            seed=seed,
        )
    )


def averages(paths):
    return running_average(paths)


class SavedOriginalPolynomial:
    def __init__(self, beta, kind, average_only=False):
        self.beta = beta
        self.kind = kind
        self.average_only = average_only

    def predict(self, x):
        if self.kind == "vanilla":
            X = basis(x[:, 0], 1.0, degree=len(self.beta) - 1)
        elif self.average_only:
            X = asian_basis(x[:, 0], x[:, 0], 1.0, average_only=True)
        else:
            X = asian_basis(x[:, 0], x[:, 1], 1.0)
        return X @ self.beta


def state_and_payoff(paths, c):
    a = averages(paths)
    payoff = asian_put_payoffs(a if c.kind == "asian" else paths, c.strike)
    return a, payoff


def features(paths, a, t, c, average_only=False):
    if c.kind == "vanilla":
        return paths[:, t, None] / c.strike
    if average_only:
        return a[:, t, None] / c.strike
    return np.column_stack([paths[:, t], a[:, t]]) / c.strike


def train_policy(paths, c, spec, seed=0, average_only=False):
    from ml_models import make_model

    start = perf_counter()
    if spec["kind"] == "quadratic":
        kwargs = dict(strike=c.strike, rate=c.rate, maturity=c.maturity)
        if c.kind == "asian":
            coefficients = train_asian_put(paths, **kwargs, average_only=average_only)
        else:
            coefficients, _ = train_lsm_put(paths, **kwargs, degree=2)
        return {
            "models": {
                t: SavedOriginalPolynomial(beta, c.kind, average_only)
                for t, beta in coefficients.items()
            },
            "average_only": average_only,
            "seconds": perf_counter() - start,
            "spec": spec,
        }
    a, p = state_and_payoff(paths, c)
    cash = p[:, -1].copy()
    times = np.full(len(paths), c.steps, dtype=int)
    models = {}
    for t in range(c.steps - 1, 0, -1):
        idx = np.flatnonzero(p[:, t] > 0)
        if len(idx) < 20:
            continue
        x = features(paths, a, t, c, average_only)[idx]
        y = cash[idx] * np.exp(-c.rate * (times[idx] - t) * c.maturity / c.steps)
        m = make_model(spec, seed + t)
        m.fit(x, y)
        models[t] = m
        chosen = idx[p[idx, t] > m.predict(x)]
        cash[chosen] = p[chosen, t]
        times[chosen] = t
    return {
        "models": models,
        "average_only": average_only,
        "seconds": perf_counter() - start,
        "spec": spec,
    }


def evaluate_policy(paths, c, policy, start_step=1):
    a, p = state_and_payoff(paths, c)
    cash = p[:, -1].copy()
    times = np.full(len(paths), c.steps, dtype=int)
    alive = np.ones(len(paths), bool)
    for t in range(start_step, c.steps):
        if t not in policy["models"]:
            continue
        idx = np.flatnonzero(alive & (p[:, t] > 0))
        if not len(idx):
            continue
        x = features(paths, a, t, c, policy["average_only"])[idx]
        chosen = idx[p[idx, t] > policy["models"][t].predict(x)]
        cash[chosen] = p[chosen, t]
        times[chosen] = t
        alive[chosen] = False
    return cash * np.exp(-c.rate * times * c.maturity / c.steps), times


def fixed_targets(paths, c, teacher, t=25):
    """All methods share realized future payoffs under an independently fit teacher."""
    a, p = state_and_payoff(paths, c)
    idx = np.flatnonzero(p[:, t] > 0)
    # Starting t+1 forces continuation at t and excludes its exercise decision.
    discounted, _ = evaluate_policy(paths, c, teacher, start_step=t + 1)
    return (
        features(paths, a, t, c)[idx],
        discounted[idx] * np.exp(c.rate * t * c.maturity / c.steps),
        p[idx, t],
    )


def hold_payoffs(paths, c):
    return state_and_payoff(paths, c)[1][:, -1] * np.exp(-c.rate * c.maturity)


def summary(x):
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return {
        "mean": mean,
        "se": se,
        "ci_low": mean - 1.96 * se,
        "ci_high": mean + 1.96 * se,
    }


def tree_price(c, tree_steps=500):
    if c.kind != "vanilla" or tree_steps % c.steps:
        raise ValueError("Vanilla and aligned grid required")
    dt = c.maturity / tree_steps
    u = np.exp(c.volatility * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(c.rate * dt) - d) / (u - d)
    if not 0 < q < 1:
        raise ValueError("Invalid tree probability")
    j = np.arange(tree_steps + 1)
    v = np.maximum(c.strike - c.spot * u**j * d ** (tree_steps - j), 0)
    for i in range(tree_steps - 1, -1, -1):
        v = np.exp(-c.rate * dt) * (q * v[1:] + (1 - q) * v[:-1])
        # Future dates only: same as learned policies, no time-zero exercise.
        if i > 0 and i % (tree_steps // c.steps) == 0:
            j = np.arange(i + 1)
            v = np.maximum(v, c.strike - c.spot * u**j * d ** (i - j))
    return float(v[0])
