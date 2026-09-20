import numpy as np

from simulation import simulate_paths
from asians_payoff import running_average, asian_put_payoffs


def asian_basis(stock, average, strike, average_only=False):
    a = np.asarray(average) / strike

    if average_only:
        return np.column_stack([np.ones_like(a), a, a**2])

    s = np.asarray(stock) / strike

    return np.column_stack([
        np.ones_like(s),
        s,
        a,
        s**2,
        s * a,
        a**2,
    ])

def train_asian_put(
    paths, strike, rate, maturity, average_only=False
):
    paths = np.asarray(paths, dtype=float)
    averages = running_average(paths)

    n_paths, n_dates = paths.shape
    steps = n_dates - 1
    dt = maturity / steps

    cashflow = asian_put_payoffs(averages[:, -1], strike)
    exercise_time = np.full(n_paths, steps, dtype=int)
    models = {}

    for t in range(steps - 1, 0, -1):
        # In the money depends on the average, not current stock price.
        indices = np.flatnonzero(averages[:, t] < strike)

        if len(indices) < 6:
            continue

        X = asian_basis(
            paths[indices, t],
            averages[indices, t],
            strike,
            average_only=average_only,
        )

        y = cashflow[indices] * np.exp(
            -rate * (exercise_time[indices] - t) * dt
        )

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        models[t] = beta

        continuation = X @ beta
        immediate = asian_put_payoffs(averages[indices, t], strike)

        exercise_now = immediate > continuation
        chosen = indices[exercise_now]

        cashflow[chosen] = immediate[exercise_now]
        exercise_time[chosen] = t

    return models


def evaluate_asian_put(paths, models, strike, rate, maturity):
    paths = np.asarray(paths, dtype=float)
    averages = running_average(paths)

    n_paths, n_dates = paths.shape
    steps = n_dates - 1
    dt = maturity / steps

    cashflow = asian_put_payoffs(averages[:, -1], strike)
    exercise_time = np.full(n_paths, steps, dtype=int)
    alive = np.ones(n_paths, dtype=bool)

    for t in range(1, steps):
        if t not in models:
            continue

        indices = np.flatnonzero(
            alive & (averages[:, t] < strike)
        )

        if len(indices) == 0:
            continue

        X = asian_basis(
            paths[indices, t],
            averages[indices, t],
            strike,
            average_only=(len(models[t]) == 3),
        )

        continuation = X @ models[t]
        immediate = asian_put_payoffs(averages[indices, t], strike)

        exercise_now = immediate > continuation
        chosen = indices[exercise_now]

        cashflow[chosen] = immediate[exercise_now]
        exercise_time[chosen] = t
        alive[chosen] = False

    discounted_payoffs = cashflow * np.exp(
        -rate * exercise_time * dt
    )

    return discounted_payoffs, exercise_time


if __name__ == "__main__":
    settings = dict(
        stock_price=100,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        steps=50,
    )
    contract = dict(strike=100, rate=0.05, maturity=1.0)

    training = np.array(
        simulate_paths(**settings, paths=20_000, seed=42)
    )
    validation = np.array(
        simulate_paths(**settings, paths=50_000, seed=456)
    )

    models = train_asian_put(training, **contract)

    payoffs, exercise_times = evaluate_asian_put(
        validation, models, **contract
    )

    # Hold-until-expiration strategy on exactly the same paths.
    validation_averages = running_average(validation)
    european_payoffs = np.exp(-0.05) * asian_put_payoffs(
        validation_averages[:, -1], strike=100
    )

    value = payoffs.mean()
    se = payoffs.std(ddof=1) / np.sqrt(len(payoffs))

    differences = payoffs - european_payoffs
    gain = differences.mean()
    gain_se = differences.std(ddof=1) / np.sqrt(len(differences))

    print(f"Fitted exercise-date models: {len(models)}")
    print(f"European Asian put:          ${european_payoffs.mean():.6f}")
    print(f"Learned early-exercise value: ${value:.6f}")
    print(
        f"Value 95% interval: "
        f"[{value - 1.96 * se:.6f}, {value + 1.96 * se:.6f}]"
    )
    print(f"\nGain over holding: ${gain:.6f}")
    print(
        f"Paired gain 95% interval: "
        f"[{gain - 1.96 * gain_se:.6f}, "
        f"{gain + 1.96 * gain_se:.6f}]"
    )
    print(
        f"Early exercise fraction: "
        f"{np.mean(exercise_times < settings['steps']):.2%}"
    )