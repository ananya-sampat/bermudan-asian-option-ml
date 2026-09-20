import numpy as np

from simulation import simulate_paths

def basis(stock_prices, strike, degree=2):
    x = np.asarray(stock_prices) / strike
    return np.column_stack([x**power for power in range(degree + 1)])


def train_lsm_put(paths, strike, rate, maturity, degree = 2):
    """
    Learn a put exercise policy on dates 1 through steps - 1.

    Returns one regression per date and an in-sample diagnostic value.
    Time-zero exercise is not included here.
    """
    paths = np.asarray(paths, dtype=float)

    if paths.ndim != 2 or paths.shape[0] < 3 or paths.shape[1] < 3:
        raise ValueError("Need at least 3 paths and 2 time steps.")

    if strike <= 0 or maturity <= 0 or np.any(paths <= 0):
        raise ValueError("Strike, maturity, and stock prices must be positive.")

    n_paths, n_dates = paths.shape
    steps = n_dates - 1
    dt = maturity / steps

    # Initially, every path waits until expiration.
    cashflow = np.maximum(strike - paths[:, -1], 0)
    exercise_time = np.full(n_paths, steps, dtype=int)

    models = {}

    for t in range(steps - 1, 0, -1):
        stock_now = paths[:, t]
        indices = np.flatnonzero(stock_now < strike)

        # Not enough observations for our three regression coefficients:
        # keep the existing later-exercise policy at this date.
        if len(indices) < degree + 1:
            continue

        X = basis(stock_now[indices], strike, degree=degree)

        # Discount each selected future payoff back to this date.
        y = cashflow[indices] * np.exp(
            -rate * (exercise_time[indices] - t) * dt
        )

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        models[t] = beta

        continuation = X @ beta
        immediate = strike - stock_now[indices]

        exercise_now = immediate > continuation
        chosen = indices[exercise_now]

        # Earlier exercise replaces the later payoff; it doesn't add to it.
        cashflow[chosen] = immediate[exercise_now]
        exercise_time[chosen] = t

    discounted_cashflows = cashflow * np.exp(
        -rate * exercise_time * dt
    )

    return models, discounted_cashflows.mean()
def evaluate_lsm_put(
    paths, models, strike, rate, maturity, return_payoffs=False
):
    """Evaluate a fixed policy on independent paths."""
    paths = np.asarray(paths, dtype=float)

    n_paths, n_dates = paths.shape
    steps = n_dates - 1
    dt = maturity / steps

    # Default outcome: hold until expiration.
    cashflow = np.maximum(strike - paths[:, -1], 0)
    exercise_time = np.full(n_paths, steps, dtype=int)
    alive = np.ones(n_paths, dtype=bool)

    for t in range(1, steps):
        if t not in models:
            continue

        stock_now = paths[:, t]

        # Only consider options that haven't been exercised.
        indices = np.flatnonzero(alive & (stock_now < strike))

        if len(indices) == 0:
            continue

        degree = len(models[t]) - 1
        X = basis(stock_now[indices], strike, degree=degree)

        # Predict using the saved model—no fitting on test data.
        continuation = X @ models[t]
        immediate = strike - stock_now[indices]

        exercise_now = immediate > continuation
        chosen = indices[exercise_now]

        cashflow[chosen] = immediate[exercise_now]
        exercise_time[chosen] = t
        alive[chosen] = False

    discounted_cashflows = cashflow * np.exp(
        -rate * exercise_time * dt
    )

    estimate = discounted_cashflows.mean()
    se = discounted_cashflows.std(ddof=1) / np.sqrt(n_paths)
    early_exercise_fraction = np.mean(exercise_time < steps)

    if return_payoffs:
        return estimate, se, early_exercise_fraction, discounted_cashflows

    return estimate, se, early_exercise_fraction

if __name__ == "__main__":
    from binomial import crr_price

    training_paths = simulate_paths(
        stock_price=100,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        steps=50,
        paths=20_000,
        seed=42,
    )

    models, training_value = train_lsm_put(
        training_paths,
        strike=100,
        rate=0.05,
        maturity=1.0,
    )

    benchmark = crr_price(
        stock_price=100,
        strike=100,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        steps=500,
        option_type="put",
        american=True,
    )

    print(f"Fitted exercise-date models: {len(models)}")
    print(f"LSM training value:          ${training_value:.6f}")
    print(f"500-step American tree:     ${benchmark:.6f}")

    test_paths = simulate_paths(
        stock_price=100,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        steps=50,
        paths=50_000,
        seed=123,  # Different from the training seed.
    )

    test_value, se, early_fraction = evaluate_lsm_put(
        test_paths,
        models,
        strike=100,
        rate=0.05,
        maturity=1.0,
    )

    print(f"\nIndependent test value:     ${test_value:.6f}")
    print(f"Standard error:             ${se:.6f}")
    print(
        f"Approx. 95% interval:       "
        f"[{test_value - 1.96 * se:.6f}, "
        f"{test_value + 1.96 * se:.6f}]"
    )
    print(f"Early exercise fraction:    {early_fraction:.2%}")