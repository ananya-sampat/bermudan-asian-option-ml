import numpy as np

from simulation import simulate_paths
from longstaff_schwartz import (
    basis,
    train_lsm_put,
    evaluate_lsm_put,
)


def fit_ridge(X, y, penalty):
    """
    Minimize mean squared error + penalty * squared coefficients.
    Standardize nonconstant features; don't penalize the intercept.
    """
    features = X[:, 1:]

    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-12] = 1.0

    standardized = (features - center) / scale

    Z = np.column_stack([
        np.ones(len(y)),
        standardized,
    ])

    # Ridge can be solved as an augmented least-squares problem.
    regularizer = np.eye(Z.shape[1])
    regularizer[0, 0] = 0.0

    augmented_X = np.vstack([
        Z,
        np.sqrt(len(y) * penalty) * regularizer,
    ])
    augmented_y = np.concatenate([
        y,
        np.zeros(Z.shape[1]),
    ])

    fitted = np.linalg.lstsq(
        augmented_X, augmented_y, rcond=None
    )[0]

    # Convert back so predictions can use the original X directly.
    beta = np.empty_like(fitted)
    beta[1:] = fitted[1:] / scale
    beta[0] = fitted[0] - center @ beta[1:]

    return beta


def train_ridge_put(
    paths, strike, rate, maturity, degree=5, penalty=0.01
):
    if penalty < 0:
        raise ValueError("penalty must be nonnegative.")

    paths = np.asarray(paths, dtype=float)
    n_paths, n_dates = paths.shape
    steps = n_dates - 1
    dt = maturity / steps

    cashflow = np.maximum(strike - paths[:, -1], 0)
    exercise_time = np.full(n_paths, steps, dtype=int)
    models = {}

    for t in range(steps - 1, 0, -1):
        indices = np.flatnonzero(paths[:, t] < strike)

        if len(indices) < degree + 1:
            continue

        X = basis(paths[indices, t], strike, degree=degree)

        y = cashflow[indices] * np.exp(
            -rate * (exercise_time[indices] - t) * dt
        )

        beta = fit_ridge(X, y, penalty)
        models[t] = beta

        continuation = X @ beta
        immediate = strike - paths[indices, t]

        exercise_now = immediate > continuation
        chosen = indices[exercise_now]

        cashflow[chosen] = immediate[exercise_now]
        exercise_time[chosen] = t

    return models


if __name__ == "__main__":
    settings = dict(
        stock_price=100,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        steps=50,
    )

    training = np.array(
        simulate_paths(**settings, paths=20_000, seed=42)
    )
    validation = np.array(
        simulate_paths(**settings, paths=50_000, seed=456)
    )

    evaluation_settings = dict(
        strike=100,
        rate=0.05,
        maturity=1.0,
        return_payoffs=True,
    )

    baseline_models, _ = train_lsm_put(
        training,
        strike=100,
        rate=0.05,
        maturity=1.0,
        degree=2,
    )

    baseline_value, _, _, baseline_payoffs = evaluate_lsm_put(
        validation, baseline_models, **evaluation_settings
    )

    print(f"Quadratic baseline: ${baseline_value:.6f}")
    print("\nDegree-5 ridge models:")
    print("Penalty | Value | Gain vs baseline | Paired 95% interval")

    for penalty in [0.0, 0.0001, 0.01, 1.0]:
        models = train_ridge_put(
            training,
            strike=100,
            rate=0.05,
            maturity=1.0,
            degree=5,
            penalty=penalty,
        )

        value, _, _, payoffs = evaluate_lsm_put(
            validation, models, **evaluation_settings
        )

        differences = payoffs - baseline_payoffs
        gain = differences.mean()
        se = differences.std(ddof=1) / np.sqrt(len(differences))

        print(
            f"{penalty:7g} | ${value:.6f} | "
            f"${gain:+.6f} | "
            f"[{gain - 1.96 * se:.6f}, "
            f"{gain + 1.96 * se:.6f}]"
        )