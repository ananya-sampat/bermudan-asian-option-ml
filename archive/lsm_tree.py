import numpy as np
from sklearn.tree import DecisionTreeRegressor

from simulation import simulate_paths
from longstaff_schwartz import train_lsm_put, evaluate_lsm_put

from sklearn.ensemble import RandomForestRegressor


def train_tree_put(
    paths,
    strike,
    rate,
    maturity,
    max_depth=4,
    min_samples_leaf=100,
    model_type="tree",
):
    paths = np.asarray(paths, dtype=float)
    n_paths, n_dates = paths.shape
    steps = n_dates - 1
    dt = maturity / steps

    cashflow = np.maximum(strike - paths[:, -1], 0)
    exercise_time = np.full(n_paths, steps, dtype=int)
    models = {}

    for t in range(steps - 1, 0, -1):
        indices = np.flatnonzero(paths[:, t] < strike)

        if len(indices) < 2:
            continue

        # One feature: current stock price relative to strike.
        X = (paths[indices, t] / strike).reshape(-1, 1)

        y = cashflow[indices] * np.exp(
            -rate * (exercise_time[indices] - t) * dt
        )

        if model_type == "tree":
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            )
        elif model_type == "forest":
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                bootstrap=True,
                max_features=1.0,
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError("model_type must be 'tree' or 'forest'.")
        model.fit(X, y)
        models[t] = model

        continuation = model.predict(X)
        immediate = strike - paths[indices, t]

        exercise_now = immediate > continuation
        chosen = indices[exercise_now]

        cashflow[chosen] = immediate[exercise_now]
        exercise_time[chosen] = t



    return models


def evaluate_tree_put(paths, models, strike, rate, maturity):
    """Return one discounted payoff per independent evaluation path."""
    paths = np.asarray(paths, dtype=float)
    n_paths, n_dates = paths.shape
    steps = n_dates - 1
    dt = maturity / steps

    cashflow = np.maximum(strike - paths[:, -1], 0)
    exercise_time = np.full(n_paths, steps, dtype=int)
    alive = np.ones(n_paths, dtype=bool)

    for t in range(1, steps):
        if t not in models:
            continue

        indices = np.flatnonzero(
            alive & (paths[:, t] < strike)
        )

        if len(indices) == 0:
            continue

        X = (paths[indices, t] / strike).reshape(-1, 1)

        continuation = models[t].predict(X)
        immediate = strike - paths[indices, t]

        exercise_now = immediate > continuation
        chosen = indices[exercise_now]

        cashflow[chosen] = immediate[exercise_now]
        exercise_time[chosen] = t
        alive[chosen] = False

    return cashflow * np.exp(-rate * exercise_time * dt)


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

    contract = dict(
        strike=100,
        rate=0.05,
        maturity=1.0,
    )

    baseline_models, _ = train_lsm_put(
        training,
        **contract,
        degree=2,
    )

    baseline_value, _, _, baseline_payoffs = evaluate_lsm_put(
        validation,
        baseline_models,
        **contract,
        return_payoffs=True,
    )

    print(f"Quadratic baseline: ${baseline_value:.6f}")
    print(
        "\nDepth | Forest value | Gain vs baseline | "
        "Paired 95% interval"
    )

    for depth in [2, 4, 6]:
        models = train_tree_put(
            training,
            **contract,
            max_depth=depth,
            min_samples_leaf=100,
            model_type="forest",
        )

        payoffs = evaluate_tree_put(
            validation,
            models,
            **contract,
        )

        differences = payoffs - baseline_payoffs
        gain = differences.mean()
        paired_se = (
            differences.std(ddof=1) / np.sqrt(len(differences))
        )

        print(
            f"{depth:5d} | ${payoffs.mean():.6f} | "
            f"${gain:+.6f} | "
            f"[{gain - 1.96 * paired_se:.6f}, "
            f"{gain + 1.96 * paired_se:.6f}]"
        )