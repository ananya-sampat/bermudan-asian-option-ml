import numpy as np
import matplotlib.pyplot as plt

from simulation import simulate_paths
from longstaff_schwartz import basis, train_lsm_put
from lsm_tree import train_tree_put


if __name__ == "__main__":
    strike = 100
    rate = 0.05
    maturity = 1.0
    steps = 50
    t = 25  # Halfway through the one-year contract.

    training = np.array(
        simulate_paths(
            stock_price=100,
            rate=rate,
            volatility=0.20,
            maturity=maturity,
            steps=steps,
            paths=20_000,
            seed=42,
        )
    )

    contract = dict(
        strike=strike,
        rate=rate,
        maturity=maturity,
    )

    quadratic_models, _ = train_lsm_put(
        training, **contract, degree=2
    )

    forest_models = train_tree_put(
        training,
        **contract,
        max_depth=4,
        min_samples_leaf=100,
        model_type="forest",
    )

    # Plot within the central range of in-the-money training prices.
    observed = training[:, t]
    observed = observed[observed < strike]
    low, high = np.quantile(observed, [0.01, 0.99])
    stock_grid = np.linspace(low, high, 500)

    quadratic = basis(stock_grid, strike) @ quadratic_models[t]

    forest = forest_models[t].predict(
        (stock_grid / strike).reshape(-1, 1)
    )

    immediate = np.maximum(strike - stock_grid, 0)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        stock_grid, immediate,
        color="black", linestyle="--", label="Exercise payoff",
    )
    ax.plot(stock_grid, quadratic, label="Quadratic continuation")
    ax.plot(stock_grid, forest, label="Forest continuation")

    ax.set(
        xlabel="Stock price",
        ylabel="Value at the decision date ($)",
        title="Exercise or wait? Six months remaining",
    )
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig("continuation_comparison.png", dpi=180)
    plt.show()