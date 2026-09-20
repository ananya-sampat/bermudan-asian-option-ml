import numpy as np
import matplotlib.pyplot as plt

from simulation import simulate_paths
from longstaff_schwartz import basis, train_lsm_put
from lsm_tree import train_tree_put


def tree_reference(
    stock_price,
    strike,
    rate,
    volatility,
    maturity,
    tree_steps=500,
    exercise_dates=50,
    decision_step=25,
):
    """Return continuation values at one scheduled exercise date."""
    if tree_steps % exercise_dates != 0:
        raise ValueError("tree_steps must be divisible by exercise_dates.")

    if not 0 < decision_step < exercise_dates:
        raise ValueError("Choose an exercise date before expiration.")

    spacing = tree_steps // exercise_dates
    target = decision_step * spacing
    dt = maturity / tree_steps

    u = np.exp(volatility * np.sqrt(dt))
    d = 1 / u
    q = (np.exp(rate * dt) - d) / (u - d)
    discount = np.exp(-rate * dt)

    if not 0 < q < 1:
        raise ValueError("Invalid risk-neutral probability.")

    j = np.arange(tree_steps + 1)
    terminal_stock = stock_price * u**j * d**(tree_steps - j)
    values = np.maximum(strike - terminal_stock, 0)

    for i in range(tree_steps - 1, -1, -1):
        continuation = discount * (
            q * values[1:] + (1 - q) * values[:-1]
        )

        j = np.arange(i + 1)
        stocks = stock_price * u**j * d**(i - j)
        immediate = np.maximum(strike - stocks, 0)

        if i == target:
            saved_stock = stocks.copy()
            saved_continuation = continuation.copy()

        if i % spacing == 0:
            values = np.maximum(immediate, continuation)
        else:
            values = continuation

    return saved_stock, saved_continuation, float(values[0])


if __name__ == "__main__":
    strike = 100
    rate = 0.05
    volatility = 0.20
    maturity = 1.0
    steps = 50
    t = 25

    training = np.array(
        simulate_paths(
            stock_price=100,
            rate=rate,
            volatility=volatility,
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

    tree_stock, tree_continuation, tree_price = tree_reference(
        stock_price=100,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        tree_steps=500,
        exercise_dates=steps,
        decision_step=t,
    )

    # Restrict the plot to well-represented in-the-money training prices.
    observed = training[:, t]
    observed = observed[observed < strike]
    low, high = np.quantile(observed, [0.01, 0.99])
    stock_grid = np.linspace(low, high, 500)

    immediate = np.maximum(strike - stock_grid, 0)
    quadratic = basis(stock_grid, strike) @ quadratic_models[t]
    forest = forest_models[t].predict(
        (stock_grid / strike).reshape(-1, 1)
    )

    tree_immediate = np.maximum(strike - tree_stock, 0)
    tree_gap = tree_continuation - tree_immediate

    # Locate the last strictly preferable early-exercise node.
    exercise_nodes = np.flatnonzero(
        (tree_immediate > 0) & (tree_gap < -1e-8)
    )

    print(f"Tree price with matched exercise dates: ${tree_price:.6f}")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.axhline(0, color="black", linestyle="--", label="Decision threshold")
    ax.plot(
        stock_grid, quadratic - immediate,
        label="Quadratic continuation minus payoff",
    )
    ax.plot(
        stock_grid, forest - immediate,
        label="Forest continuation minus payoff",
    )

    visible = (tree_stock >= low) & (tree_stock <= high)
    ax.plot(
        tree_stock[visible],
        tree_gap[visible],
        "o-",
        color="green",
        markersize=4,
        label="Tree reference at its price nodes",
    )

    if len(exercise_nodes):
        last = exercise_nodes[-1]

        if last + 1 < len(tree_stock):
            left = tree_stock[last]
            right = tree_stock[last + 1]

            print(
                f"Tree boundary bracket at six months: "
                f"${left:.4f} to ${right:.4f}"
            )

            ax.axvspan(
                left, right, color="green", alpha=0.12,
                label="Tree boundary bracket",
            )

    ax.set(
        xlabel="Stock price",
        ylabel="Continuation minus exercise payoff ($)",
        title="Six months remaining: below zero = exercise",
        xlim=(low, high),
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig("boundary_comparison.png", dpi=180)
    plt.show()