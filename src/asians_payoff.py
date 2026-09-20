import numpy as np

from simulation import simulate_paths


def running_average(paths):
    """Average each path from time zero through each current date."""
    paths = np.asarray(paths, dtype=float)

    if paths.ndim != 2 or paths.shape[1] == 0:
        raise ValueError("Expected a 2D array with at least one date.")

    observation_counts = np.arange(1, paths.shape[1] + 1)

    return np.cumsum(paths, axis=1) / observation_counts


def asian_put_payoffs(averages, strike):
    return np.maximum(strike - np.asarray(averages), 0)


if __name__ == "__main__":
    # Hand-checkable example.
    example_paths = np.array([
        [100, 80, 90],
        [100, 110, 90],
    ])

    averages = running_average(example_paths)
    payoffs = asian_put_payoffs(averages, strike=100)

    print("Example running averages:")
    print(averages)

    print("\nExample exercise payoffs:")
    print(payoffs)

    np.testing.assert_allclose(
        averages,
        [[100, 90, 90], [100, 105, 100]],
    )

    np.testing.assert_allclose(
        payoffs,
        [[0, 10, 10], [0, 0, 0]],
    )

    # Verify that later observations cannot change earlier averages.
    modified_paths = example_paths.copy()
    modified_paths[:, -1] = 999

    np.testing.assert_allclose(
        running_average(modified_paths)[:, :-1],
        averages[:, :-1],
    )

    print("\nRunning-average checks passed.")

    # Price the European version: exercise only at expiration.
    paths = np.array(
        simulate_paths(
            stock_price=100,
            rate=0.05,
            volatility=0.20,
            maturity=1.0,
            steps=50,
            paths=50_000,
            seed=456,
        )
    )

    averages = running_average(paths)

    discounted_payoffs = (
        np.exp(-0.05)
        * asian_put_payoffs(averages[:, -1], strike=100)
    )

    value = discounted_payoffs.mean()
    se = discounted_payoffs.std(ddof=1) / np.sqrt(len(paths))

    print(f"\nEuropean Asian put estimate: ${value:.6f}")
    print(
        f"Approx. 95% interval: "
        f"[{value - 1.96 * se:.6f}, "
        f"{value + 1.96 * se:.6f}]"
    )