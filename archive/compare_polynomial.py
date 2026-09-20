import numpy as np

from simulation import simulate_paths
from longstaff_schwartz import train_lsm_put, evaluate_lsm_put


if __name__ == "__main__":
    settings = dict(
        stock_price=100,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        steps=50,
    )

    # All models get identical training and validation paths.
    training = np.array(
        simulate_paths(**settings, paths=20_000, seed=42)
    )

    validation = np.array(
        simulate_paths(**settings, paths=50_000, seed=456)
    )

    print("Degree | Training value | Validation value | Std. error")
    payoffs_by_degree = {}

    for degree in [1, 2, 3, 5]:
        models, training_value = train_lsm_put(
            training,
            strike=100,
            rate=0.05,
            maturity=1.0,
            degree=degree,
        )

        value, se, _, payoffs = evaluate_lsm_put(
            validation,
            models,
            strike=100,
            rate=0.05,
            maturity=1.0,
            return_payoffs=True,
        )

        payoffs_by_degree[degree] = payoffs

        print(
            f"{degree:6d} | ${training_value:13.6f} | "
            f"${value:15.6f} | ${se:.6f}"
        )
    print("\nPaired comparisons against degree 2:")

    for degree in [1, 3, 5]:
        differences = payoffs_by_degree[degree] - payoffs_by_degree[2]

        improvement = differences.mean()
        paired_se = differences.std(ddof=1) / np.sqrt(len(differences))

        lower = improvement - 1.96 * paired_se
        upper = improvement + 1.96 * paired_se

        print(
            f"Degree {degree} minus degree 2: "
            f"${improvement:.6f}, "
            f"approx. 95% interval [{lower:.6f}, {upper:.6f}]"
        )