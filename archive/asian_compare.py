import numpy as np

from simulation import simulate_paths
from asian_lsm import train_asian_put, evaluate_asian_put


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

    results = {}

    for label, average_only in [
        ("Average only", True),
        ("Stock + average", False),
    ]:
        models = train_asian_put(
            training,
            **contract,
            average_only=average_only,
        )

        payoffs, _ = evaluate_asian_put(
            validation, models, **contract
        )

        results[label] = payoffs
        print(f"{label}: ${payoffs.mean():.6f}")

    differences = (
        results["Stock + average"] - results["Average only"]
    )
    gain = differences.mean()
    se = differences.std(ddof=1) / np.sqrt(len(differences))

    print(f"\nGain from the expanded feature set: ${gain:.6f}")
    print(
        f"Paired 95% interval: "
        f"[{gain - 1.96 * se:.6f}, {gain + 1.96 * se:.6f}]"
    )