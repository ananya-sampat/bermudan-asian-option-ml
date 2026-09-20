import numpy as np

from simulation import simulate_paths
from longstaff_schwartz import train_lsm_put, evaluate_lsm_put
from lsm_tree import train_tree_put, evaluate_tree_put


if __name__ == "__main__":
    settings = dict(
        stock_price=100,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        steps=50,
    )

    contract = dict(
        strike=100,
        rate=0.05,
        maturity=1.0,
    )

    # Same validation paths for every comparison.
    validation = np.array(
        simulate_paths(**settings, paths=50_000, seed=456)
    )

    print("Training seed | Quadratic | Forest depth 4 | Forest minus quadratic")

    for seed in [42, 43, 44]:
        training = np.array(
            simulate_paths(**settings, paths=20_000, seed=seed)
        )

        quadratic_models, _ = train_lsm_put(
            training, **contract, degree=2
        )

        quadratic_value, _, _, _ = evaluate_lsm_put(
            validation,
            quadratic_models,
            **contract,
            return_payoffs=True,
        )

        forest_models = train_tree_put(
            training,
            **contract,
            max_depth=4,
            min_samples_leaf=100,
            model_type="forest",
        )

        forest_payoffs = evaluate_tree_put(
            validation, forest_models, **contract
        )
        forest_value = forest_payoffs.mean()

        print(
            f"{seed:13d} | ${quadratic_value:.6f} | "
            f"${forest_value:.6f} | "
            f"${forest_value - quadratic_value:+.6f}"
        )