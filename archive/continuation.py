import numpy as np

from simulation import simulate_paths


if __name__ == "__main__":
    strike = 100
    rate = 0.05
    maturity = 1.0
    steps = 50
    dt = maturity / steps

    paths = np.array(
        simulate_paths(
            stock_price=100,
            rate=rate,
            volatility=0.20,
            maturity=maturity,
            steps=steps,
            paths=10_000,
            seed=42,
        )
    )

    # All paths, at the date just before expiration.
    stock_now = paths[:, -2]
    stock_final = paths[:, -1]

    # For this put, consider early exercise only when S < K.
    in_the_money = stock_now < strike

    x = stock_now[in_the_money] / strike

    # Training targets: discounted expiration payoffs.
    y = np.exp(-rate * dt) * np.maximum(
        strike - stock_final[in_the_money], 0
    )

    # Quadratic regression: beta_0 + beta_1*x + beta_2*x^2.
    X = np.column_stack([np.ones_like(x), x, x**2])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    print(f"Training observations: {len(x)}")
    print("Regression coefficients:", beta)

    # Apply the fitted rule at some example stock prices.
    print("\nStock | Exercise payoff | Estimated continuation | Decision")

    for stock in [80, 90, 95, 99]:
        features = np.array([1, stock / strike, (stock / strike)**2])

        continuation = features @ beta
        exercise = max(strike - stock, 0)

        decision = "Exercise" if exercise > continuation else "Wait"

        print(
            f"{stock:5.0f} | {exercise:15.4f} | "
            f"{continuation:22.4f} | {decision}"
        )
        