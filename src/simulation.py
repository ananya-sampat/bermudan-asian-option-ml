from math import exp, sqrt
from random import Random


def simulate_paths(
    stock_price,
    rate,
    volatility,
    maturity,
    steps,
    paths,
    seed=42,
):
    if stock_price <= 0:
        raise ValueError("Stock price must be positive.")

    if volatility <= 0 or maturity <= 0:
        raise ValueError("Volatility and maturity must be positive.")

    for name, value in [("steps", steps), ("paths", paths)]:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")

    rng = Random(seed)
    dt = maturity / steps

    drift = (rate - 0.5 * volatility**2) * dt
    diffusion = volatility * sqrt(dt)

    all_paths = []

    for _ in range(paths):
        path = [stock_price]

        for _ in range(steps):
            z = rng.gauss(0, 1)
            next_price = path[-1] * exp(drift + diffusion * z)
            path.append(next_price)

        all_paths.append(path)

    return all_paths


if __name__ == "__main__":
    from statistics import mean, stdev

    from black_scholes import black_scholes_price
    from payoffs import put_payoff

    stock_price = 100
    strike = 100
    rate = 0.05
    volatility = 0.20
    maturity = 1.0

    simulated = simulate_paths(
        stock_price=stock_price,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        steps=50,
        paths=10_000,
    )

    print(f"Number of paths: {len(simulated)}")
    print(f"Prices per path: {len(simulated[0])}")
    print("First path, first 6 prices:")
    print([round(price, 2) for price in simulated[0][:6]])

    # Check that full-path simulation still prices a European put.
    discounted_payoffs = [
        exp(-rate * maturity) * put_payoff(path[-1], strike)
        for path in simulated
    ]

    estimate = mean(discounted_payoffs)
    se = stdev(discounted_payoffs) / sqrt(len(simulated))

    exact = black_scholes_price(
        stock_price=stock_price,
        strike=strike,
        rate=rate,
        volatility=volatility,
        maturity=maturity,
        option_type="put",
    )

    print(f"\nEuropean put estimate: ${estimate:.6f}")
    print(f"Black–Scholes price:   ${exact:.6f}")
    print(
        f"Approx. 95% interval: "
        f"[{estimate - 1.96 * se:.6f}, "
        f"{estimate + 1.96 * se:.6f}]"
    )