from math import erf, exp, log, sqrt


def normal_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def black_scholes_price(
    stock_price,
    strike,
    rate,
    volatility,
    maturity,
    option_type="put",
):
    if stock_price <= 0 or strike <= 0:
        raise ValueError("Stock price and strike must be positive.")

    if volatility <= 0 or maturity <= 0:
        raise ValueError("Volatility and maturity must be positive.")

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'.")

    d1 = (
        log(stock_price / strike)
        + (rate + 0.5 * volatility**2) * maturity
    ) / (volatility * sqrt(maturity))

    d2 = d1 - volatility * sqrt(maturity)
    discounted_strike = strike * exp(-rate * maturity)

    if option_type == "call":
        return (
            stock_price * normal_cdf(d1)
            - discounted_strike * normal_cdf(d2)
        )

    return (
        discounted_strike * normal_cdf(-d2)
        - stock_price * normal_cdf(-d1)
    )


if __name__ == "__main__":
    from binomial import crr_price

    inputs = dict(
        stock_price=100,
        strike=100,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        option_type="put",
    )

    exact = black_scholes_price(**inputs)

    print(f"Black–Scholes European put: ${exact:.6f}")
    print("Steps | Tree price | Absolute error")

    for steps in [10, 50, 100, 200, 500]:
        tree = crr_price(**inputs, steps=steps, american=False)
        error = abs(tree - exact)

        print(f"{steps:5d} | ${tree:9.6f} | {error:.6f}")