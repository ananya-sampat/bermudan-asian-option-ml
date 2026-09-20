from math import exp, sqrt
from random import Random
from statistics import mean, stdev

from payoffs import call_payoff, put_payoff


def monte_carlo_price(
    stock_price,
    strike,
    rate,
    volatility,
    maturity,
    paths,
    option_type="put",
    seed=42,
):
    if stock_price <= 0 or strike <= 0:
        raise ValueError("Stock price and strike must be positive.")

    if volatility <= 0 or maturity <= 0:
        raise ValueError("Volatility and maturity must be positive.")

    if not isinstance(paths, int) or isinstance(paths, bool) or paths < 2:
        raise ValueError("paths must be an integer of at least 2.")

    if option_type == "put":
        payoff = put_payoff
    elif option_type == "call":
        payoff = call_payoff
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    rng = Random(seed)

    drift = (rate - 0.5 * volatility**2) * maturity
    diffusion = volatility * sqrt(maturity)
    discount = exp(-rate * maturity)

    discounted_payoffs = []

    for _ in range(paths):
        z = rng.gauss(0, 1)

        terminal_price = stock_price * exp(drift + diffusion * z)

        discounted_payoffs.append(
            discount * payoff(terminal_price, strike)
        )

    estimate = mean(discounted_payoffs)
    standard_error = stdev(discounted_payoffs) / sqrt(paths)

    return estimate, standard_error


if __name__ == "__main__":
    from black_scholes import black_scholes_price

    inputs = dict(
        stock_price=100,
        strike=100,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        option_type="put",
    )

    exact = black_scholes_price(**inputs)
    print(f"Black–Scholes price: ${exact:.6f}")
    print("Paths | MC estimate | Standard error | Approx. 95% interval")

    for paths in [1_000, 10_000, 100_000]:
        estimate, se = monte_carlo_price(**inputs, paths=paths)

        lower = estimate - 1.96 * se
        upper = estimate + 1.96 * se

        print(
            f"{paths:6d} | ${estimate:.6f} | "
            f"{se:.6f} | [{lower:.6f}, {upper:.6f}]"
        )