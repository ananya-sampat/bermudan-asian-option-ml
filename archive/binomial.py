
from payoffs import call_payoff, put_payoff
from math import exp, sqrt

def one_step_price(
    stock_price,
    strike,
    up_factor,
    down_factor,
    rate,
    maturity,
    option_type="put",
    american=False,
):
    """
    Price an option in a one-step binomial model.

    Assumptions: no dividends, constant continuously compounded rate.
    Maturity is measured in years.
    """
    if stock_price <= 0 or strike <= 0:
        raise ValueError("Stock price and strike must be positive.")

    if maturity <= 0:
        raise ValueError("Maturity must be positive.")

    growth = exp(rate * maturity)

    # Ensures the risk-neutral probability lies strictly between 0 and 1.
    if not 0 < down_factor < growth < up_factor:
        raise ValueError(
            "Require 0 < down_factor < exp(rate * maturity) < up_factor."
        )

    if option_type == "put":
        payoff = put_payoff
    elif option_type == "call":
        payoff = call_payoff
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    stock_up = stock_price * up_factor
    stock_down = stock_price * down_factor

    payoff_up = payoff(stock_up, strike)
    payoff_down = payoff(stock_down, strike)

    q = (growth - down_factor) / (up_factor - down_factor)

    continuation_value = exp(-rate * maturity) * (
        q * payoff_up + (1 - q) * payoff_down
    )

    if american:
        exercise_value = payoff(stock_price, strike)
        return max(exercise_value, continuation_value)

    return continuation_value
def two_step_price(
    stock_price,
    strike,
    up_factor,
    down_factor,
    rate,
    maturity,
    option_type="put",
    american=False,
):
    dt = maturity / 2

    # Option value after the first stock move is up.
    value_up = one_step_price(
        stock_price * up_factor,
        strike,
        up_factor,
        down_factor,
        rate,
        dt,
        option_type,
        american,
    )

    # Option value after the first stock move is down.
    value_down = one_step_price(
        stock_price * down_factor,
        strike,
        up_factor,
        down_factor,
        rate,
        dt,
        option_type,
        american,
    )

    # Work backward one more step to today.
    q = (exp(rate * dt) - down_factor) / (
        up_factor - down_factor
    )

    continuation = exp(-rate * dt) * (
        q * value_up + (1 - q) * value_down
    )

    if american:
        payoff = put_payoff if option_type == "put" else call_payoff
        return max(payoff(stock_price, strike), continuation)

    return continuation

def binomial_price(
    stock_price,
    strike,
    up_factor,
    down_factor,
    rate,
    maturity,
    steps,
    option_type="put",
    american=False,
):
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer.")

    if stock_price <= 0 or strike <= 0 or maturity <= 0:
        raise ValueError("Stock price, strike, and maturity must be positive.")

    dt = maturity / steps
    growth = exp(rate * dt)

    if not 0 < down_factor < growth < up_factor:
        raise ValueError("Up/down factors violate the no-arbitrage condition.")

    if option_type == "put":
        payoff = put_payoff
    elif option_type == "call":
        payoff = call_payoff
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    q = (growth - down_factor) / (up_factor - down_factor)
    discount = exp(-rate * dt)

    # Final layer: option payoffs, ordered by number of up moves.
    values = [
        payoff(
            stock_price * up_factor**j * down_factor**(steps - j),
            strike,
        )
        for j in range(steps + 1)
    ]

    # Move backward from the penultimate layer to today.
    for i in range(steps - 1, -1, -1):
        previous_values = []

        for j in range(i + 1):
            continuation = discount * (
                q * values[j + 1] + (1 - q) * values[j]
            )

            if american:
                current_stock = (
                    stock_price * up_factor**j * down_factor**(i - j)
                )
                value = max(payoff(current_stock, strike), continuation)
            else:
                value = continuation

            previous_values.append(value)

        values = previous_values

    return values[0]

def crr_price(
    stock_price,
    strike,
    rate,
    volatility,
    maturity,
    steps,
    option_type="put",
    american=False,
):
    if volatility <= 0:
        raise ValueError("Volatility must be positive.")

    if maturity <= 0:
        raise ValueError("Maturity must be positive.")

    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive integer.")

    dt = maturity / steps

    up_factor = exp(volatility * sqrt(dt))
    down_factor = exp(-volatility * sqrt(dt))

    return binomial_price(
        stock_price=stock_price,
        strike=strike,
        up_factor=up_factor,
        down_factor=down_factor,
        rate=rate,
        maturity=maturity,
        steps=steps,
        option_type=option_type,
        american=american,
    )


if __name__ == "__main__":
    inputs = dict(
        stock_price=100,
        strike=100,
        rate=0.05,
        volatility=0.20,
        maturity=1.0,
        option_type="put",
    )

    print("Steps | European put | American put")

    for steps in [10, 50, 100, 200, 500]:
        european = crr_price(
            **inputs, steps=steps, american=False
        )

        american = crr_price(
            **inputs, steps=steps, american=True
        )

        assert american >= european - 1e-10

        print(
            f"{steps:5d} | "
            f"${european:11.4f} | "
            f"${american:11.4f}"
        )