def call_payoff(stock_price, strike):
    """Value received from exercising a call."""
    return max(stock_price - strike, 0.0)


def put_payoff(stock_price, strike):
    """Value received from exercising a put."""
    return max(strike - stock_price, 0.0)


if __name__ == "__main__":
    strike = 100

    for stock_price in [80, 100, 120]:
        call = call_payoff(stock_price, strike)
        put = put_payoff(stock_price, strike)

        print(
            f"Stock: {stock_price} | "
            f"Call payoff: {call:.2f} | "
            f"Put payoff: {put:.2f}"
        )

    assert call_payoff(120, 100) == 20
    assert call_payoff(80, 100) == 0
    assert put_payoff(80, 100) == 20
    assert put_payoff(120, 100) == 0

    print("All payoff tests passed.")