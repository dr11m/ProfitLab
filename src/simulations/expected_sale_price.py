

def calculate_expected_sale_price(
    future_prices: list[float],
    buy_price: float,
    min_profit: float,
    delta: float,
    success_prob: float,
    fee: float
) -> float:
    r"""
    formulas can be used in docstrings, to see them in vscode install 2 extensions: 
    `Comment Formula` and `MathJax Highlight` both by `Casper Huang`.

    Computes the expected sale price.

    *Mean of 2 last prices is chosen if none of the previous prices satisfies the condition.

    This function calculates the expected sale price based on a list of forecasted future prices,
    the purchase price, and specified profit parameters. It considers the probability of a sale on each day,
    where the probability for day $$i$$ is given by:

    $$
    p_i = success\_prob \times prob\_reach\_day
    $$

    where:

    $$
    prob\_reach\_day(i) = \prod_{j=0}^{i-1} (1 - success\_prob)
    $$

    The expected sale price is then computed as:

    $$
    E[P] = \sum_{i=0}^{n-1} p_i \cdot P_i + \left( \prod_{i=0}^{n-1}(1 - success\_prob) \right) \cdot P_{n-1}
    $$

    where $$P_i$$ represents the sale price on day $$i$$ after adjustment.


    Parameters
    ----------
    future_prices : List[float]
        A list of forecasted sale prices for each day.
    buy_price : float
        The purchase price used to compute profit percentages.
    min_profit : float
        The minimum required profit percentage (e.g., 0.25 for 25%).
    delta : float
        The daily decrement in the required profit percentage (e.g., 0.01 for a 1% decrease per day).
    success_prob : float
        The probability of a successful sale on any given day (a value between 0 and 1).
    fee : floar
        Сommission of the platform on which the sale is planned (0.99 for Binance for example)

    Returns
    -------
    float
        The calculated expected sale price.


    Examples
    --------
    >>> future_prices = [100, 105, 110, 115]
    >>> expected_price = calculate_expected_sale_price(future_prices, 80, 0.25, 0.01, 0.75)
    >>> print(expected_price)
    """
    if not future_prices:
        raise ValueError("The list of future_prices cannot be empty.")

    expected_price = 0.0
    prob_reach_day = 1.0  # Probability of reaching day i without a sale

    for i, price in enumerate(future_prices):
        current_min_profit = min_profit - delta * i
        adjusted_price = price * fee
        raw_profit = get_profit_percent(buy_price, adjusted_price)
        if raw_profit >= current_min_profit:
            prob_sell = prob_reach_day * success_prob
            expected_price += price * prob_sell
            prob_reach_day *= (1 - success_prob)
    
    if prob_reach_day == 1:  # means none of the previous prices satisfies the condition
        return round((future_prices[-1] + future_prices[-2]) / 2, 3)

    expected_price += future_prices[-1] * prob_reach_day

    return round(expected_price, 3)


def get_profit_percent(buy_price: float, sell_price: float) -> float:
    if buy_price < sell_price:
        profit = round((buy_price / sell_price - 1) * -1, 3)
    else:
        profit = round((sell_price / buy_price - 1), 3)
    
    return profit
