

def get_average_sale_price(future_prices: list[float], num_prices: int = 3) -> float:
    r"""
    formulas can be used in docstrings, to see them in vscode install 2 extensions: 
    `Comment Formula` and `MathJax Highlight` both by `Casper Huang`.
    
    Computes the average sale price based on the specified number of prices.

    The sale price is calculated as:

    $$
    Sale\_Price = \frac{1}{N} \sum_{i=0}^{N-1} P_i
    $$

    where:
    $$
    N = num\_prices
    $$
    $$
    P_i \text{ is the price on day } i
    $$

    If fewer than $$N$$ prices are available, the average is taken over all available prices.

    Parameters
    ----------
    future_prices : list[float]
        A list of forecasted sale prices.
    num_prices : int, optional
        The number of initial prices to include in the average (default is 3).

    Returns
    -------
    float
        The average of the selected number of prices.

    Raises
    ------
    ValueError
        If the list of future_prices is empty or num_prices is less than 1.

    Examples
    --------
    >>> get_average_sale_price([100, 105, 110, 115], 3)
    105.0
    >>> get_average_sale_price([120, 130], 5)
    125.0
    """
    if not future_prices:
        raise ValueError("The list of future_prices cannot be empty.")
    
    if num_prices < 1:
        raise ValueError("num_prices must be at least 1.")
    
    if len(future_prices) < num_prices:
        raise ValueError(f"The list must have at least {num_prices} elements.")
    
    return round(sum(future_prices[:num_prices]) / num_prices, 3)
