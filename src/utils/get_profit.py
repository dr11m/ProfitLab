

def get_profit_percent(buy_price: float, sell_price: float) -> float:
    if buy_price < sell_price:
        profit = round((buy_price / sell_price - 1) * -1, 3)
    else:
        profit = round((sell_price / buy_price - 1), 3)
    
    return profit