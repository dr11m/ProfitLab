from src.utils.logger_setup import logger
from src.models import SaleRow


def get_profit_percent(v1: float, v2: float, fee=0.87):
    v2 = v2 * fee
    if v1 < v2:
        profit = round((v1 / v2 - 1) * -1, 3)
    else:
        profit = round((v2 / v1 - 1), 3)
    
    logger.info(f"{__name__} - profit is {profit} for {v1} and {v1} with a fee = {fee}")
    return profit


def make_decision(row: SaleRow, min_profit: float, predicted_profit: float)-> tuple[bool, float]:
    logger.debug(f"item to make decision: {row}")

    if predicted_profit > min_profit:
        logger.info(f"{__name__} - profit is {predicted_profit} is higher than min {min_profit}")
        return True, min_profit
    else:
        logger.info(f"{__name__} - profit is {predicted_profit} is less than min {min_profit}")
        return False, min_profit
