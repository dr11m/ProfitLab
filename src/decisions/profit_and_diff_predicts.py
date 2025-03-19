from src.utils.logger_setup import logger
import numpy as np
from src.models import SaleRow


def get_profit_percent(v1: float, v2: float, fee=0.87):
    v2 = v2 * fee
    if v1 < v2:
        profit = round((v1 / v2 - 1) * -1, 3)
    else:
        profit = round((v2 / v1 - 1), 3)
    
    logger.info(f"{__name__} - profit is {profit} for {v1} and {v1} with a fee = {fee}")
    return profit


def get_profit_correction_based_on_preds_diff(preds_diff: float):
    pred_diffs =         np.array([0, 0.05, 0.09, 0.15,   0.3,  0.5,  0.7, 1])
    profit_corrections = np.array([0, 0.01, 0.04, 0.059,  0.15, 0.24, 0.4, 1])

    # Интерполяция значения
    interpolated_value = np.interp(preds_diff, pred_diffs, profit_corrections)

    return round(interpolated_value, 3)


def get_profit_correction_based_on_price(item_price: float):
    pred_diffs =         np.array([0,   0.5,  1.5,   4,     10,    20,  70,     500])
    profit_corrections = np.array([0.2, 0.04, 0.017, 0.01,  0.005, 0,   -0.015, -0.025])

    # Интерполяция значения
    interpolated_value = np.interp(item_price, pred_diffs, profit_corrections)

    return round(interpolated_value, 3)



def make_decision(row: SaleRow, min_profit: float, predicted_profit: float)-> tuple[bool, float]:
    logger.debug(f"item to make decision: {row}")

    predicts_diff = abs(get_profit_percent(row.predicted_price_1, row.predicted_price_2, 1))
    preds_diff_corr = get_profit_correction_based_on_preds_diff(predicts_diff)
    final_min_profit = min_profit + preds_diff_corr

    if predicted_profit > final_min_profit:
        logger.info(f"{__name__} - profit is {predicted_profit} is higher than min {min_profit}")
        return True, final_min_profit
    else:
        logger.info(f"{__name__} - profit is {predicted_profit} is less than min {min_profit}")
        return False, final_min_profit
