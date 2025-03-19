import numpy as np
import pandas as pd
from typing import Optional
from src.models import SaleRow, DecisionFunc
from  src.utils.get_profit import get_profit_percent
    

def evaluate_decision(
    df: pd.DataFrame,
    decision_func: DecisionFunc,
    decision_params: dict,
    predicted_price_attr: str,
    second_predicted_price_attr: str = "",
) -> pd.DataFrame:
    r"""
    Для каждой сделки:
    
      1. Вычисляется ожидаемая (симулированная) цена продажи с помощью
         `calculate_expected_sale_price`.
         
      2. Рассчитывается прибыль в USD и в процентах с учетом корректировок:
      
         \[
         sale\_profit\_usd = (sale\_price \times SALE\_COMMISSION\_FACTOR) - \left(\frac{buy\_price}{BUY\_ADJUSTMENT\_FACTOR}\right)
         \]
         \[
         sale\_profit\_percent = \text{get\_profit\_percent}\left(\frac{buy\_price}{BUY\_ADJUSTMENT\_FACTOR}, sale\_price \times SALE\_COMMISSION\_FACTOR\right)
         \]
      
      3. На основе предсказанной цены (используем поле `predict_1_models_mean`)
         рассчитывается предсказанная прибыль.
      
      4. Принятие решения: если предсказанная прибыль в % \(\ge\) decision_threshold, то решение True.
      
      5. Дополнительно рассчитывается ошибка предсказания как разница между
         скорректированными ценами предсказанной и симулированной продажи.
    """
    results = []
    for _, row in df.iterrows():
        item = SaleRow(**row.replace({np.nan: None}).to_dict())


        
        sale_profit_usd = item.sold_price - item.buy_price
        sale_profit_percent = get_profit_percent(item.buy_price, item.sold_price)
        
        # Предсказанная цена (используем predict_1_models_mean)
        if hasattr(row, predicted_price_attr):
            predicted_price = getattr(row, predicted_price_attr)
        else:
            raise AttributeError(f"Item object does not have attribute {decision_params['predicted_price_attr']}")
        
        preds_diff = _get_preds_diff_if_exist(predicted_price, row.get(second_predicted_price_attr, ""))
        
        predicted_profit_percent = get_profit_percent(item.buy_price, predicted_price)

        decision_params["predicted_profit"] = predicted_profit_percent
        decision, decision_min_profit = decision_func(item, **decision_params)
        
        results.append({
            'name': item.name,
            'buy_price': item.buy_price,
            'simulated_sale_price': item.sold_price,
            'simulated_prices': item.future_validated_prices,
            'predicted_price': predicted_price,
            'sale_profit_usd': sale_profit_usd,
            'sale_profit_percent': sale_profit_percent,
            'predicted_profit_percent': predicted_profit_percent,
            'decision': decision,
            'actual_profit_usd': sale_profit_usd if decision else 0.0,
            'potential_loss_usd': sale_profit_usd if (decision and sale_profit_usd < 0) else 0.0,
            'potential_profit_usd': sale_profit_usd if (not decision and sale_profit_percent > decision_min_profit) else 0.0,
            'error_percent': round(sale_profit_percent - predicted_profit_percent, 2),
            'preds_diff': preds_diff
        })

    results_df =  pd.DataFrame(results)

    results_df.to_csv("logs/results_df.csv", index=False)

    return results_df


def _get_preds_diff_if_exist(predicted_price_1 , predicted_price_2)-> Optional[float]:
    if predicted_price_2 == "":
        return None
    
    return get_profit_percent(predicted_price_1, predicted_price_2)
    