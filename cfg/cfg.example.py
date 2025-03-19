from src.decisions import simple_profit, profit_and_diff_predicts


data_to_test = "test_data_15_958_with_real_sold_prices.csv"  # only files in datasets/ready_to_work*


approaches = [
    {
        "name": "Approach A",
        "description": "...",
        "decision_func": simple_profit.make_decision,
        "params": {
            "min_profit": 0.1,
        },
        "predicted_price_attr": "predicted_price_1",
        "second_predicted_price_attr": "predicted_price_2"
    },

    {
        "name": "Approach B",
        "description": "...",
        "decision_func": simple_profit.make_decision,
        "params": {
            "min_profit": 0.05,
        },
        "predicted_price_attr": "predicted_price_1",
        "second_predicted_price_attr": "predicted_price_2"
    },
]