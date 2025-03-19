import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from typing import List
import os
from sklearn.preprocessing import MinMaxScaler

from src.ml.models.loss_func_lstm_model import combined_loss



current_directory = os.getcwd()
print("current dir:", current_directory)


cb_predict_1_model = CatBoostRegressor()

cb_predict_1_model.load_model('../src/ml/models/catboost_model_predict_next_based_on_10_sales.cbm')

lstm_predict_1_model = load_model('../src/ml/models/lstm_model_predict_next_based_on_10_sales.h5',
                                  custom_objects={'combined_loss': combined_loss})


class PredictedData:
    predicted_1_lstm: float = -1
    predicted_1_cb: float = -1

    @property
    def mean_1(self):
        return round((self.predicted_1_lstm + self.predicted_1_cb) / 2 if self.predicted_1_lstm != -1 and self.predicted_1_cb != -1 else -1, 3)
    
    def to_dict(self):
        return {
            'predicted_1_cb': self.predicted_1_cb,
            'predicted_1_lstm': self.predicted_1_lstm,
            'mean_1': self.mean_1,
        }

    def validate(self):
        if (
            self.predicted_1_lstm == -1
         or self.predicted_1_cb == -1
         or self.mean_1 == -1
        ):
            raise ValueError("Invalid predicted values")


def get_predicted_data(prices: List[float])-> PredictedData:
    how_many_last_price_to_get = 10

    assert len(prices) >= how_many_last_price_to_get

    predicted_data = PredictedData()

    last_10 = prices[-how_many_last_price_to_get:]

    scaler_last_10 = MinMaxScaler()

    norm_last_10 = scaler_last_10.fit_transform(pd.DataFrame(last_10).values.reshape(-1, 1))

    #1d matrix shape
    norm_predicted_1_cb = cb_predict_1_model.predict(norm_last_10.reshape(-1))
    #3d matrix shape
    norm_predicted_1_lstm = lstm_predict_1_model.predict(norm_last_10.reshape(1, norm_last_10.shape[0], 1))[0][0]

    # return back to normal values (from norm 0-1)
    predicted_data.predicted_1_cb = scaler_last_10.inverse_transform(norm_predicted_1_cb.reshape(-1, 1))[0][0]
    predicted_data.predicted_1_lstm = scaler_last_10.inverse_transform(norm_predicted_1_lstm.reshape(-1, 1))[0][0]

    predicted_data.validate()

    
    return predicted_data


if __name__ == "__main__":
    data = get_predicted_data([1,2,3,4,5,6,7,8,9,10])
    print(data.to_dict())
