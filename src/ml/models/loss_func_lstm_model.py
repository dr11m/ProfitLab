import tensorflow as tf


def combined_loss(y_true, y_pred):
    # Определение ошибки
    error = y_pred - y_true

    # Асимметричный штраф
    overestimation_cost = 0.7
    underestimation_cost = 1.0
    asymmetric_loss = tf.where(error > 0, overestimation_cost * error, underestimation_cost * (-error))

    # Расчет MAE и MSE для оценки разброса ошибок
    mae = tf.reduce_mean(tf.abs(error))
    mse = tf.reduce_mean(tf.square(error))

    # Комбинирование штрафа за разброс ошибок и асимметричного штрафа
    return tf.reduce_mean(asymmetric_loss) + 0.3 * tf.sqrt(mse - tf.square(mae))