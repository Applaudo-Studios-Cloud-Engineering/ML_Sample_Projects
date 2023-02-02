from typing import Tuple, Any, List
import tensorflow as tf


def create_sequential_model(input_shape: Tuple[int, int], neurons_hl_1: int, activation_hl_1: str, neurons_ol: int):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(units=neurons_hl_1), activation=activation_hl_1)
    model.add(tf.keras.layers.Dense(units=neurons_ol))

    return model


def compile_model(optimizer_fn: Any, loss_fn: Any, metrics: List[Any], model: tf.keras.Model):
    model.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=metrics)

    return model


def train__and_evaluate_model(model: tf.keras.Model, X, y, epochs: int, train_verbosity: int, X_test, y_test):
    model.fit(X, y, epochs=epochs, verbose=train_verbosity)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

    return model, test_loss, test_acc


