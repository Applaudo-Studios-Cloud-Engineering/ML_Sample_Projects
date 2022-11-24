from typing import List

import pandas as pd
import tensorflow as tf


def split_datasets(df: pd.DataFrame, label_col: str = 'label') -> [pd.DataFrame, pd.DataFrame]:
    """

    :param df:
    :param label_col:
    :return:
    """

    Y = df[label_col].astype('float32')
    X = df.drop([label_col]).astype('int32')

    return X, Y


def normalize_and_reshape_rows_intro_matrices(X: pd.DataFrame, normal_max: int = 255) -> pd.DataFrame:
    """

    :param X:
    :param normal_max:
    :return:
    """

    X = X / normal_max
    X = X.values.reshape(-1, 28, 28, 1)

    return X


def create_category_vectors_from_Y(Y: pd.DataFrame, num_classes: int = 10) -> pd.DataFrame:
    Y = tf.keras.utils.to_categorical(Y, num_classes)

    return Y


def create_model(optimizer: str = 'Adam', loss_fn: str = 'CategoricalCrossentropy', metrics=List[str]):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), input_shape=(28, 28, 1), name='Conv_1', activation='relu'),
        tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='same', name='Conv_2', activation='relu'),
        tf.keras.layers.BatchNormalization(name='Batch_Norm_1'),
        tf.keras.layers.MaxPooling2D(name='Max_Pool_1'),
        tf.keras.layers.Dropout(0.2, name='Drop_1'),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), name='Conv_3', activation='relu'),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', name='Conv_4', activation='relu'),
        tf.keras.layers.BatchNormalization(name='Batch_Norm_2'),
        tf.keras.layers.MaxPooling2D(name='Max_Pool_2'),
        tf.keras.layers.Dropout(0.25, name='Drop_2'),
        tf.keras.layers.Flatten(name='Flat_1'),
        tf.keras.layers.Dense(96, activation='relu', name='Dense_1'),
        tf.keras.layers.Dense(10, activation='softmax', name='Dense_2')
    ])

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    return model


def train_model(model, X_train, Y_train, validation_split: float = 0.2, epochs: int = 80, add_early_stop: bool = True):
    if add_early_stop:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
        history = model.fit(X_train, Y_train, validation_split=validation_split, epochs=epochs, callbacks=[early_stop])

        return history

    history = model.fit(X_train, Y_train, validation_split=0.2, epochs=80)

    return history


def predict(model, X, get_probs: bool = False):
    if get_probs:
        y_pred = model.predict(X)

        return y_pred

    y_pred = model.predict(X).argmax(axis=1)

    return y_pred

