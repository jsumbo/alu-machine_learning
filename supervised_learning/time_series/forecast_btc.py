#!/usr/bin/env python3
"""
Build and train an RNN to forecast Bitcoin close prices.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def load_data():
    """Loads preprocessed datasets and returns tf.data.Dataset"""
    X = np.load("X.npy")
    y = np.load("y.npy")

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(10000, seed=42)

    data_len = len(X)
    train_size = int(data_len * 0.7)
    val_size = int(data_len * 0.2)

    train = dataset.take(train_size).batch(64)
    val = dataset.skip(train_size).take(val_size).batch(64)
    test = dataset.skip(train_size + val_size).batch(64)

    return train, val, test

def build_model(input_shape):
    """Creates the LSTM model"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_data()
    for x, _ in train_ds.take(1):
        input_shape = x.shape[1:]
        break

    model = build_model(input_shape)

    model.fit(train_ds,
              epochs=20,
              validation_data=val_ds,
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

    loss, mae = model.evaluate(test_ds)
    print(f"Test Loss: {loss}, MAE: {mae}")

    model.save("btc_rnn_model.h5")
