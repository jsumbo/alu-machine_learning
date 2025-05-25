#!/usr/bin/env python3
"""
Preprocess Bitcoin data from Coinbase and Bitstamp.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def load_and_merge_data(coinbase_path, bitstamp_path):
    """Loads and merges datasets based on timestamps"""
    coinbase = pd.read_csv(coinbase_path)
    bitstamp = pd.read_csv(bitstamp_path)

    coinbase['timestamp'] = pd.to_datetime(coinbase['Timestamp'], unit='s')
    bitstamp['timestamp'] = pd.to_datetime(bitstamp['Timestamp'], unit='s')

    coinbase.set_index('timestamp', inplace=True)
    bitstamp.set_index('timestamp', inplace=True)

    combined = pd.concat([coinbase, bitstamp]).sort_index()
    combined = combined.groupby(combined.index).mean()
    return combined


def preprocess_data(df):
    """Cleans, scales and formats data into sequences for RNN"""
    df = df[['Close']]  # Only use the close price for prediction
    df = df.dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    SEQ_LEN = 60 * 24  # 24 hours of 1-min intervals
    TARGET_OFFSET = 60  # 1 hour later

    for i in range(SEQ_LEN, len(scaled) - TARGET_OFFSET):
        X.append(scaled[i - SEQ_LEN:i])
        y.append(scaled[i + TARGET_OFFSET][0])

    X = np.array(X)
    y = np.array(y)

    np.save('X.npy', X)
    np.save('y.npy', y)
    np.save('scaler_minmax.npy', scaler.data_min_)
    np.save('scaler_scale.npy', scaler.scale_)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    data = load_and_merge_data("coinbase.csv", "bitstamp.csv")
    preprocess_data(data)
