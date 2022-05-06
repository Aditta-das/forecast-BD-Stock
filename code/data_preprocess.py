import os
import pandas as pd
import torch
from sliding_window import sliding_window

def get_training_datasets(features, test_len):
    ts = pd.read_csv("/media/aditta/NewVolume/forecast_research/data/ACIF_Historical_Data.csv")
    ts = ts["Price"].tolist()
    X, Y = sliding_window(ts, features)
    X_train, Y_train, X_test, Y_test = X[0:-test_len], Y[0:-test_len], X[-test_len:], Y[-test_len:]
    train_len = round(len(ts) * 0.7)
    X_train, X_val, Y_train, Y_val = X_train[0:train_len], X_train[train_len:], Y_train[0:train_len], Y_train[train_len:]
    x_train = torch.tensor(X_train)
    y_train = torch.tensor(Y_train)
    x_val = torch.tensor(X_val)
    y_val = torch.tensor(Y_val)
    x_test = torch.tensor(X_test)
    y_test = torch.tensor(Y_test)

    return x_train, x_val, x_test, y_train, y_val, y_test
