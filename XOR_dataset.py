""" MLP Exercise -  test script - Python course ITC
author: Roi Weinberger. Date: 2020-01-01 """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MLP import *
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    df = pd.read_csv('XOR_dataset.csv')
    X = np.array(df.iloc[:, 0:2])
    y = np.array(df.iloc[:, 2]).reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # use binary cross-entropy loss function
    net_bce = Network(2, 1, Sigmoid, 2, ReLU, binary_cross_entropy, a=1e-6, mini_batch_size=100, max_runs=10)
    net_bce.fit(X_train, y_train)
    y_pred = net_bce.predict(X_train)
    mse = MSE()
    error = mse.calculate(true=y_train, pred=y_pred)
    print(error)
