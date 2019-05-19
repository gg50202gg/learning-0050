import json
import datetime
import pandas as pd
import numpy as np

import models

from keras.callbacks import EarlyStopping


def readTrain():
    # 讀取資料
    train = pd.read_csv("TW50.csv")
    return train


def augFeatures(train):
    # 日期正規
    train["Date"] = pd.to_datetime(train["Date"])
    train["year"] = train["Date"].dt.year
    train["month"] = train["Date"].dt.month
    train["date"] = train["Date"].dt.day
    train["day"] = train["Date"].dt.dayofweek
    return train


def normalize(train):
    # 資料正規
    train = train.drop(["Date"], axis=1)
    train_norm = train.apply(lambda x: (
        x - np.mean(x)) / (np.max(x) - np.min(x)))
    return train_norm


def buildTrain(train, pastDay=30, futureDay=5):
    """
    建立訓練資料
    """
    X_train, Y_train = [], []
    for i in range(train.shape[0] - futureDay - pastDay):
        x = np.array(train.iloc[i: i + pastDay])
        y = np.array(train.iloc[i + pastDay: i +
                                pastDay + futureDay]["Open"])

        X_train.append(x)
        Y_train.append(y)
    return np.array(X_train), np.array(Y_train)


def shuffle(X, Y):
    """
    亂序
    """
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]


def splitData(X, Y, rate):
    """
    訓練資料 驗證資料
    """
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val


def f1t1():
    train = readTrain()
    train_Aug = augFeatures(train)
    train_norm = normalize(train_Aug)
    # change the last day and next day
    X_train, Y_train = buildTrain(train_norm, 1, 1)
    X_train, Y_train = shuffle(X_train, Y_train)
    X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

    # from 2 dimmension to 3 dimension
    Y_train = Y_train[:, np.newaxis]
    Y_val = Y_val[:, np.newaxis]

    model = models.buildOneToOneModel(X_train.shape)
    callback = EarlyStopping(
        monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(X_train, Y_train, epochs=1000, batch_size=128,
              validation_data=(X_val, Y_val), callbacks=[callback])
    return model


def fnt1(pastDay=30):
    train = readTrain()
    train_Aug = augFeatures(train)
    train_norm = normalize(train_Aug)
    # change the last day and next day
    X_train, Y_train = buildTrain(train_norm, pastDay, 1)
    X_train, Y_train = shuffle(X_train, Y_train)
    # because no return sequence, Y_train and Y_val shape must be 2 dimension
    X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

    model = models.buildManyToOneModel(X_train.shape)
    callback = EarlyStopping(
        monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(X_train, Y_train, epochs=1000, batch_size=128,
              validation_data=(X_val, Y_val), callbacks=[callback])
    return model


def lstm_stock(pastDay=30):
    train = readTrain()[:-90]
    train_Aug = augFeatures(train)
    train_norm = normalize(train_Aug)
    # change the last day and next day
    X_train, Y_train = buildTrain(train_norm, pastDay, 1)
    X_train, Y_train = shuffle(X_train, Y_train)
    # because no return sequence, Y_train and Y_val shape must be 2 dimension
    X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

    model = models.lstm_stock_model(X_train.shape)
    callback = EarlyStopping(
        monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(X_train, Y_train, epochs=1000, batch_size=128,
              validation_data=(X_val, Y_val), callbacks=[callback])
    return model


if __name__ == "__main__":
    m = lstm_stock(30)
    m.summary()
    with open("model.json", "w") as mfile:
        mfile.write(m.to_json())
