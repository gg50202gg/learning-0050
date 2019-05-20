import pandas as pd
import numpy as np


def readTrain(file):
    """ 讀取資料 """
    train = pd.read_csv(file)
    return train


def augFeatures(train):
    """ 日期正規 """
    train["Date"] = pd.to_datetime(train["Date"])
    train["year"] = train["Date"].dt.year
    train["month"] = train["Date"].dt.month
    train["date"] = train["Date"].dt.day
    train["day"] = train["Date"].dt.dayofweek
    return train


def normalize(train):
    """ 資料正規 """
    train = train.drop(["Date"], axis=1)
    train_norm = train.apply(lambda x: (
        x - np.mean(x)) / (np.max(x) - np.min(x)))
    return train_norm


def train_test_data(train):
    """ 資料切割 學習/測試 """
    train_size = int(len(train) * 0.66)
    return [train[:train_size], train[train_size+1:]]


def buildTrain(train, pastDay=30, futureDay=5):
    """ 建立訓練資料 """
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
    """ 訓練資料 驗證資料 """
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val
