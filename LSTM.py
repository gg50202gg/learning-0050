import models
from funtions import *

from keras.callbacks import EarlyStopping


def f1t1(train):
    # change the last day and next day
    X_train, Y_train = buildTrain(train, 1, 1)
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


def fnt1(train, pastDay=30):
    # change the last day and next day
    X_train, Y_train = buildTrain(train, pastDay, 1)
    X_train, Y_train = shuffle(X_train, Y_train)
    # because no return sequence, Y_train and Y_val shape must be 2 dimension
    X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

    model = models.buildManyToOneModel(X_train.shape)
    callback = EarlyStopping(
        monitor="loss", patience=10, verbose=1, mode="auto")
    model.fit(X_train, Y_train, epochs=1000, batch_size=128,
              validation_data=(X_val, Y_val), callbacks=[callback])
    return model


def lstm_stock(train, pastDay=30):
    # change the last day and next day
    X_train, Y_train = buildTrain(train, pastDay, 1)
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
    file = "TW50.csv"
    pastDay = 30

    train_Org = readTrain(file)  # 讀取
    train_Aug = augFeatures(train_Org)  # 日期
    train_norm = normalize(train_Aug)  # 正規化
    train, test = train_test_data(train_norm)

    m = lstm_stock(train, 30)
    m.summary()
    with open(file + "model.json", "w") as mfile:
        mfile.write(m.to_json())
