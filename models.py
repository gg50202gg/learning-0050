from keras import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.layers import RepeatVector


def buildManyToManyModel(shape):
    model = Sequential()
    model.add(
        LSTM(10, input_length=shape[1], input_dim=shape[2], return_sequences=True))
    # output shape: (5, 1)
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model


def buildOneToManyModel(shape):
    model = Sequential()
    model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
    # output shape: (5, 1)
    model.add(Dense(1))
    model.add(RepeatVector(5))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model


def buildManyToOneModel(shape):
    model = Sequential()
    model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
    # output shape: (1, 1)
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model


def buildOneToOneModel(shape):
    model = Sequential()
    model.add(LSTM(10, input_length=shape[1],
                   input_dim=shape[2], return_sequences=True))
    # output shape: (1, 1)
    model.add(TimeDistributed(Dense(1)))    # or use model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model


def lstm_stock_model(shape):
    """
    資料模型
    shape [seq_len, data]
    """
    model = Sequential()
    model.add(
        LSTM(
            256,
            input_shape=(shape[1], shape[2]),
            return_sequences=True
        )
    )

    model.add(LSTM(1, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(5, activation='linear'))
    model.add(Dense(1))
    model.compile(
        loss="mean_absolute_error",
        optimizer="adam",
        metrics=['mean_absolute_error']
    )
    model.summary()
    return model
