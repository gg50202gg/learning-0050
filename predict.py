import keras
from funtions import *

csv_path = "TW50.csv"
model_path = "model.json"

train_Org = readTrain(csv_path)  # 讀取
train_Aug = augFeatures(train_Org)  # 日期
train_norm = normalize(train_Aug)  # 正規化
train, test = train_test_data(train_norm)

with open(model_path) as fmodel:
    model = keras.models.model_from_json(fmodel.read())


if __name__ == "__main__":
    model.predict(test)
    pass
