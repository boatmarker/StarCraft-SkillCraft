import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import neural_network_model as nm

train_data = pd.read_csv("data/train_rows.csv")
test_data = pd.read_csv("data/test_rows.csv")

train_data_Y = np.reshape(np.array(train_data["LeagueIndex"]), (1, -1))  # shape 1 x m_train
test_data_Y = np.reshape(np.array(test_data["LeagueIndex"]), (1, -1))  # shape 1 x m_test

# remove useless columns
del train_data["Unnamed: 0"]
del train_data["GameID"]
del train_data["LeagueIndex"]
del test_data["Unnamed: 0"]
del test_data["GameID"]
del test_data["LeagueIndex"]

train_data_X = np.array(train_data)
test_data_X = np.array(test_data)

# scale data
scaler = StandardScaler()
train_data_X = scaler.fit_transform(train_data_X)
test_data_X = scaler.transform(test_data_X)

train_data_X = train_data_X.T  # shape n x m_train
test_data_X = test_data_X.T  # shape n x m_test

layer_dims = [train_data_X.shape[0], 15, 10, 1]
num_leagues = 7
nm.multi_class_model(layer_dims, num_leagues, train_data_X, train_data_Y, test_data_X, test_data_Y, 20000, 0.5, 0.05, True)
