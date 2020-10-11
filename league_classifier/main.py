import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_parameters(dim):
    """
    Initialize weights vector w and bias (scalar) b with 0's.
    :param dim: length of w vector; same as number of features n
    :return w: weights vector with shape n x 1
    :return b: bias (scalar)
    """
    return np.zeros((dim, 1)), 0

def propagate(w, b, X, Y):
    """
    Calculate gradients and cost.
    :param w: weights vector
    :param b: bias (scalar)
    :param X: training data with shape n x m
    :param Y: labels with shape 1 x m
    :return cost: cost (J) calculated from labels (Y) vs. calculated activations (A)
    :return dJdw: gradient of loss with respect to w, shape n x 1
    :return dJdb: gradient of loss with respect to b, scalar
    """
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)  # shape 1 x m
    cost = -1/m * (np.dot(Y, np.log(A).T) + np.dot(1 - Y, np.log(1 - A).T))
    cost = np.squeeze(cost)

    dJdw = 1/m * np.dot(X, (A - Y).T)
    dJdb = 1/m * np.sum(A - Y)

    return cost, dJdw, dJdb

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    Run gradient descent for num_iterations to optimize the parameters w and b.
    :param w: initial weights
    :param b: initial bias
    :param X: training data
    :param Y: labels
    :param num_iterations: number of iterations to run gradient descent
    :param learning_rate: learning rate (alpha)
    :param print_cost: print calculated cost every 100 iterations for tracking progress
    :return w: final optimized weights
    :return b: final optimized bias
    """
    for i in range(num_iterations):
        cost, dJdw, dJdb = propagate(w, b, X, Y)

        w = w - learning_rate * dJdw
        b = b - learning_rate * dJdb

        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")

    return w, b

def predict(w, b, X):
    """
    Calculate predicted labels given the weights, bias, and input data.
    :param w: weights
    :param b: bias
    :param X: input data
    :return Y_prediction: predicted labels
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    Y_prediction = (A > 0.5).astype(int)

    return Y_prediction

def model(train_X, train_Y, test_X, test_Y, num_iterations, learning_rate, print_cost = False):
    """

    :param X_train:
    :param Y_train:
    :param X_test:
    :param Y_test:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    """
    n = train_X.shape[0]
    w, b = initialize_parameters(n)
    w, b = optimize(w, b, train_X, train_Y, num_iterations, learning_rate, print_cost)

    Y_prediction_train = predict(w, b, train_X)
    Y_prediction_test = predict(w, b, test_X)

    print(f"Training set prediction accuracy: {100 - np.mean(np.abs(train_Y - Y_prediction_train)) * 100}%")
    print(f"Test set prediction accuracy: {100 - np.mean(np.abs(test_Y - Y_prediction_test)) * 100}%")


train_data = pd.read_csv("data/train_rows.csv")
test_data = pd.read_csv("data/test_rows.csv")
all_data = pd.concat([train_data, test_data])
num_train_rows = train_data.shape[0]

# train Bronze League classifier
all_Y = np.reshape(np.array(all_data["LeagueIndex"]), (1, -1))
all_Y = (all_Y == 1).astype(int)
train_Y = all_Y[:, :num_train_rows]  # shape 1 x m_train
test_Y = all_Y[:, num_train_rows:]  # shape 1 x m_test

# remove useless columns and scale data
del all_data["Unnamed: 0"]
del all_data["GameID"]
del all_data["LeagueIndex"]
all_X = np.array(all_data).T
all_X = StandardScaler().fit_transform(all_X)
train_X = all_X[:, :num_train_rows]  # shape n x m_train
test_X = all_X[:, num_train_rows:]  # shape n x m_test

model(train_X, train_Y, test_X, test_Y, 1000, 0.1, True)
