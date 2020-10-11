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


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
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


def binary_predict(w, b, X):
    """
    Calculate predicted labels given weights, bias, and input data.
    :param w: weights
    :param b: bias
    :param X: input data
    :return Y_prediction: predicted labels
    """
    A = sigmoid(np.dot(w.T, X) + b)
    Y_prediction = (A > 0.5).astype(int)

    return Y_prediction


def multi_class_predict(w_aggregate, b_aggregate, X):
    """
    Calculate multi-class predicted labels given weights, bias, and input data.
    :param w_aggregate: weights vectors for all K binary classifiers stacked together, shape n x K
    :param b_aggregate: bias units for all K binary classifiers stacked together, shape K x 1
    :param X: input data, shape n x m
    :return Y_prediction: predicted labels, shape 1 x m
    """
    # activations from all K classifiers stacked together, shape K x m
    A_aggregate = sigmoid(np.dot(w_aggregate.T, X) + b_aggregate)
    Y_prediction = np.argmax(A_aggregate, axis=0) + 1
    Y_prediction = np.reshape(Y_prediction, (1, Y_prediction.size))

    return Y_prediction


def binary_model(train_X, train_Y, test_X, test_Y, num_iterations, learning_rate, print_cost=False):
    """
    Learn a binary classifier and test its accuracy on the test set.
    :param train_X: training data, shape n x m_train
    :param train_Y: training labels (binary 0 or 1), shape 1 x m_train
    :param test_X: test data, shape n x m_test
    :param test_Y: test labels (binary 0 or 1), shape 1 x m_test
    :param num_iterations: number of iterations to run gradient descent
    :param learning_rate: learning rate (alpha)
    :param print_cost: print calculated cost every 100 iterations for tracking progress
    """
    n = train_X.shape[0]
    w, b = initialize_parameters(n)
    w, b = optimize(w, b, train_X, train_Y, num_iterations, learning_rate, print_cost)

    Y_prediction_train = binary_predict(w, b, train_X)
    Y_prediction_test = binary_predict(w, b, test_X)

    print(f"Training set prediction accuracy: {100 - np.mean(np.abs(train_Y - Y_prediction_train)) * 100}%")
    print(f"Test set prediction accuracy: {100 - np.mean(np.abs(test_Y - Y_prediction_test)) * 100}%")


def multi_class_model(num_classes, train_X, train_Y, test_X, test_Y, num_iterations, learning_rate, print_cost=False):
    """
    Learn a multi-class classifier and test its accuracy on the test set.
    :param num_classes: number of distinct classes (K)
    :param train_X: training data, shape n x m_train
    :param train_Y: training labels (multi-class 1 to K), shape 1 x m_train
    :param test_X: test data, shape n x m_test
    :param test_Y: test labels (multi-class 1 to K), shape 1 x m_test
    :param num_iterations: number of iterations to run gradient descent
    :param learning_rate: learning rate (alpha)
    :param print_cost: print calculated cost every 100 iterations for tracking progress
    """
    n = train_X.shape[0]
    m_train = train_X.shape[1]
    m_test = test_X.shape[1]
    w_aggregate = np.zeros((n, num_classes))  # weights vectors for all K binary classifiers stacked together
    b_aggregate = np.zeros((num_classes, 1))  # bias units for all K binary classifiers stacked together

    for k in range(1, num_classes + 1):
        print(f"Training classifier for league {k}")
        train_Y_k = (train_Y == k).astype(int)
        w, b = initialize_parameters(n)
        w, b = optimize(w, b, train_X, train_Y_k, num_iterations, learning_rate, print_cost)
        w_aggregate[:, k - 1] = np.reshape(w, (n,))
        b_aggregate[k - 1, 0] = b

    Y_prediction_train = multi_class_predict(w_aggregate, b_aggregate, train_X)
    Y_prediction_test = multi_class_predict(w_aggregate, b_aggregate, test_X)

    print("Predicted Y for test set:")
    print(Y_prediction_test)

    print(f"Training set prediction accuracy: {sum((Y_prediction_train == train_Y).squeeze().astype(int)) / m_train * 100}%")
    print(f"Test set prediction accuracy: {sum((Y_prediction_test == test_Y).squeeze().astype(int)) / m_test * 100}%")


train_data = pd.read_csv("data/train_rows.csv")
test_data = pd.read_csv("data/test_rows.csv")
all_data = pd.concat([train_data, test_data])
num_train_rows = train_data.shape[0]

all_Y = np.reshape(np.array(all_data["LeagueIndex"]), (1, -1))
train_data_Y = all_Y[:, :num_train_rows]  # shape 1 x m_train
test_data_Y = all_Y[:, num_train_rows:]  # shape 1 x m_test

# remove useless columns and scale data
del all_data["Unnamed: 0"]
del all_data["GameID"]
del all_data["LeagueIndex"]
all_X = np.array(all_data).T
all_X = StandardScaler().fit_transform(all_X)
train_data_X = all_X[:, :num_train_rows]  # shape n x m_train
test_data_X = all_X[:, num_train_rows:]  # shape n x m_test

multi_class_model(7, train_data_X, train_data_Y, test_data_X, test_data_Y, 30000, 0.1, True)
