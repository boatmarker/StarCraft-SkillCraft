import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def initialize_parameters(layer_dims):
    """
    Randomly initialize weights matrices W and bias vectors b for all layers.
    :param layer_dims: list containing the size of each layer in the network (starting with the input layer), total L+1 layers (input layer not included in L)
    :return W_list: list of L weight matrices (excludes input layer), W for lth layer is shaped layer_dims[l] x layer_dims[l-1]
    :return b_list: list of L bias vectors (excludes input layer), b for lth layer is shaped layer_dims[l] x 1
    """
    L = len(layer_dims) - 1
    W_list = []
    b_list = []

    for l in range(1, L + 1):
        W_list.append(np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01)
        b_list.append(np.zeros((layer_dims[l], 1)))

    return W_list, b_list


def linear_forward(A_prev, W, b):
    """
    Compute one linear forward step Z[l] = W[l].A[l-1] + b[l].
    :param A_prev: activation matrix from previous layer, shape layer_dims[l-1] x m
    :param W: weights matrix for current layer, shape layer_dims[l] x layer_dims[l-1]
    :param b: bias vector for current layer, shape layer_dims[l] x 1
    :return Z: input to the activation function for current layer, shape layer_dims[l] x m
    """
    return np.dot(W, A_prev) + b


def activation_forward(Z):
    """
    Compute one activation step A[l] = g[l](Z[l]).
    :param Z: input to the activation function for the current layer, shape layer_dims[l] x m
    :return A: activation matrix for the current layer, shape layer_dims[l] x m
    """
    return sigmoid(Z)


def model_forward(X, W_list, b_list):
    """
    Forward propagation calculations.
    :param X: input data, shape layer_dims[0] x m
    :param W_list: list of L weights matrices, starting with layer 1 (first hidden layer), W for lth layer is shaped layer_dims[l] x layer_dims[l-1]
    :param b_list: list of L bias vectors, starting with layer 1, b for lth layer is shaped layer_dims[l] x 1
    :return Z_list: list of L inputs to activation functions, starting with layer 1, Z for lth layer is shaped layer_dims[l] x m
    :return A_list: list of L+1 activation matrices, starting with the input layer, A for lth layer is shaped layer_dims[l] x m
    """
    L = len(W_list)
    A = X
    Z_list = []
    A_list = [A]
    for l in range(L):
        A_prev = A
        W = W_list[l]
        b = b_list[l]
        Z = linear_forward(A_prev, W, b)
        Z_list.append(Z)
        A = activation_forward(Z)
        A_list.append(A)

    return Z_list, A_list


def compute_cost(AL, Y):
    """
    Compute the cost from the activation matrix from the Lth layer.
    :param AL: activation matrix for the output layer, shape layer_dims[L] x m = 1 x m for binary classification
    :param Y: labels, shape layer_dims[L] x m = 1 x m for binary classification
    :return cost: cost
    """
    m = Y.shape[1]

    # to prevent log(0) errors
    epsilon = 1e-8
    AL[AL == 0] = epsilon
    AL[AL == 1] = 1 - epsilon

    return -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)).squeeze()


def linear_backward(dZ, A_prev, W):
    """
    Compute one backward linear step, calculate dW, db, dA_prev.
    :param dZ: derivative of cost with respect to Z of current layer, shape layer_dims[l] x m
    :param A_prev: activation matrix from previous layer, shape layer_dims[l-1] x m
    :param W: weight matrix for current layer, shape layer_dims[l] x layer_dims[l-1]
    :return dW:
    :return db:
    :return dA_prev:
    """
    m = dZ.shape[1]
    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dW, db, dA_prev


def activation_backward(dA, Z):
    """

    :param dA:
    :param Z:
    :return dZ:
    """
    return dA * sigmoid_derivative(Z)


def model_backward(W_list, Z_list, A_list, Y):
    """

    :param W_list: L
    :param Z_list: L
    :param A_list: L+1
    :param Y: 1 x m
    :return dA_list: L+1
    :return dW_list: L
    :return db_list: L
    """
    L = len(W_list)

    A = A_list[L]
    dA = -Y/A + (1 - Y)/(1 - A)
    dA_list = [dA]
    dW_list = []
    db_list = []
    for l in range(L - 1, -1, -1):
        Z = Z_list[l]
        W = W_list[l]
        A_prev = A_list[l]
        dZ = activation_backward(dA, Z)
        dW, db, dA_prev = linear_backward(dZ, A_prev, W)
        dA_list.insert(0, dA_prev)
        dW_list.insert(0, dW)
        db_list.insert(0, db)
        dA = dA_prev

    return dA_list, dW_list, db_list


def update_parameters(W_list, b_list, dW_list, db_list, learning_rate):
    """

    :param W_list: L
    :param b_list: L
    :param dW_list: L
    :param db_list: L
    :param learning_rate:
    :return:
    """
    for l in range(len(W_list)):
        W_list[l] = W_list[l] - learning_rate * dW_list[l]

    for l in range(len(b_list)):
        b_list[l] = b_list[l] - learning_rate * db_list[l]

    return W_list, b_list


def binary_predict(W_list, b_list, X):
    """
    :param W_list:
    :param b_list:
    :param X: input data
    :return Y_prediction: predicted labels
    """
    Z_list, A_list = model_forward(X, W_list, b_list)
    AL = A_list[len(A_list) - 1]
    Y_prediction = (AL > 0.5).astype(int)

    return Y_prediction


def binary_model(layer_dims, train_X, train_Y, test_X, test_Y, num_iterations, learning_rate, print_cost=False):
    """
    Learn a binary classifier and test its accuracy on the test set.
    :param layer_dims:
    :param train_X: training data, shape n x m_train
    :param train_Y: training labels (binary 0 or 1), shape 1 x m_train
    :param test_X: test data, shape n x m_test
    :param test_Y: test labels (binary 0 or 1), shape 1 x m_test
    :param num_iterations: number of iterations to run gradient descent
    :param learning_rate: learning rate (alpha)
    :param print_cost: print calculated cost every 100 iterations for tracking progress
    """
    W_list, b_list = initialize_parameters(layer_dims)
    for i in range(num_iterations):
        Z_list, A_list = model_forward(train_X, W_list, b_list)
        if print_cost and i % 100 == 0:
            AL = A_list[len(A_list) - 1]
            print(f"Cost after iteration {i}: {compute_cost(AL, train_Y)}")

        dA_list, dW_list, db_list = model_backward(W_list, Z_list, A_list, train_Y)
        W_list, b_list = update_parameters(W_list, b_list, dW_list, db_list, learning_rate)

    Y_prediction_train = binary_predict(W_list, b_list, train_X)
    Y_prediction_test = binary_predict(W_list, b_list, test_X)

    print(f"Training set prediction accuracy: {100 - np.mean(np.abs(train_Y - Y_prediction_train)) * 100}%")
    print(f"Test set prediction accuracy: {100 - np.mean(np.abs(test_Y - Y_prediction_test)) * 100}%")


train_data = pd.read_csv("data/train_rows.csv")
test_data = pd.read_csv("data/test_rows.csv")

league = 4
train_data_Y = np.reshape(np.array(train_data["LeagueIndex"]), (1, -1))  # shape 1 x m_train
train_data_Y = (train_data_Y == league).astype(int)
test_data_Y = np.reshape(np.array(test_data["LeagueIndex"]), (1, -1))  # shape 1 x m_test
test_data_Y = (test_data_Y == league).astype(int)

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

binary_model([train_data_X.shape[0], 15, 10, 1], train_data_X, train_data_Y, test_data_X, test_data_Y, 20000, 0.5, True)
