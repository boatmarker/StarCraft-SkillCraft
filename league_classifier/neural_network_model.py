import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return z * (z > 0)


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def relu_derivative(z):
    return (z > 0).astype(int)


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


def activation_forward_sigmoid(Z):
    """
    Compute one activation step A[l] = g[l](Z[l]) using sigmoid function.
    :param Z: input to the activation function for the current layer, shape layer_dims[l] x m
    :return A: activation matrix for the current layer, shape layer_dims[l] x m
    """
    return sigmoid(Z)


def activation_forward_relu(Z):
    """
    Compute one activation step A[l] = g[l](Z[l]) using ReLU function.
    :param Z: input to the activation function for the current layer, shape layer_dims[l] x m
    :return A: activation matrix for the current layer, shape layer_dims[l] x m
    """
    return relu(Z)


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
        if l < L - 1:
            A = activation_forward_relu(Z)
        else:
            A = activation_forward_sigmoid(Z)
        A_list.append(A)

    return Z_list, A_list


def compute_cost(AL, Y, W_list, lambd):
    """
    Compute the cost from the activation matrix from the Lth layer.
    :param AL: activation matrix for the output layer, shape layer_dims[L] x m = 1 x m for binary classification
    :param Y: labels, shape layer_dims[L] x m = 1 x m for binary classification
    :param W_list:
    :param lambd:
    :return cost: cost
    """
    m = Y.shape[1]

    # to prevent log(0) errors
    epsilon = 1e-8
    AL[AL == 0] = epsilon
    AL[AL == 1] = 1 - epsilon

    cross_entropy_cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)).squeeze()
    L2_regularization_cost = lambd / (2 * m) * sum(np.sum(np.square(W)).squeeze() for W in W_list)

    return cross_entropy_cost + L2_regularization_cost


def linear_backward(dZ, A_prev, W, lambd):
    """
    Compute one backward linear step, calculate dW, db, dA_prev.
    :param dZ: derivative of cost with respect to Z of current layer, shape layer_dims[l] x m
    :param A_prev: activation matrix from previous layer, shape layer_dims[l-1] x m
    :param W: weight matrix for current layer, shape layer_dims[l] x layer_dims[l-1]
    :param lambd:
    :return dW:
    :return db:
    :return dA_prev:
    """
    m = dZ.shape[1]
    dW = 1/m * np.dot(dZ, A_prev.T) + lambd/m * W
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dW, db, dA_prev


def activation_backward_sigmoid(dA, Z):
    """

    :param dA:
    :param Z:
    :return dZ:
    """
    return dA * sigmoid_derivative(Z)


def activation_backward_relu(dA, Z):
    """

    :param dA:
    :param Z:
    :return dZ:
    """
    return dA * relu_derivative(Z)


def model_backward(W_list, Z_list, A_list, Y, lambd):
    """

    :param W_list: L
    :param Z_list: L
    :param A_list: L+1
    :param Y: 1 x m
    :param lambd:
    :return dA_list: L+1
    :return dW_list: L
    :return db_list: L
    """
    L = len(W_list)
    A = A_list[L]

    # to prevent divide by 0 errors
    epsilon = 1e-8
    A[A == 0] = epsilon
    A[A == 1] = 1 - epsilon

    dA = -Y/A + (1 - Y)/(1 - A)
    dA_list = [dA]
    dW_list = []
    db_list = []
    for l in range(L - 1, -1, -1):
        Z = Z_list[l]
        W = W_list[l]
        A_prev = A_list[l]
        if l < L - 1:
            dZ = activation_backward_relu(dA, Z)
        else:
            dZ = activation_backward_sigmoid(dA, Z)
        dW, db, dA_prev = linear_backward(dZ, A_prev, W, lambd)
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


def multi_class_predict(W_list_list, b_list_list, X):
    """
    :param W_list_list: K
    :param b_list_list: K
    :param X: input data, shape n x m
    :return Y_prediction: predicted labels, shape 1 x m
    """
    K = len(W_list_list)
    m = X.shape[1]
    A_aggregate = np.zeros((K, m))  # activations from all K classifiers stacked together, shape K x m

    for k in range(K):
        Z_list, A_list = model_forward(X, W_list_list[k], b_list_list[k])
        AL = A_list[len(A_list) - 1]
        A_aggregate[k, :] = AL

    Y_prediction = np.argmax(A_aggregate, axis=0) + 1
    Y_prediction = np.reshape(Y_prediction, (1, Y_prediction.size))

    return Y_prediction


def optimize(W_list, b_list, train_X, train_Y, num_iterations, learning_rate, lambd, print_cost=False):
    for i in range(num_iterations):
        Z_list, A_list = model_forward(train_X, W_list, b_list)
        if print_cost and i % 100 == 0:
            AL = A_list[len(A_list) - 1]
            print(f"Cost after iteration {i}: {compute_cost(AL, train_Y, W_list, lambd)}")

        dA_list, dW_list, db_list = model_backward(W_list, Z_list, A_list, train_Y, lambd)
        W_list, b_list = update_parameters(W_list, b_list, dW_list, db_list, learning_rate)

    return W_list, b_list


def binary_model(layer_dims, train_X, train_Y, test_X, test_Y, num_iterations, learning_rate, lambd, print_cost=False):
    """
    Learn a binary classifier and test its accuracy on the test set.
    :param layer_dims:
    :param train_X: training data, shape n x m_train
    :param train_Y: training labels (binary 0 or 1), shape 1 x m_train
    :param test_X: test data, shape n x m_test
    :param test_Y: test labels (binary 0 or 1), shape 1 x m_test
    :param num_iterations: number of iterations to run gradient descent
    :param learning_rate: learning rate (alpha)
    :param lambd:
    :param print_cost: print calculated cost every 100 iterations for tracking progress
    """
    W_list, b_list = initialize_parameters(layer_dims)
    W_list, b_list = optimize(W_list, b_list, train_X, train_Y, num_iterations, learning_rate, lambd, print_cost)

    Y_prediction_train = binary_predict(W_list, b_list, train_X)
    Y_prediction_test = binary_predict(W_list, b_list, test_X)

    print(f"Training set prediction accuracy: {100 - np.mean(np.abs(train_Y - Y_prediction_train)) * 100}%")
    print(f"Test set prediction accuracy: {100 - np.mean(np.abs(test_Y - Y_prediction_test)) * 100}%")


def multi_class_model(layer_dims, num_classes, train_X, train_Y, test_X, test_Y, num_iterations, learning_rate, lambd, print_cost=False):
    """
    Learn a multi-class classifier and test its accuracy on the test set.
    :param layer_dims:
    :param num_classes: number of distinct classes (K)
    :param train_X: training data, shape n x m_train
    :param train_Y: training labels (multi-class 1 to K), shape 1 x m_train
    :param test_X: test data, shape n x m_test
    :param test_Y: test labels (multi-class 1 to K), shape 1 x m_test
    :param num_iterations: number of iterations to run gradient descent
    :param learning_rate: learning rate (alpha)
    :param lambd:
    :param print_cost: print calculated cost every 100 iterations for tracking progress
    """
    m_train = train_X.shape[1]
    m_test = test_X.shape[1]
    W_list_list = []  # list of weight matrix lists for all K binary classifiers
    b_list_list = []  # list of bias unit lists for all K binary classifiers

    for k in range(1, num_classes + 1):
        print(f"Training classifier for league {k}")
        train_Y_k = (train_Y == k).astype(int)
        W_list, b_list = initialize_parameters(layer_dims)
        W_list, b_list = optimize(W_list, b_list, train_X, train_Y_k, num_iterations, learning_rate, lambd, print_cost)
        W_list_list.append(W_list)
        b_list_list.append(b_list)

    Y_prediction_train = multi_class_predict(W_list_list, b_list_list, train_X)
    Y_prediction_test = multi_class_predict(W_list_list, b_list_list, test_X)

    print("Predicted Y for test set:")
    print(Y_prediction_test)

    print(f"Training set prediction accuracy: {sum((Y_prediction_train == train_Y).squeeze().astype(int)) / m_train * 100}%")
    print(f"Test set prediction accuracy: {sum((Y_prediction_test == test_Y).squeeze().astype(int)) / m_test * 100}%")
