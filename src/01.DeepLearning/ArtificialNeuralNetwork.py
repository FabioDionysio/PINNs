
# %%
# Imports
from sklearn.model_selection import train_test_split  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load data
x_l = np.load('input/X.npy')
y_l = np.load('input/Y.npy')

plt.subplot(121)
plt.imshow(x_l[260].reshape(64, 64))
plt.axis('off')
plt.subplot(122)
plt.imshow(x_l[900].reshape(64, 64))
plt.axis('off')

# Concatenando apenas os 0 e 1 on um array
X = np.concatenate((x_l[204:409], x_l[822:1027]), axis=0)
Y = np.concatenate((np.zeros(205), np.ones(205)),
                   axis=0).reshape(X.shape[0], 1)
print('X shape:', X.shape)
print('Y shape:', Y.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42)
# Flatten the images
X_train_flatten = X_train.reshape(
    X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test.reshape(
    X_test.shape[0], X_test.shape[1]*X_test.shape[2])


# Transpose data
x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = y_train.T
y_test = y_test.T
print('X_Train:', X_train.shape, 'X_Test:', X_test.shape,
      'y_train:', y_train.shape, 'y_shape:', y_test.shape)

# %%
# Initializing Parameters


def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    parameters = {'weight1': np.random.randn(3, x_train.shape[0]) * 0.1,
                  'bias1': np.zeros((3, 1)),
                  'weight2': np.random.randn(y_train.shape[0], 3) * 0.1,
                  'bias2': np.zeros((y_train.shape[0], 1))}
    return parameters


def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head


def forward_propagation_NN(x_train, parameters):
    z1 = np.dot(parameters['weight1'], x_train) + parameters['bias1']
    a1 = np.tanh(z1)
    z2 = np.dot(parameters['weight2'], a1) + parameters['bias2']
    a2 = sigmoid(z2)
    cache = {'Z1': z1, 'A1': a1, 'Z2': z2, 'A2': a2}
    return a2, cache


def compute_cost_NN(a2, Y, paramters):
    logprobs = np.multiply(np.log(a2), Y)
    cost = - np.sum(logprobs) / Y.shape[1]
    return cost


def backward_propagation_NN(parameters, cache, X, Y):
    dz2 = cache['A2'] - Y
    dw2 = np.dot(dz2, cache['A1'].T) / X.shape[1]
    db2 = np.sum(dz2, axis=1, keepdims=True) / X.shape[1]
    dz1 = np.dot(parameters['weight2'].T, dz2) * (1 - np.power(cache['A1'], 2))
    dw1 = np.dot(dz1, X.T) / X.shape[1]
    db1 = np.sum(dz1, axis=1, keepdims=True) / X.shape[1]
    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}
    return grads


def update_parameters_NN(parameters, grads, learning_rate=0.01):
    parameters = {'weight1': parameters['weight1'] - learning_rate * grads['dw1'],
                  'bias1': parameters['bias1'] - learning_rate * grads['db1'],
                  'weight2': parameters['weight2'] - learning_rate * grads['dw2'],
                  'bias2': parameters['bias2'] - learning_rate * grads['db2']}
    return parameters


def predict_NN(parameters, x_test):
    a2, cache = forward_propagation_NN(x_test, parameters)
    y_prediction = np.zeros((1, x_test.shape[1]))
    for i in range(a2.shape[1]):
        if a2[0, i] <= 0.5:
            y_prediction[0, i] = 0
        else:
            y_prediction[0, i] = 1
    return y_prediction


def two_layer_neural_network(x_train, y_train, x_test, y_test, num_iterations):
    cost_list = []
    index_list = []
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)
    for i in range(0, num_iterations):
        a2, cache = forward_propagation_NN(x_train, parameters)
        cost = compute_cost_NN(a2, y_train, parameters)
        grads = backward_propagation_NN(parameters, cache, x_train, y_train)
        parameters = update_parameters_NN(parameters, grads)
        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print('Cost after iteration %i: %f' % (i, cost))
    plt.plot(index_list, cost_list)
    plt.xticks(index_list, rotation='vertical')
    plt.xlabel('Number of Iterarion')
    plt.ylabel('Cost')
    plt.show()

    y_prediction_test = predict_NN(parameters, x_test)
    y_prediction_train = predict_NN(parameters, x_train)

    print('train accuracy: {} %'.format(
        100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print('test accuracy: {} %'.format(
        100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters


# %%
ann = two_layer_neural_network(
    x_train, y_train, x_test, y_test, num_iterations=2500)
