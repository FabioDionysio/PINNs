# %%
# Imports

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn import linear_model  # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %%
# Load data
x_l = np.load('input/X.npy')
y_l = np.load('input/Y.npy')

plt.subplot(121)
plt.imshow(x_l[260].reshape(64, 64))
plt.axis('off')
plt.subplot(122)
plt.imshow(x_l[900].reshape(64, 64))
plt.axis('off')

# %%
# Concatenando apenas os 0 e 1 on um array
X = np.concatenate((x_l[204:409], x_l[822:1027]), axis=0)
Y = np.concatenate((np.zeros(205), np.ones(205)),
                   axis=0).reshape(X.shape[0], 1)
print('X shape:', X.shape)
print('Y shape:', Y.shape)

# %%
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.15, random_state=42)
# Flatten the images
X_train_flatten = X_train.reshape(
    X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test.reshape(
    X_test.shape[0], X_test.shape[1]*X_test.shape[2])

# %%
# Transpose data
x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = y_train.T
y_test = y_test.T
print('X train shape:', X_train.shape, 'X test shape:', X_test.shape,
      'y train shape:', y_train.shape, 'y test shape:', y_test.shape)

# %%
# Initializing Parameters


def initialize_weights_bias(dimension, w_init=0.01, b_init=0):
    w = np.full((dimension, 1), w_init)
    b = b_init
    return w, b

# Forward Propagation


def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head


def forwand_propagation(w, b, x_train, y_train):
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -(1 - y_train) * np.log(1 - y_head) - y_train * np.log(y_head)
    cost = np.sum(loss) / x_train.shape[1]
    return cost

# Forward and Backward Propagation


def forward_backward_propagation(w, b, x_train, y_train):
    # Forward Propagation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -(1 - y_train) * np.log(1 - y_head) - y_train * np.log(y_head)
    cost = np.sum(loss) / x_train.shape[1]
    # Backward Propagation
    derivative_weight = np.dot(
        x_train, ((y_head - y_train).T)) / x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]
    gradients = {'derivative_weight': derivative_weight,
                 'derivative_bias': derivative_bias}
    return cost, gradients

# Update Parameters


def update(w, b, x_train, y_train, learning_rate, number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    # Updating (learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        # Update
        w = w - learning_rate * gradients['derivative_weight']
        b = b - learning_rate * gradients['derivative_bias']
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print('Cost after iteration %i: %f' % (i, cost))
    # We update (learn) parameters weights and bias
    parameters = {'weight': w, 'bias': b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel('Number of Iterarion')
    plt.ylabel('Cost')
    plt.show()
    return parameters, gradients, cost_list

# Prediction


def predict(w, b, x_test):
    z = sigmoid(np.dot(w.T, x_test) + b)
    y_prediction = np.zeros((1, x_test.shape[1]))
    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            y_prediction[0, i] = 0
        else:
            y_prediction[0, i] = 1
    return y_prediction


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, number_of_iterarion):
    dimension = x_train.shape[0]
    w, b = initialize_weights_bias(dimension)
    parameters, gradients, cost_list = update(
        w, b, x_train, y_train, learning_rate, number_of_iterarion)
    y_prediction_test = predict(
        parameters['weight'], parameters['bias'], x_test)
    print('Test Accuracy: {:.2f}%'.format(
        100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


# %%
logistic_regression(x_train, y_train, x_test, y_test,
                    learning_rate=0.01, number_of_iterarion=150)


# %%
# Sklearn
logreg = linear_model.LogisticRegression(random_state=42, max_iter=150)
print("train accuracy: {} ".format(logreg.fit(
    x_train.T, y_train.T).score(x_train.T, y_train.T)))
print('Test Accuracy: {}'.format(logreg.fit(
    x_train.T, y_train.T).score(x_test.T, y_test.T)))
