import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')
data = data.sample(frac=1).reset_index(drop=True)
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, X):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def cost_function(A2, Y):
    one_hot_Y = np.eye(10)[Y].T
    return -np.mean(np.sum(one_hot_Y * np.log(A2), axis=0))


def initialize_horses(population_size, num_params):
    return [np.random.uniform(-0.5, 0.5, num_params) for _ in range(population_size)]


def update_horses(horses, alpha, leader_index):
    for i, horse in enumerate(horses):
        if i != leader_index:
            direction = horses[leader_index] - horse
            perturbation = np.random.uniform(-alpha, alpha, horse.shape)
            horses[i] += direction + perturbation


def train_neural_network_with_who(X_train, Y_train, alpha, iterations):
    population_size = 10
    num_params_W1 = 50 * 784
    num_params_b1 = 50
    num_params_W2 = 10 * 50
    num_params_b2 = 10
    num_params = num_params_W1 + num_params_b1 + num_params_W2 + num_params_b2
    horses = initialize_horses(population_size, num_params)
    best_horse_index = None
    best_cost = float('inf')
    for i in range(iterations):
        for j, horse in enumerate(horses):
            index = 0
            W1_horse = horse[index:index + num_params_W1].reshape((50, 784))
            index += num_params_W1
            b1_horse = horse[index:index + num_params_b1].reshape((50, 1))
            index += num_params_b1
            W2_horse = horse[index:index + num_params_W2].reshape((10, 50))
            index += num_params_W2
            b2_horse = horse[index:index + num_params_b2].reshape((10, 1))
            Z1, A1, Z2, A2 = forward_prop(W1_horse, b1_horse, W2_horse, b2_horse, X_train)
            cost = cost_function(A2, Y_train)
            if cost < best_cost:
                best_horse_index = j
                best_cost = cost
        update_horses(horses, alpha, best_horse_index)
        if (iterations - i) == 300:
            alpha = 0.01
        if (i + 1) % int(iterations / 10) == 0:
            predictions = np.argmax(A2, axis=0)
            accuracy = np.mean(predictions == Y_train)
            print(f"Iteration: {i + 1}, Cost: {cost:.4f}, Accuracy: {accuracy * 100:.2f}%")
    best_horse = horses[best_horse_index]
    index = 0
    W1_best = best_horse[index:index + num_params_W1].reshape((50, 784))
    index += num_params_W1
    b1_best = best_horse[index:index + num_params_b1].reshape((50, 1))
    index += num_params_b1
    W2_best = best_horse[index:index + num_params_W2].reshape((10, 50))
    index += num_params_W2
    b2_best = best_horse[index:index + num_params_b2].reshape((10, 1))
    return W1_best, b1_best, W2_best, b2_best

W1_final, b1_final, W2_final, b2_final = (
    train_neural_network_with_who(X_train, Y_train, alpha=0.03, iterations=1500))


# Make predictions and test the model
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return np.argmax(A2, axis=0)


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_dev[:, index, None]
    prediction = predict(current_image, W1, b1, W2, b2)
    label = Y_dev[index]
    print("Prediction:", prediction)
    print("Label:", label)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


# Test the model
for i in range(10):
    test_prediction(i, W1_final, b1_final, W2_final, b2_final)
