import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

data = pd.read_csv('train.csv')
data=data.sample(frac=1).reset_index(drop=True)
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

l=int(input("Give the total number of layers :"))
ne=int(input("Give the number of neurons for each layer :"))
def init_params(l):
    W = []
    b = []
    for i in range(l - 1):
        if i == 0:
            W.append(np.random.uniform(-0.5, 0.5, (ne, 784)))
            b.append(np.random.uniform(-0.5, 0.5, (ne, 1)))
        else:

            if i == l - 2:
                W.append(np.random.uniform(-0.5, 0.5, (10, ne)))
                b.append(np.random.uniform(-0.5, 0.5, (10, 1)))
            else:
                W.append(np.random.uniform(-0.5, 0.5, (ne, ne)))
                b.append(np.random.uniform(-0.5, 0.5, (ne, 1)))
    return W, b

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(Z):
    return np.maximum(Z, 0)

def leaky_relu(Z, alpha=0.01):
    return np.maximum(alpha * Z, Z)
def leaky_relu_derivative(Z, alpha=0.01):
    return np.where(Z > 0, 1, alpha)


def tanh(Z):
    return np.tanh(Z)

def tanh_deriv(Z):
    return 1 - np.tanh(Z) ** 2

def softmax(Z):
    Z -= np.max(Z, axis=0)
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A


def forward_prop(W, b, X):
    Z = [None] * (len(W) + 1)
    A = [None] * (len(W) + 1)
    Z[0] = np.dot(W[0], X) + b[0]
    A[0] = leaky_relu(Z[0])
    for i in range(1, len(W)-1):
        Z[i] = np.dot(W[i], A[i - 1]) + b[i]
        A[i] = leaky_relu(Z[i])
    Z[len(W)-1] = np.dot(W[len(W)-1], A[len(W) - 2]) + b[len(W)-1]
    A[len(W)-1] = softmax(Z[len(W)-1])
    return Z, A

def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z, A, W, b, X, Y):
    one_hot_Y = one_hot(Y)
    dW = [np.zeros_like(w) for w in W]
    db = [np.zeros_like(b) for b in b]
    dZ = [None] * (len(W) + 1)
    for i in range(len(W) - 1, -1, -1):
        if i == len(W) - 1:
            dZ[i] = 2*(A[i] - one_hot_Y)
            dW[i] = 1 / m * np.dot(dZ[i], A[i - 1].T)
            db[i] = 1 / m * np.sum(dZ[i], axis=1, keepdims=True)
        elif i == 0:
            dZ[i] = np.dot(W[i + 1].T, dZ[i + 1]) * leaky_relu_derivative(Z[i])
            dW[i] = 1 / m * np.dot(dZ[i], X.T)
            db[i] = 1 / m * np.sum(dZ[i], axis=1, keepdims=True)
        else:
            dZ[i] = np.dot(W[i + 1].T, dZ[i + 1]) * leaky_relu_derivative(Z[i])
            dW[i] = 1 / m * np.dot(dZ[i], A[i - 1].T)
            db[i] = 1 / m * np.sum(dZ[i], axis=1, keepdims=True)

    return dW, db



def update_params(W,b ,dW, db, alpha):
    for i in range(l-1):
        W[i] = W[i] - alpha * dW[i]
        b[i] = b[i] - alpha * db[i]
    return W, b

def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def updateAccumulators(vW,vb,dW,db,B=0.9):
    for i in range(l-1):
        vW[i] = (B * vW[i]) + ((1-B) * (dW[i] ** 2))
        vb[i] = (B * vb[i]) + ((1-B) * (db[i] ** 2))
    return vW , vb
epsilon=1e-8
def update_params2(W,b ,vW, vb,dW,db, alpha):
    for i in range(l-1):
        W[i] -= (alpha / (np.sqrt(vW[i]) + epsilon)) * dW[i]
        b[i] -= (alpha / (np.sqrt(vb[i]) + epsilon)) * db[i]
    return W, b

def gradient_descent(X, Y, alpha, iterations):
    W, b = init_params(l)
    vW = [np.zeros_like(w) for w in W]
    vb = [np.zeros_like(b) for b in b]
    for i in range(iterations):
        Z, A = forward_prop(W, b, X)
        dW, db = backward_prop(Z, A, W, b, X, Y)
        vW,vb=updateAccumulators(vW,vb, dW, db)
        W, b = update_params2(W, b, vW, vb,dW,db, alpha)
        if (i + 1) % int(iterations / 10) == 0:
            print(f"Iteration: {i + 1} / {iterations}")
            prediction = get_predictions(A[l - 2])
            print(f'{get_accuracy(prediction, Y):.3%}')
    return W, b



W, b = gradient_descent(X_train, Y_train, 0.15, 500)


def make_predictions(X, W, b):
    _, A = forward_prop(W, b, X)
    Af=A[l-2]
    predictions = get_predictions(Af)
    return predictions

params = {'W': W, 'b': b}

with open('parameter.pkl', 'wb') as f:
    pickle.dump(params, f)


def test_prediction(index, W, b):
    current_image = X_dev[:, index, None]
    prediction = make_predictions(X_dev[:, index, None], W, b)
    label = Y_dev[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


for i in range(10):
    test_prediction(i, W, b)

