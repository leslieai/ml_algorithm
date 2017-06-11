import numpy as np
import math


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1.0 - np.tanh(x) * np.tanh(x)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


def hardlim(n):
    if n >= 0:
        return 1
    else:
        return 0


class NN(object):
    def __init__(self, W):
        self.W_old = W
        self.learning_rate = 0.2

    def fit(self, X, y):
        for k in range(1, 10000):
            for idx, val in enumerate(X):
                # if idx <4:
                # print(y[idx],val)
                X_bar = np.hstack((val, 1))
                g_x = np.dot(X_bar, self.W_old)
                y_hat = logistic(g_x)
                # f is logistic function, f' is f*(1-f)
                # delta_W = -e*(f'(gx))*X_bar
                delta_W = -(y_hat - y[idx]) * y_hat * (1 - y_hat) * X_bar
                # print(y_hat)
                W_new = self.W_old + delta_W
                self.W_old = W_new
        print(self.W_old)

    def predict(self, X):
        for k in X:
            X_bar = np.hstack((k, 1))
            y = hardlim(np.dot(X_bar, self.W_old))
            print(y)


nn = NN([0, 0, 0])
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])
# nn.fit(X, y)
# print(10**(-5))
# print(math.pow(10,-5))
# nn.predict(X)
print(y.dot([1, 1, 1, 2]))
# for i in X:
# 	print(np.hstack((i,1)))
# print(len([2,3,4]))
