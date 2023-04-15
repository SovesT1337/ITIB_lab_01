import numpy as np
import matplotlib.pyplot as plt


def simulated_boolean_function(x1, x2, x3, x4):
    return int(((not x1) or (not x2) or (not x3)) and ((not x2) or (not x3) or x4))


X1 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
X2 = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
X3 = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
X4 = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
Y = np.array([simulated_boolean_function(X1[i], X2[i], X3[i], X4[i]) for i in range(16)])
print('y = ', Y)

X_test = [[X1[i], X2[i], X3[i], X4[i]] for i in range(16)]
size_of_train = len(X1)
EPOCH_NUMBER = 50


def predict_for_binary_step(x_test):
    y_predicted_ = [net.forward(x_) for x_ in x_test]
    return np.array(y_predicted_)


def predict_for_softsign(x_test):
    y_predicted_ = [int(net.forward(x_) > 0.5) for x_ in x_test]
    return np.array(y_predicted_)


# 1 Task
print('Task 1 AF')


class BinaryStepNet:

    def __init__(self, seed=1):
        self.x = None
        np.random.seed(seed)
        self.W = np.zeros(shape=4)
        self.b = 0

    def forward(self, x_):
        self.x = x_
        net_ = np.dot(self.W, x_) + self.b
        y_ = int(net_ >= 0)
        return y_

    def backward(self, delta_):
        dw = np.dot(0.3 * delta_, self.x)
        db = 0.3 * delta_
        self.W = self.W + dw
        self.b = self.b + db


net = BinaryStepNet()
L_iter = []

for epoch in range(EPOCH_NUMBER):
    error = 0
    print("\n\tEpoch ", epoch)
    Y_predicted = []
    for i in range(size_of_train):
        x = [X1[i], X2[i], X3[i], X4[i]]
        y_true = Y[i]
        y_predicted = net.forward(x)
        delta = y_true - y_predicted
        error += abs(delta)
        net.backward(delta)
        Y_predicted.append(y_predicted)
    L_iter.append(error)
    print("W = ", net.W)
    print("Y = ", Y_predicted)
    print("E = ", error)
    if error == 0:
        break

fig, ax = plt.subplots()
ax.plot(L_iter)
ax.set_xlabel('epoch number')
ax.set_ylabel('error')
ax.set_title('Error(epoch)')
plt.show()

# 1.2 Task
print('--------------------------------------------------------------------------------')
print('Task 1.2')

net = BinaryStepNet()
L_iter = []
index = [6, 7, 10, 12, 15]
for epoch in range(EPOCH_NUMBER):
    error_on_train = 0
    print("\n\tEpoch ", epoch)
    for i in index:
        x = [X1[i], X2[i], X3[i], X4[i]]
        y_true = Y[i]
        y_predicted = net.forward(x)
        delta = y_true - y_predicted
        error_on_train += abs(delta)
        net.backward(delta)
    Y_predicted = predict_for_softsign(X_test)
    error_on_test = np.sum(np.abs(Y_predicted - Y))
    L_iter.append(error_on_test)
    print("W = ", net.W)
    print("Y = ", Y_predicted)
    print("E = ", error_on_test)
    if error_on_test == 0:
        break

print("\nfinal W = ", net.W)
fig, ax = plt.subplots()
ax.plot(L_iter)
ax.set_xlabel('epoch number')
ax.set_ylabel('error')
ax.set_title('Error(epoch)')
plt.show()

# 2 Task
print('---------------------------------------------------------------------')
print('Task 2')


class SoftsignNet:

    def __init__(self, seed=1):
        self.softsign_ = None
        self.x = None
        np.random.seed(seed)
        self.W = np.zeros(shape=4)
        self.b = 0

    def forward(self, x_):
        self.x = x_
        net_ = np.dot(self.W, x_) + self.b
        self.softsign_ = 0.5 * (net_ / (1 + abs(net_)) + 1)
        return int(self.softsign_ > 0.5)

    def backward(self, delta):
        dz = 0.5 / (1 + np.abs(self.softsign_) ** 2)
        dw = np.dot(0.3 * delta * dz, self.x)
        db = 0.3 * delta * dz
        self.W = self.W + dw
        self.b = self.b + db


net = SoftsignNet()
L_iter = []

for epoch in range(EPOCH_NUMBER):
    error = 0.
    print("\n\tEpoch ", epoch)
    Y_predicted = []
    for i in range(size_of_train):
        x = [X1[i], X2[i], X3[i], X4[i]]
        y_true = Y[i]
        y_predicted = net.forward(x)
        delta = y_true - y_predicted
        error += abs(delta)
        net.backward(delta)
        Y_predicted.append(y_predicted)
    L_iter.append(error)
    print("W = ", net.W)
    print("Y = ", Y_predicted)
    print("E = ", error)
    if error == 0:
        break

fig, ax = plt.subplots()
ax.plot(L_iter)
ax.set_xlabel('epoch number')
ax.set_ylabel('error')
ax.set_title('Error(epoch)')
plt.show()

# 2.2 Task
print('-------------------------------------------------------------------------------')
print('Task 2.2')

net = SoftsignNet()
L_iter = []
index = [6, 7, 10, 12, 15]
for epoch in range(EPOCH_NUMBER):
    error_on_train = 0.
    print("\n\tEpoch ", epoch)
    for i in index:
        x = [X1[i], X2[i], X3[i], X4[i]]
        y_true = Y[i]
        y_predicted = net.forward(x)
        delta = y_true - y_predicted
        error_on_train += abs(delta)
        net.backward(delta)
    Y_predicted = predict_for_softsign(X_test)
    error_on_test = np.sum(np.abs(Y_predicted - Y))
    L_iter.append(error_on_test)
    print("W = ", net.W)
    print("Y = ", Y_predicted)
    print("E = ", error_on_test)
    if error_on_test == 0:
        break

print("\nfinal W = ", net.W)
fig, ax = plt.subplots()
ax.plot(L_iter)
ax.set_xlabel('epoch number')
ax.set_ylabel('error')
ax.set_title('Error(epoch)')
plt.show()
