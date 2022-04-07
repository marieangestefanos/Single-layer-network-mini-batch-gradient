import enum
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator

file_train = "./Dataset/data_batch_1"
file_valid = "./Dataset/data_batch_2"
file_test = "./Dataset/test_batch"

def LoadBatch(filename):
    """ Copied from the dataset website """
    import pickle
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    X = dict[b'data'].T #size d*n = 3072 * 10000
    y = dict[b'labels'] #size K*n = 10 * 10000

    _, n = X.shape
    K = 10

    Y = np.zeros((K, n))
    for j in range(n):
        Y[y[j], j] = 1

    return X, Y, y


def prePreprocessing(X, mean, std):
    return (X - mean)/std


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def EvaluateClassifier(X, W, b):
    return softmax(W @ X + b)


def ComputeCost(X, Y, W, b, lbda):
    _, n = Y.shape
    P = EvaluateClassifier(X, W, b)
    lcross = - np.array([Y[:, i].T @ np.log(P[:, i]) for i in range(n)])
    # lcross = np.zeros
    # for i in range(n):
    #     lcross 
    J = np.sum(lcross)/n + lbda * np.sum(W * W)
    return J, np.sum(lcross)/n


def ComputeAccuracy(X, y, W, b):
    n = len(y)
    P = EvaluateClassifier(X, W, b)
    prediction = np.argmax(P, axis=0)
    acc = np.sum([np.where(prediction[i] == y[i], 1, 0) for i in range(n)])/n
    return acc


def ComputeGradients(X, Y, P, W, lbda):
    G_batch = - (Y - P)
    _, n_batch = X.shape
    grad_W = (G_batch @ X.T)/n_batch + 2 * lbda * W
    grad_b = (G_batch @ np.ones((n_batch, 1)))/n_batch
    return grad_W, grad_b


def ComputeGradsNum(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no 	= 	W.shape[0]
    d 	= 	X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))

    c, _ = ComputeCost(X, Y, W, b, lamda)
    
    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2, _ = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2-c) / h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i,j] += h
            c2, _ = ComputeCost(X, Y, W_try, b, lamda)
            grad_W[i,j] = (c2-c) / h

    return [grad_W, grad_b]


def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no 	= 	W.shape[0]
    d 	= 	X.shape[0]

    grad_W = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))
    
    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1, _ = ComputeCost(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2, _ = ComputeCost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2-c1) / (2*h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i,j] -= h
            c1, _ = ComputeCost(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i,j] += h
            c2, _ = ComputeCost(X, Y, W_try, b, lamda)

            grad_W[i,j] = (c2-c1) / (2*h)

    return [grad_W, grad_b]


def ComputeRelativeError(grad_W_an, grad_b_an, grad_W_num, grad_b_num, eps):
    grad_W_err = np.abs(grad_W_an - grad_W_num) / np.maximum(eps, np.abs(grad_W_num) + np.abs(grad_W_an))
    grad_b_err = np.abs(grad_b_an - grad_b_num) / np.maximum(eps, np.abs(grad_b_num) + np.abs(grad_b_an))
    return grad_W_err, grad_b_err


def MiniBatchGD(X_train, Y_train, X_valid, Y_valid, GDparams, W, b, lbda):
    _, n = X_train.shape
    n_batch = GDparams["n_batch"]
    eta = GDparams["eta"]
    J_list_train = []
    loss_list_train = []
    J_list_valid = []
    loss_list_valid = []

    for i in range(GDparams["n_epochs"]):
        idx_permutation = rng.permutation(n)
        j_start_array = np.arange(0, n-n_batch+1, n_batch)
        j_end_array = np.arange(n_batch-1, n, n_batch)
        for i in range(len(j_start_array)):
            j_start = j_start_array[i]
            j_end = j_end_array[i]
            X_batch = X_train[:, idx_permutation[j_start:j_end+1]]
            Y_batch = Y_train[:, idx_permutation[j_start:j_end+1]]

            P_batch = EvaluateClassifier(X_batch, W, b)

            grad_W, grad_b = ComputeGradients(X_batch, Y_batch, P_batch, W, lbda)
            
            #update W
            W -= eta * grad_W
            #update b
            b -= eta * grad_b

        J_train, loss_train = ComputeCost(X_train, Y_train, W, b, lbda)
        J_valid, loss_valid = ComputeCost(X_valid, Y_valid, W, b, lbda)

        J_list_train.append(J_train)
        loss_list_train.append(loss_train)
        J_list_valid.append(J_valid)
        loss_list_valid.append(loss_valid)

    Wstar = W
    bstar = b
    return Wstar, bstar, J_list_train, loss_list_train, J_list_valid, loss_list_valid


def plot(x_axis, y_axis, x_ticks, legends, title, x_label, y_label):
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xticks(x_ticks)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i in range(len(x_axis)):
        plt.plot(x_axis[i], y_axis[i], label=legends[i])
    
    plt.legend()
    plt.show()


  ## SCRIPT

## Parameters
lbda = 0
end_batch = 5
d_batch = 20
h = 1e-6
eps = 1e-6
eta = 0.001
n_epochs = 40
n_batch = 100
GDparams = {"n_batch": n_batch, "eta": eta, "n_epochs": n_epochs}

## Step 1: Read and store train, valid and test datasets
X_train, Y_train, y_train = LoadBatch(file_train)
X_valid, Y_valid, y_valid = LoadBatch(file_valid)
X_test, Y_test, y_test = LoadBatch(file_test)

d, n = np.shape(X_train)
K, _ = np.shape(Y_train)

## Step 2: Preprocessing train, valid and test sets
mean = np.mean(X_train, axis = 1).reshape(-1, 1)
std = np.std(X_train, axis = 1).reshape(-1, 1)

X_train = prePreprocessing(X_train, mean, std)
X_valid = prePreprocessing(X_valid, mean, std)
X_test = prePreprocessing(X_test, mean, std)

## Step 3: Initialize weights

rng = np.random.default_rng(400)

W = rng.normal(0, 0.01, (K, d))
b = rng.normal(0, 0.01, (K, 1))

## Step 4: Evaluate the network function
P = EvaluateClassifier(X_train, W, b)

## Step 5: Compute the cost function
J = ComputeCost(X_train, Y_train, W, b, lbda)

## Step 6: Compute accuracy of network's prediction
acc = ComputeAccuracy(X_train, y_train, W, b)

## Step 7: Compute gradients for a mini-batch
X_batch = X_train[:d_batch, :end_batch]
Y_batch = Y_train[:, :end_batch]
W_batch = W[:, :d_batch]
b_batch = b[:d_batch]

P_batch = EvaluateClassifier(X_batch, W_batch, b_batch)

grad_W_an, grad_b_an = ComputeGradients(X_batch, Y_batch, P_batch, W_batch, lbda)

#finite difference
grad_W_num_fast, grad_b_num_fast = ComputeGradsNum(X_batch, Y_batch,
                                P_batch, W_batch, b_batch, lbda, h)

#centered difference
grad_W_num_slow, grad_b_num_slow = ComputeGradsNumSlow(X_batch,
                    Y_batch, P_batch, W_batch, b_batch, lbda, h)

grad_W_err_fast, grad_b_err_fast = ComputeRelativeError(grad_W_an, grad_b_an, grad_W_num_fast, grad_b_num_fast, eps)
grad_W_err_slow, grad_b_err_slow = ComputeRelativeError(grad_W_an, grad_b_an, grad_W_num_slow, grad_b_num_slow, eps)
grad_W_err_given, grad_b_err_given = ComputeRelativeError(grad_W_num_slow, grad_b_num_slow, grad_W_num_fast, grad_b_num_fast, eps)

print("\nChecking error on gradients: \n")
print("max(grad_W_err_fast) = {0:.9f}".format(np.amax(grad_W_err_fast)))
print("max(grad_b_err_fast) = {0:.9f}".format(np.amax(grad_b_err_fast)))
print("max(grad_W_err_slow) = {0:.9f}".format(np.amax(grad_W_err_slow)))
print("max(grad_b_err_slow) = {0:.9f}".format(np.amax(grad_b_err_slow)))
print("max(grad_W_err_given) = {0:.9f}".format(np.amax(grad_W_err_given)))
print("max(grad_b_err_given) = {0:.9f}".format(np.amax(grad_b_err_given)))

try:
    msg = np.testing.assert_array_almost_equal(grad_W_num_slow, grad_W_an, -np.log10(eps))
except:
    print("grad_W is incorrect.")

try:
    msg = np.testing.assert_array_almost_equal(grad_b_num_slow, grad_b_an, -np.log10(eps))
except:
    print("grad_b is incorrect.")


## Step 8: mini-batch gradient descent algorithm

"""
for j in range(n/n_batch):
    j_start = j * n_batch #MATLAB: goes from 1 to n-n_batch+1
    j_end = j * (n_batch) - 1 #MATLAB: goes from n_batch to n
    X_batch = X_train[:, j_start:j_end]
    Y_batch = Y_train[:, j_start:j_end]
"""

Wstar, bstar, J_list_train, loss_list_train, J_list_valid, loss_list_valid = \
    MiniBatchGD(X_train, Y_train, X_valid, Y_valid, GDparams, W, b, lbda)

acc_train = ComputeAccuracy(X_train, y_train, W, b)
acc_valid = ComputeAccuracy(X_valid, y_valid, W, b)

print(f"\nAccuracy of the training set: {acc_train*100}%")
print(f"Accuracy of the validation set: {acc_valid*100}%\n")

    # Plot cost after each epoch
x_axis = np.array([np.arange(0, n_epochs), np.arange(0, n_epochs)])
y_axis = np.array([J_list_train, J_list_valid])
x_ticks = np.arange(0, n_epochs, 5)
legends = ["training", "validation"]
x_label = "epoch"
y_label = "cost function J"
title = "Cost after every epoch"

plot(x_axis, y_axis, x_ticks, legends, title, x_label, y_label)

    # Plot loss after each epoch
x_axis = np.array([np.arange(0, n_epochs), np.arange(0, n_epochs)])
y_axis = np.array([loss_list_train, loss_list_valid])
x_ticks = np.arange(0, n_epochs, 5)
legends = ["training", "validation"]
x_label = "epoch"
y_label = "loss"
title = "Loss after every epoch"

plot(x_axis, y_axis, x_ticks, legends, title, x_label, y_label)


debug = 0