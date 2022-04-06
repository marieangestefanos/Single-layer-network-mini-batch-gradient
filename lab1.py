import numpy as np

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
    lcross = np.array([Y[:, i].T @ np.log(P[:, i]) for i in range(n)])
    J = np.sum(lcross)/n + lbda * W * W
    return J


def ComputeAccuracy(X, y, W, b):
    n = len(y)
    P = EvaluateClassifier(X, W, b)
    prediction = np.argmax(P, axis=0)
    acc = np.sum([np.where(prediction[i] == y[i], 1, 0) for i in range(n)])/n
    return acc

## Step 1: Read and store train, valid and test datasets
X_train, Y_train, y_train = LoadBatch(file_train)
X_valid, Y_valid, y_valid = LoadBatch(file_valid)
X_test, Y_test, y_test = LoadBatch(file_test)

## Step 2: Preprocessing train, valid and test sets
mean = np.mean(X_train, axis = 1).reshape(-1, 1)
std = np.std(X_train, axis = 1).reshape(-1, 1)

X_train = prePreprocessing(X_train, mean, std)
X_valid = prePreprocessing(X_valid, mean, std)
X_test = prePreprocessing(X_test, mean, std)

## Step 3: Initialize weights
d, n = np.shape(X_train)
K, _ = np.shape(Y_train)

rng = np.random.default_rng()

W = rng.normal(0, 0.01, (K, d))
b = rng.normal(0, 0.01, (K, 1))

## Step 4: Evaluate the network function
P = EvaluateClassifier(X_train, W, b)

## Step 5: Compute the cost function
lbda = 0
J = ComputeCost(X_train, Y_train, W, b, lbda)

## Step 6: Compute accuracy of network's prediction
acc = ComputeAccuracy(X_train, y_train, W, b)

debug = 0