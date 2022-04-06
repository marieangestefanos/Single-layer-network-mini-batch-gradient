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

## Step3: Initialize weights
d, n = np.shape(X_train)
K, _ = np.shape(Y_train)

rng = np.random.default_rng()

W = rng.normal(0, 0.01, (K, d))
b = rng.normal(0, 0.01, (K, 1))

debug = 0