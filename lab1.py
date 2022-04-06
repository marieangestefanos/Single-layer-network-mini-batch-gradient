import numpy as np

file_train = "./Dataset/data_batch_1"
file_valid = "./Dataset/data_batch_2"
file_test = "./Dataset/test_batch"

def LoadBatch(filename):
    """ Copied from the dataset website """
    import pickle
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    X = dict[b'data'].T
    y = dict[b'labels']

    _, n = X.shape
    K = 10

    Y = np.zeros((K, n))
    for j in range(n):
        Y[y[j], j] = 1

    return X, Y, y

## Step 1: Read and store train, valid and test datasets
X_train, Y_train, y_train = LoadBatch(file_train)
X_valid, Y_valid, y_valid = LoadBatch(file_valid)
X_test, Y_test, y_test = LoadBatch(file_test)

debug = 0