# import numpy as np

file_train = "./Dataset/data_batch_1"

def LoadBatch(filename):
    """ Copied from the dataset website """
    import pickle
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    return dict
    X = dict["b'data'"].T
    y = dict["b'labels'"]

    _, n = X.shape
    K = 10
    Y = np.zeros((K, n))



    return X, Y, y

## Step 1: Load data
dict = LoadBatch(file_train)

print("Everything's fine.")
debug = 0