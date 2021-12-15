import numpy as np
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def loadData(test_size = 0.2, val_size = 0.25, batchsize = 16):
    # val_size : percentage of test_size

    path = os.getcwd() + "\code\data\\"

    X = np.load(path + "X.npy")
    Y = np.load(path + "Y.npy")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size = val_size)

    X_train, X_test, X_val = torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(X_val)
    Y_train, Y_test, Y_val = torch.Tensor(Y_train), torch.Tensor(Y_test), torch.Tensor(Y_val)

    train_set = TensorDataset(X_train, Y_train.long())
    test_set = TensorDataset(X_test, Y_test.long())
    val_set = TensorDataset(X_val, Y_val.long())

    trainLoader = DataLoader(train_set, batch_size=batchsize)
    testLoader = DataLoader(test_set, batch_size=batchsize)
    valLoader = DataLoader(val_set, batch_size=batchsize)

    print("Data importing complete!\n")

    return trainLoader, testLoader, valLoader