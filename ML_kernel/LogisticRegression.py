from numpy import *
from ML_kernel.sigmoid import sigmoid

def gradient(trainset, label, a, cycle):
    traindata = mat(trainset)
    y = mat(label).T
    m, n = shape(traindata)
    weight = ones((n, 1))
    b = 1
    for i in range(cycle):
        h = sigmoid(traindata*weight + b)
        error = h-y
        weight = weight-a*(1/m)*traindata.T*error
        b = b-a*(1/m)*sum(error)

    return weight,b
