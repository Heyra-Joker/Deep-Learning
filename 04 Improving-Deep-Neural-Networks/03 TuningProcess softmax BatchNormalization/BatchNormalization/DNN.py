import numpy as np
from data_utils import get_CIFAR10_data
import matplotlib.pyplot as plt
from  matplotlib import  gridspec
from miniBatch import random_mini_batche
def init_parameters(n, layers,is_small):
    layers.insert(0, n)
    L = len(layers)
    parameters = {}
    for l in range(1, L):
        if is_small:
            parameters['W' + str(l)] = np.random.randn(layers[l - 1], layers[l]) / np.sqrt(layers[l - 1])
        else:
            parameters['W' + str(l)] = np.random.randn(layers[l - 1], layers[l]) 
        parameters['b' + str(l)] = np.zeros((1, layers[l]))

    return parameters, L


def relu(Z):
    return np.maximum(0, Z)


def softmax(Z):
    t = np.exp(Z)
    res = t / np.sum(t, axis=1, keepdims=True)  # softmax是每一列的和为1
    return res


def forward(X, parameters, L):
    """
    Build forward propagation
    Parameters:
    ----------
        X: training data.
        parameters: weights and bias.
        L: lengths of layers.

    Returns:
    -------
        A: output layer value. and can using it to compute loss value
        cache: cache Z,A
    """
    A = X
    cache = {'A0': X}
    for l in range(1, L):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(A, W) + b
        cache['Z' + str(l)] = Z
        if l == L - 1:
            A = softmax(Z)
        else:
            A = relu(Z)
        cache['A' + str(l)] = A
    return A, cache


def cacl_loss(A, y):
    m = y.shape[0]
    assert A.shape == y.shape
    loss = - np.sum(y * np.log(A)) / m
    return loss


def backward(y, m, parameters, cache, L):
    """
    Build backward propagation

    Parameters:
    ----------
        A: output layer value.
        y: true labels.
        m: m example.
        parameters: weights and bias
        cache:cache Z,A
        L: lengths of layers.

    Return:
    -------
        d_parameters: parameters derivative value
    """
    d_parameters = {}

    for l in range(L - 1, 0, -1):
        if l == L - 1:
            d_parameters['dZ' + str(L - 1)] = cache['A' + str(L - 1)] - y
        else:
            d_parameters['dZ' + str(l)] = np.multiply(d_parameters['dA' + str(l)], np.int64(cache['Z' + str(l)] > 0))

        d_parameters['dW' + str(l)] = np.dot(cache['A' + str(l - 1)].T,d_parameters['dZ' + str(l)]) / m
        d_parameters['db' + str(l)] = np.sum(d_parameters['dZ' + str(l)], axis=0, keepdims=True) / m
        if l != 1:
            d_parameters['dA' + str(l - 1)] = np.dot(d_parameters['dZ' + str(l)],parameters['W' + str(l)].T)

    return d_parameters,L


def Update(d_parameters, parameters, alpha, L):
    """
    Update parameters.

    Parameters:
    ----------
        d_parameters:parameters derivative value
        parameters:weights and bias
        alpha:learnLearning  rate
        L: lengths of layers

    Return:
    ------
        parameters: Updated parameters.
    """
    for l in range(1, L):
        parameters['W' + str(l)] -= alpha * d_parameters['dW' + str(l)]
        parameters['b' + str(l)] -= alpha * d_parameters['db' + str(l)]

    return parameters





def score_base(X, y, parameters, L):
    m = y.shape[0]
    A, cache = forward(X, parameters, L)
    
    accuracy = (np.argmax(A,axis=1)==y).sum() / m
    return accuracy


def MODEL(X, y, X_val, y_val, epocs, layers, alpha, batch_size=64,is_small=True):
    m, n = X.shape
    classes = len(np.unique(y))
    parameters, L = init_parameters(n, layers,is_small)
    costs = []
    cache_acc_train = []
    cache_acc_val = []
    seed = 0
    num_minibatches = X.shape[0] / batch_size

    for epoch in range(epocs):
        epoch_cost = 0
        seed = seed + 1
        mini_batches = random_mini_batche(X, y, batch_size, seed)
        for minibatch in mini_batches:
            (mini_batche_X, mini_batche_Y) = minibatch
            mini_batche_Y_hot = np.eye(classes)[mini_batche_Y]
            A, cache = forward(mini_batche_X, parameters, L)

            loss = cacl_loss(A, mini_batche_Y_hot)
            epoch_cost += loss / num_minibatches

            dparameters, L = backward(mini_batche_Y_hot, m, parameters, cache, L)
            parameters = Update(dparameters, parameters, alpha, L)

        
        if epoch % 10 == 0:
            costs.append(epoch_cost)
            # calculate train accuracy
            accuracy_train = score_base(X, y, parameters, L)
            cache_acc_train.append(accuracy_train)
            # calculate val data
            accuracy_val = score_base(X_val, y_val, parameters, L)
            cache_acc_val.append(accuracy_val)
        if epoch % 100 == 0:
            print('After epoch:{} loss:{} train acc:{} val acc:{}'.format(epoch,
                                                                          epoch_cost,accuracy_train,accuracy_val))


    return parameters,costs, cache_acc_train, cache_acc_val,L





