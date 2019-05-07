import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import os




def ONE_HOT(y,n_classes):
    return np.eye(n_classes)[y]



def initial_parameters(L, layers):
    """
    initialization parameters.
    Arguments:
    ---------
        L: #input'units + hidden layers'units
        layers: hidden layers's units
    Return:
    ------
        parameters: include weights and bias.
    """
    np.random.seed(1)
    parameters = {}
    V = {}
    M = {}
    for l in range(L - 1):
        W = np.random.randn(layers[l], layers[l + 1]) / np.sqrt(layers[l])
        b = np.zeros((1, layers[l + 1]))

        V['V_dW' + str(l + 1)] = np.zeros(W.shape)
        V['V_db' + str(l + 1)] = np.zeros(b.shape)
        M['M_dW' + str(l + 1)] = np.zeros(W.shape)
        M['M_db' + str(l + 1)] = np.zeros(b.shape)
        parameters['W' + str(l + 1)] = W
        parameters['b' + str(l + 1)] = b

    return parameters, V, M

def relu(Z):
    """
    ReLu activation
    """
    return np.maximum(0,Z)

def softmax(Z):
    """
    softmax activation
    """
    t = np.exp(Z)
    return t/np.sum(t,axis=1,keepdims=True)

def forward(X,L,parameters):
    """
    forward propagation
    """
    A = X
    cache = {'A0':X}
    for l in range(L-1):
        W = parameters['W'+str(l+1)]
        b = parameters['b'+str(l+1)]
        Z = np.add(np.dot(A,W),b)
        cache['Z'+str(l+1)] = Z
        if l != L -2 :
            A = relu(Z)
        else:
            A = softmax(Z)
        cache['A'+str(l+1)] = A
    return A,cache

def Loss(A,y):
    """
    caculate loss value in mini-batchs or score data.
    """
    m = y.shape[0]
    loss = - np.sum(np.multiply(y,np.log(A))) / m
    return loss


def backward(A, y, cache, parameters, L):
    """
    Backward propagation
    """
    m = y.shape[0]
    dparameters = {}
    for l in range(L - 1, 0, -1):
        if l == L - 1:
            dZ = A - y
            dW = np.dot(cache['A' + str(l - 1)].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
        else:
            dA = np.dot(dZ, parameters['W' + str(l + 1)].T)
            dZ = np.multiply(dA, np.int64(cache['Z' + str(l)] > 0))
            A = cache['A' + str(l - 1)]
            dW = np.dot(A.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
        dparameters['dW' + str(l)] = dW
        dparameters['db' + str(l)] = db

    return dparameters


def Update(L, dparameters, parameters, V, M, beta_1, beta_2, t, lr, epsilon=1e-8):
    """
    Updating parameters and using Adam optimizeer.
    """
    for l in range(L - 1):
        M['M_dW' + str(l + 1)] = beta_1 * M['M_dW' + str(l + 1)] + (1 - beta_1) * dparameters['dW' + str(l + 1)]
        M['M_db' + str(l + 1)] = beta_1 * M['M_db' + str(l + 1)] + (1 - beta_1) * dparameters['db' + str(l + 1)]

        V['V_dW' + str(l + 1)] = beta_2 * V['V_dW' + str(l + 1)] + (1 - beta_2) * np.square(
            dparameters['dW' + str(l + 1)])
        V['V_db' + str(l + 1)] = beta_2 * V['V_db' + str(l + 1)] + (1 - beta_2) * np.square(
            dparameters['db' + str(l + 1)])

        M_correct_dW = M['M_dW' + str(l + 1)] / (1 - np.power(beta_1, t))
        M_correct_db = M['M_db' + str(l + 1)] / (1 - np.power(beta_1, t))

        V_correect_dW = V['V_dW' + str(l + 1)] / (1 - np.power(beta_2, t))
        V_correect_db = V['V_db' + str(l + 1)] / (1 - np.power(beta_2, t))

        parameters['W' + str(l + 1)] -= lr * M_correct_dW / (np.sqrt(V_correect_dW + epsilon))
        parameters['b' + str(l + 1)] -= lr * M_correct_db / (np.sqrt(V_correect_db + epsilon))

    return parameters, V, M


def random_mini_batchs(X, y, seed, batc_size=64):
    """
    Create mini-batchs.
    """
    np.random.seed(seed)  # make sure every epochs the data is shuffle.

    m = X.shape[0]
    mini_batchs = []

    index_ = np.random.permutation(m)

    shuffle_X = X[index_, :]
    shuffle_y = y[index_, :]

    num_compute_minibatch_size = m // batc_size
    for i in range(num_compute_minibatch_size):
        mini_x = shuffle_X[i * batc_size:(i + 1) * batc_size, :]

        mini_y = shuffle_y[i * batc_size:(i + 1) * batc_size, :]
        mini_batch = (mini_x, mini_y)
        mini_batchs.append(mini_batch)

    if m % batc_size != 0:
        mini_x = shuffle_X[num_compute_minibatch_size * batc_size:, :]
        mini_y = shuffle_y[num_compute_minibatch_size * batc_size:, :]
        mini_batch = (mini_x, mini_y)
        mini_batchs.append(mini_batch)

    return mini_batchs

def score(data,labels,L,parameters,is_loss=False):
    """
    score model and return correct rate or loss value.
    """
    m = labels.shape[0]
    A,_ = forward(data,L,parameters)
    predict_y = np.argmax(A,axis=1)
    true_y = np.argmax(labels,axis=1)
    acc = np.equal(true_y,predict_y).sum() / m
    if is_loss:
        loss = Loss(A,labels)
        return acc,loss
    else:
        return acc


def Softmax_Model(layers, data, labels, val_data, val_labels, lr, epochs, beta_1=0.9, beta_2=0.999, batc_size=64,
                  save_path=None,lock=None):
    """
    Implement softmax model.
    NN model:Linear(25)--->Linear(12)--->Linear(10)
    """

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    SON_PATH = save_path + str(np.round(lr, 4)) + '/'
    if not os.path.exists(SON_PATH):
        os.mkdir(SON_PATH)


    layers.insert(0, data.shape[1])
    L = len(layers)
    n_classes = len(np.unique(labels))
    y_train_hot = ONE_HOT(labels, n_classes)
    y_val_hot = ONE_HOT(val_labels, n_classes)
    seed = 0
    t = 0
    losses = []
    val_losses = []
    acc_trains = []
    acc_vals = []

    parameters, V, M = initial_parameters(L, layers)


    for epoch in range(epochs):
        seed += 1
        mini_batchs = random_mini_batchs(data, y_train_hot, seed=seed, batc_size=batc_size)
        for mini_x, mini_y in mini_batchs:
            t += 1
            A, cache = forward(mini_x, L, parameters)
            dparameters = backward(A, mini_y, cache, parameters, L)
            parameters, V, M = Update(L, dparameters, parameters, V, M, beta_1, beta_2, t, lr)

        acc_train, train_loss = score(data, y_train_hot, L, parameters, True)
        acc_val, val_loss = score(val_data, y_val_hot, L, parameters, True)

        losses.append(train_loss)
        val_losses.append(val_loss)
        acc_trains.append(acc_train)
        acc_vals.append(acc_val)

        print('[{}/{}] loss:{:.4f},acc_train:{:.4f},val_loss:{:.4f},acc_val:{:.4f}'.format(epoch + 1,
                                                                                            epochs,
                                                                                            train_loss,
                                                                                            acc_train,
                                                                                            val_loss,
                                                                                            acc_val))

        FULL_PATH_TXT = SON_PATH + 'log.txt'
        with lock:
            with open(FULL_PATH_TXT,mode='a') as f:
                WRITE = '[{}/{}] loss:{:.4f},acc_train:{:.4f},val_loss:{:.4f},acc_val:{:.4f}\n'.format(epoch + 1,
                                                                                                epochs,
                                                                                                train_loss,
                                                                                                acc_train,
                                                                                                val_loss,
                                                                                                acc_val)
                f.write(WRITE)
                f.flush()




    return losses, val_losses, acc_trains, acc_vals, parameters,SON_PATH




