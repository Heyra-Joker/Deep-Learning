import numpy as np

def relu(Z):
    return np.maximum(0,Z)

def softmax(Z):
    t = np.exp(Z)
    return t/ np.sum(t,axis=0)

def Loss(A,y):
    m = y.shape[1]
    loss = -np.sum(np.multiply(y,np.log(A))) / m
    return loss

def random_mini_batche(X, Y, mini_batche_size=64, seed=0):
    '''
    Using mini batches Shuffle and partition X,Y

    Argument:
    ----------
    X: training data set
    Y: training data labels
    mini_batche_size: size of the mini batche, integer

    Return:
    mini_batches :  it's a list,(mini_batche_X,mini_batche_Y)
    '''
    np.random.seed(seed)
    m, n_h,n_w,n_c = X.shape
    X_reshape = X.reshape((m,-1))
    mini_batches = []

    # Step 1: Shuffle X and Y

    # Randomly permute a sequence, or return a permuted range.
    permutation = list(np.random.permutation(m))
    shuffle_X = X_reshape[permutation]  # change index with axis == 1 in X
    shuffle_Y = Y[:,permutation]

    # Step 2:
    # ①：
    #     num_complete_minibatches = math.floor(m / mini_batche_size) #  get the integer part in mini batche
    #   , if cannot be divisible by m,then we need use Handling the end case
    # ②:
    num_complete_minibatches = m // mini_batche_size

    # start Partition
    for k in range(num_complete_minibatches):
    
        mini_batche_X = shuffle_X[k * mini_batche_size: (k + 1) * mini_batche_size, :]
        mini_batche_Y = shuffle_Y[:,k * mini_batche_size: (k + 1) * mini_batche_size]
        mini_batche_X_reshape = mini_batche_X.reshape((-1, n_h,n_w,n_c))
        mini_batche = (mini_batche_X_reshape, mini_batche_Y)
        mini_batches.append(mini_batche)

    if m % mini_batche_size != 0:

        mini_batche_X = shuffle_X[num_complete_minibatches * mini_batche_size:, :]
        mini_batche_Y = shuffle_Y[:,num_complete_minibatches * mini_batche_size:]
        mini_batche_X_reshape = mini_batche_X.reshape((-1, n_h,n_w,n_c))
        mini_batche = (mini_batche_X_reshape, mini_batche_Y)
        mini_batches.append(mini_batche)

    return mini_batches


def Update(L,dparameters,parameters,V,M,beta_1,beta_2,t,lr,epsilon=1e-8):
    """
    Updating parameters and using Adam optimizeer.
    """
    for l in range(L-1):
        M['M_dW'+str(l+1)] = beta_1 * M['M_dW'+str(l+1)] + (1-beta_1) * dparameters['dW'+str(l+1)]
        M['M_db'+str(l+1)] = beta_1 * M['M_db'+str(l+1)] + (1-beta_1) * dparameters['db'+str(l+1)]
        
        V['V_dW'+str(l+1)] = beta_2 * V['V_dW'+str(l+1)] + (1-beta_2) * np.square(dparameters['dW'+str(l+1)])
        V['V_db'+str(l+1)] = beta_2 * V['V_db'+str(l+1)] + (1-beta_2) * np.square(dparameters['db'+str(l+1)])
        
        M_correct_dW = M['M_dW'+str(l+1)]/ (1-np.power(beta_1,t))
        M_correct_db = M['M_db'+str(l+1)]/ (1-np.power(beta_1,t))
        
        V_correect_dW = V['V_dW'+str(l+1)]/ (1-np.power(beta_2,t))
        V_correect_db = V['V_db'+str(l+1)]/ (1-np.power(beta_2,t))
        
        parameters['W'+str(l+1)] -= lr * M_correct_dW /(np.sqrt(V_correect_dW+epsilon)) 
        parameters['b'+str(l+1)] -= lr * M_correct_db /(np.sqrt(V_correect_db+epsilon)) 
    
    return parameters,V,M