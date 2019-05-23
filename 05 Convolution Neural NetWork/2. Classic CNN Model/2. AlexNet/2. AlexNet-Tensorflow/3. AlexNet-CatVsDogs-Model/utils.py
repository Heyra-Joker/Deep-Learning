import numpy as np


def random_mini_batche_v2(X, Y, mini_batche_size, seed):
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
    m = X.shape[0]
    mini_batches = []

    # Step 1: Shuffle X and Y

    # Randomly permute a sequence, or return a permuted range.
    permutation = list(np.random.permutation(m))
    shuffle_X = X[permutation,...] 
    shuffle_Y = Y[permutation,:]

    # Step 2:
    # ①：
    #     num_complete_minibatches = math.floor(m / mini_batche_size) #  get the integer part in mini batche
    #   , if cannot be divisible by m,then we need use Handling the end case
    # ②:
    num_complete_minibatches = m // mini_batche_size

    # start Partition
    for k in range(num_complete_minibatches):
    
        mini_batche_X = shuffle_X[k * mini_batche_size: (k + 1) * mini_batche_size,...]
        mini_batche_Y = shuffle_Y[k * mini_batche_size: (k + 1) * mini_batche_size,:]
        mini_batche = (mini_batche_X, mini_batche_Y)
        mini_batches.append(mini_batche)

    if m % mini_batche_size != 0:

        mini_batche_X = shuffle_X[num_complete_minibatches * mini_batche_size:,...]
        mini_batche_Y = shuffle_Y[num_complete_minibatches * mini_batche_size:,:]
        mini_batche = (mini_batche_X, mini_batche_Y)
        mini_batches.append(mini_batche)

    return mini_batches