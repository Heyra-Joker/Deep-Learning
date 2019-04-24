import numpy as np
import matplotlib.pyplot as plt
def random_miniBatch(batch_size,data,labels,seed):
    """
    Random Split data -- MiniBatchs
    Arguments:
    ---------
        batch_size: batch size to split data set,can choose 2^n.
        data: split data.
        labels: split data labels.
        seed: random seed,make sure the result same in every running.
    Return:
    ------
       mini_batchs: mini batchs, it's a list, include mini_batch_X,mini_batch_y.
    """
    
    np.random.seed(seed)
    m,n = data.shape
    mini_batchs = []
    
    premutation_index = np.random.permutation(m) # permutation m,return shuffle 0->m.
    # shuffle data and labels.
    shuffle_X = data[premutation_index,:] 
    shuffle_y = labels[premutation_index,:]
    
    # get divisible part.
    num_complet_miniBatch = m // batch_size
    
    # Spliting data and labels.
    for k in range(num_complet_miniBatch):
        mini_batch_X = shuffle_X[k * batch_size:(k+1)*batch_size,:]
        mini_batch_y = shuffle_y[k * batch_size:(k+1)*batch_size,:]
        mini_batch = (mini_batch_X,mini_batch_y)
        # append the mini_batchs list
        mini_batchs.append(mini_batch)
        
    # if have can not divisible part, then append mini_batch list last part.
    if m % num_complet_miniBatch !=0:
        mini_batch_X = shuffle_X[num_complet_miniBatch * batch_size:,:]
        mini_batch_y = shuffle_y[num_complet_miniBatch * batch_size:,:]
        mini_batch = (mini_batch_X,mini_batch_y)
        mini_batchs.append(mini_batch)
        
    return mini_batchs 


def sigmoid(Z):
        """
        Sigmoid activation
        """
        return 1./(1.+np.exp(-Z))
    
def relu(Z):
    """
    RELU activation 
    """
    return np.maximum(0,Z)

def cost(A,y):
    """
    loss function: binary cross entropy.
    """
    m = y.shape[0]
    loss = - np.sum(np.multiply(y,np.log(A))+np.multiply((1-y),np.log(1-A))) /m
    return loss

def forward(X,L,parameters):
        """
        Implement Forward Propagation 
        Arguments:
        --------
            X: training data set.
            parameters: include weights and bias.
            
        Returns:
        -------
            A: last layers value,output layer values.
            cache: cache A and Z to using backward propagation.
        """
        A = X
        cache = {'A0':X}
        for l in range(L-1):
            W,b = parameters['W'+str(l+1)],parameters['b'+str(l+1)]
            Z = np.add(np.dot(A,W),b)
            cache['Z'+str(l+1)] = Z
            if l != L - 2:
                A = relu(Z)
            else:
                A = sigmoid(Z)
            cache['A'+str(l+1)] = A
            
        return A,cache

def backward(A,y,L,cache,parameters):
        """
        Implment Backward Propagation
        Arguments:
        --------
            A:last layers value,output layer values.
            y: true labels.
            cache:cache A and Z in forward
            parameters: include weights and bias.
        Returns:
        -------
            dparameters: include dW,db to using updating parameters.
            
        """
        m = y.shape[0]
        dparameters = {}
        for l in range(L-1,0,-1):
            if l == L-1:
                dZ = A - y
            else:
                dZ = np.multiply(dA,np.int64(cache['Z'+str(l)]>0))
                
            A = cache['A'+str(l-1)]
            dW = np.dot(A.T,dZ) /m
            db = np.sum(dZ,axis=0,keepdims=True) / m
            dparameters['dW'+str(l)] = dW
            dparameters['db'+str(l)] = db
            
            if l != 1:
                W = parameters['W'+str(l)]
                dA = np.dot(dZ,W.T)
        return dparameters

def score(data,labels,L,parameters):
        """
        Get correct rate of target data set.
        Arguments:
        ---------
            data: score data.
            labels: score labels.
            parameters: The best weights and bias.
        Return:
        ------
            accuracy of target data set.
        """
        m = labels.shape[0]
        A,_ = forward(X=data,L=L,parameters=parameters)
        predict = np.round(A)
        
        acc = np.equal(predict,labels).sum()/ m
        
        return acc

def plot_decision_boundary(X,Y,L,parameters,title):
        """
        plot decision boundary
        """
        # Set min and max values and give it some padding
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

        x = np.arange(x_min,x_max,0.01)
        y = np.arange(y_min,y_max,0.01)
        xx,yy = np.meshgrid(x,y) # meshgrid x and y.

        X_ = np.c_[xx.ravel(),yy.ravel()] # shape like (m,n)

        A,_ = forward(X=X_,L=L,parameters=parameters) # predict 
        
        y_hat = np.round(A) # predict y
        
        fig = plt.figure()
        plt.subplot(1,1,1)
        plt.title('model with {} '.format(title))
        plt.scatter(X_[:,0], X_[:,1], c=np.squeeze(y_hat))
        plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y),s=5,cmap=plt.cm.Spectral,linewidths=1)



