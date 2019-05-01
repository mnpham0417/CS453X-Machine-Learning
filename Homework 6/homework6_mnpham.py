import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize
import math
import random
from sklearn.decomposition import PCA

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

def relu(z):
    return z * (z > 0)

def reluPrime(x):
    return np.where(x > 0, 1.0, 0.0)
    
def softmax(z):
    z_cp = z.copy()
    return np.exp(z)/np.sum(np.exp(z), axis = 0)

def calculate_yhat(X, w):
    W1, b1, W2, b2 = unpack(w)
    '''
    W1 : 784 x 40
    b1 : 40, 
    W2 : 40 x 10
    b2 : 10, 
    '''
    z1 = (W1.T).dot(X) + b1.reshape(NUM_HIDDEN,1) # z1: 40 x 10000
    h = relu(z1) # h : 40 x 10000
    z2 = W2.T.dot(h) + b2.reshape(10,1) # z2: 10 x 10000
    yhat = softmax(z2) #yhat : 10 x 10000
    
    return yhat

def accuracy(yhat, y):
    '''
    yhat : 10 x batch
    y : batch x 10
    '''
    result = np.argmax(yhat, axis = 1) - np.argmax(y, axis = 1)
    N = len(result)
    return len(result[result == 0])/N

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.
def unpack (w):
    '''
    Output:
    W1 : 784 x 40
    b1 : 40, 
    W2 : 40 x 10
    b2 : 20, 
    '''
    w_cp = w.copy()
    ind1 = NUM_HIDDEN*NUM_INPUT
    ind2 = ind1 + NUM_HIDDEN
    ind3 = ind2 + NUM_OUTPUT*NUM_HIDDEN
    W1 = w_cp[0:ind1].reshape((NUM_INPUT, NUM_HIDDEN))
    b1 = w_cp[ind1:ind2].T.reshape(NUM_HIDDEN)
    W2 = w_cp[ind2:ind3].reshape((NUM_HIDDEN, NUM_OUTPUT))
    b2 = w_cp[ind3:].T.reshape(NUM_OUTPUT)
    return W1, b1, W2, b2


# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack (W1, b1, W2, b2):
    '''
    Param:
    W1 : 784 x 40
    b1 : 40, 
    W2 : 40 x 10
    b2 : 10, 
    '''
    W1_flatten = W1.flatten()
    b1_flatten = b1.flatten()
    W2_flatten = W2.flatten()
    b2_flatten = b2.flatten()
    w = np.concatenate([W1_flatten, b1_flatten, W2_flatten, b2_flatten])
    return w

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("mnist_{}_images.npy".format(which))
    labels = np.load("mnist_{}_labels.npy".format(which))
    
    return images, labels

def fCE_multiple (X, Y, w_lst):  
    '''
    X : 784 x 100
    W1 : 784 x 40
    b1 : 40, 
    W2 : 40 x 10
    b2 : 10, 
    '''
    result = []
    for w in w_lst:
        yhat = calculate_yhat(X, w).T #yhat : 10000 x 10
        N = yhat.shape[0]
        cost = -1/len(Y) * np.sum(np.sum(Y * np.log(yhat), axis = 1))
        result.append(cost)
    return result

def plotSGDPath (trainX, trainY, ws):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    
    pca = PCA(n_components=2)
    pca.fit(ws)
    
    pca_ws_transform = pca.transform(ws)
    min_ = np.min(pca_ws_transform)
    max_ = np.max(pca_ws_transform)
    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(min_-3, max_+3, 0.25)
    axis2 = np.arange(min_-3, max_+3, 0.25)
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            # print(i,j)
            weights = pca.inverse_transform([Xaxis[i,j], Yaxis[i,j]])
            Zaxis[i,j] = fCE(trainX.T, trainY, weights)
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

#   Now superimpose a scatter plot showing the weights during SGD.
    Xaxis_scatter = pca_ws_transform[:,0]
    Yaxis_scatter = pca_ws_transform[:,1]
    w_lst = []
    for i in range(len(Xaxis_scatter)):
        w_lst.append(pca.inverse_transform([Xaxis_scatter[i], Yaxis_scatter[i]]))
    Zaxis_scatter = fCE_multiple (trainX.T, trainY, w_lst)
    ax.scatter(Xaxis_scatter, Yaxis_scatter, Zaxis_scatter, color='r')

    plt.show()

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def fCE (X, Y, w):  
    '''
    X : 784 x 100
    W1 : 784 x 40
    b1 : 40, 
    W2 : 40 x 10
    b2 : 10, 
    '''
    yhat = calculate_yhat(X, w).T #yhat : 10000 x 10
    N = yhat.shape[0]
    cost = -1/len(Y) * np.sum(np.sum(Y * np.log(yhat), axis = 1))
    return cost

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).
def gradCE (X, Y, w):
    '''
    Input:
    X : 784 x 100
    Y : 100 x 10
    '''
    W1, b1, W2, b2 = unpack(w.copy())
    '''
    W1 : 784 x 40
    b1 : 40, 
    W2 : 40 x 10
    b2 : 10, 
    '''
    X_cp = X.copy()
    Y_cp = Y.copy()
    z1 = (W1.T).dot(X_cp) + b1.reshape(NUM_HIDDEN,1) # z1: 40 x 100
    
    h = relu(z1) # h : 40 x 100
    z2 = W2.T.dot(h) + b2.reshape(10,1) # z2: 10 x 100
    yhat = softmax(z2) #yhat : 10 x 100
    
    g_transpose = ((yhat - Y_cp.T).T.dot(W2.T))*reluPrime(z1.T) # g_transpose : 100 x 40
    g = g_transpose.T # g : 40 x 100
    
    #Calculate gradient
    grad_W2 = (yhat - Y_cp.T).dot(h.T).T/Y.shape[0] # grad_W2 : 40 x 10
    grad_b2 = np.average(yhat - Y.T, axis = 1) #grad_b2 : 10 x 1
    grad_b2 = grad_b2.reshape((10,1))

    grad_W1 = g.dot(X.T).T/Y.shape[0] # grad_W1 : 784 x 40
    grad_b1 = np.average(g, axis = 1).reshape(NUM_HIDDEN,1) # grad_b1 : 40 x 1

    return pack(grad_W1, grad_b1, grad_W2, grad_b2)

#This function initializes weights
def init_weights(NUM_INPUT, NUM_HIDDEN, NUM_OUTPUT):
    W1 = 2*(np.random.random(size=(NUM_INPUT, NUM_HIDDEN))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    w = pack(W1, b1, W2, b2) #(31810, 1)
    return w
    
# Given training and testing datasets and an initial set of weights/biases b,
# train the NN. Then return the sequence of w's obtained during SGD.
def train (trainX, trainY, testX, testY, params = {"batch_size":32, "num_epochs":20, "EPSILON":0.01, "NUM_HIDDEN":NUM_HIDDEN, "alpha":1e-2 }):
    global NUM_HIDDEN  # Number of hidden neurons
    
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    EPSILON = params["EPSILON"]
    NUM_HIDDEN = params["NUM_HIDDEN"]
    alpha = params["alpha"]
    history_weights = []
    w = init_weights(NUM_INPUT, NUM_HIDDEN, NUM_OUTPUT)
    W1,b1,W2,b2 = unpack(w)
    b1 = b1.reshape((NUM_HIDDEN,1))
    b2 = b2.reshape((10,1))
    
    #Just to make sure original weights are not affected
    W1_cp = W1.copy()
    b1_cp = b1.copy()
    W2_cp = W2.copy()
    b2_cp = b2.copy()
    for epoch in range(num_epochs):
        num_batches = trainX.shape[1] // batch_size
        for i in range(num_batches):
            
            current_trainX = trainX[:,i*batch_size:(i+1)*batch_size]
            
            current_trainY = trainY[i*batch_size:(i+1)*batch_size]

            grad_w1, grad_b1, grad_w2, grad_b2 = unpack(gradCE(current_trainX, current_trainY, pack(W1_cp,b1_cp,W2_cp,b2_cp)))
            
            W2_cp = W2_cp - EPSILON * (grad_w2 - (alpha/len(current_trainY))*W2_cp)
            b2_cp = b2_cp - EPSILON * (grad_b2.reshape((b2_cp.shape[0],1)) - (alpha/len(current_trainY))*b2_cp)
            W1_cp = W1_cp - EPSILON * (grad_w1 - (alpha/len(current_trainY))*W1_cp)
            b1_cp = b1_cp - EPSILON * (grad_b1.reshape((b1_cp.shape[0],1)) - (alpha/len(current_trainY))*b1_cp)

            #Store some weights for each epoch
            if((i % math.ceil(num_batches/3)) == 0):
                history_weights.append(pack(W1_cp,b1_cp,W2_cp,b2_cp))
        # Uncomment the codes below to print out the stats
        if(epoch <= 5 or epoch >= (num_epochs - 5)): #print out first and last 5 epochs
            print("Epoch " + str(epoch))
            print("Cross entropy loss: " + str(fCE(testX, testY, pack(W1,b1,W2,b2))))
            ytest_hat = calculate_yhat(testX, pack(W1, b1, W2, b2)).T
            ytrain_hat = calculate_yhat(trainX, pack(W1, b1, W2, b2)).T
            print("Train Accuracy: {}".format(accuracy(ytrain_hat, trainY)))
            print("Test Accuracy: {}".format(accuracy(ytest_hat, testY)))
            print()
    return history_weights, pack(W1_cp,b1_cp,W2_cp,b2_cp)
        
def generateParamsList(num):
    hidden_layer_lst = [30, 40, 50]
    learning_rate_lst = [0.01, 0.05, 0.01, 0.05, 0.1, 0.5]
    minibatch_sz_lst = [16, 32, 64, 128, 256]
    epoch_lst = [25, 50, 75]
    alpha_lst = [1e1, 1e0, 1e-1, 1e-2]
    history = []
    result = []
    while(len(result) < num):
        hidden_layer = random.randint(0, len(hidden_layer_lst)-1)
        learning_rate = random.randint(0, len(learning_rate_lst)-1)
        minibatch_sz = random.randint(0, len(minibatch_sz_lst)-1)
        epoch = random.randint(0, len(epoch_lst)-1)
        alpha = random.randint(0, len(alpha_lst)-1)
        params = [hidden_layer, learning_rate, minibatch_sz, epoch, alpha]
        if(params not in history):
            result.append({"batch_size":minibatch_sz_lst[minibatch_sz], 
                           "num_epochs":epoch_lst[epoch], 
                           "EPSILON":learning_rate_lst[learning_rate],
                          "NUM_HIDDEN":hidden_layer_lst[hidden_layer],
                          "alpha":alpha_lst[alpha]})
    return result 

def findBestHyperparameters(trainX, trainY, validationX, validationY, num):
    params_lst = generateParamsList(num)
    accuracy_dict = {}
    for i in range(num):
        print("Parameters: ")
        print(params_lst[i])
        print()
        weights = init_weights(NUM_INPUT, params_lst[i]["NUM_HIDDEN"], NUM_OUTPUT)
        history_weights_result, w_result = train(trainX, trainY, validationX, validationY, params_lst[i])
        y_validation_hat = calculate_yhat(validationX, w_result).T
        validation_acc = accuracy(y_validation_hat, validationY)
        accuracy_dict[validation_acc] = i
    
    best_index = sorted(accuracy_dict.keys())[-1]
    return params_lst[accuracy_dict[best_index]]

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")
        valX, valY = loadData("validation")
    
    #PART I
    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_INPUT, NUM_HIDDEN))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    w = pack(W1, b1, W2, b2) #(31810, 1)

    # Check that the gradient is correct on just a few examples (randomly drawn).
    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs,:]).T, np.atleast_2d(trainY[idxs,:]), w_), \
                                    lambda w_: gradCE(np.atleast_2d(trainX[idxs,:]).T, np.atleast_2d(trainY[idxs,:]), w_), \
                                    w.T))
    trainX_transpose = trainX.T
    testX_transpose = testX.T
    valX_transpose = valX.T
    best_params = findBestHyperparameters(trainX_transpose, trainY, valX_transpose, valY, 5)
    print(best_params)

    # Train the network and obtain the sequence of w's obtained using SGD.
    ws, optimal_weight = train(trainX_transpose, trainY, valX_transpose, valY, best_params)

    #PART II
    # Plot the SGD trajectory
    plotSGDPath(trainX, trainY, ws)