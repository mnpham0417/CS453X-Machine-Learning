import numpy as np
import matplotlib.pyplot as plt
import math
from  scipy import ndimage
from skimage import transform, util
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

def shuffle(X, y):
    index = np.random.permutation(len(y))
    return X[index], y[index]

def shuffle(X, y):
    index = np.random.permutation(len(y))
    return X[index], y[index]

def reshapeAndAppend1s (matrix):
    matrix_reshaped = matrix.T
    matrix_ones = np.ones(matrix_reshaped.shape[1])
    return np.vstack((matrix_reshaped,matrix_ones)) 

def softmax(X,w):
    z = w.T.dot(X)
    return np.exp(z)/np.sum(np.exp(z), axis = 0)

def fCE (yhat, y):
    return -np.sum(y*np.log(yhat))/len(y)

def fPC(yhat, y):
    result = np.argmax(yhat, axis = 1) - np.argmax(y, axis = 1)
    return len(result[result == 0])/len(y)

def gradfCE (w, Xtilde, y):
    n = len(y)
    return Xtilde.dot(softmax(Xtilde,w).T-y)/n

def gradientDescent (X, y, batch, epoch):
    EPSILON = 0.1
    #Initialize w
    weights = np.random.normal(0,0.1,(X_train.shape[1],10))
    ones = np.ones((1,10))
    w = np.vstack((weights,ones))
    
    for itr in range(epoch):
        '''
        Apparently the correct mini-batch gradient descent
        requires shuffling data each epoch
        '''
        X, y = shuffle(X, y)
        ind = 0
        for ep in range(math.ceil(len(y)/epoch)):
            Xbatch = reshapeAndAppend1s(X[ind:ind+batch])
            w = w - EPSILON*gradfCE(w,Xbatch,y[ind:ind+batch])
            ind = ind + batch
        if (itr >= 80):
            Xtilde_train = reshapeAndAppend1s(X)
            yhat_train = softmax(Xtilde_train,w).T
            print("Epoch: {}".format(itr+1))
            print("Training Accuracy in epoch: {}".format(fPC(yhat_train,y)))
            print("Training Cross Entropy: {}".format(fCE(yhat_train, y)))
            print()
    return w

def translation_helper(image):
    result = image.reshape(28,28)
    return ndimage.interpolation.shift(result,(-1,2)).reshape(28*28)

def translation(X):
    return np.apply_along_axis(translation_helper, 1, X)

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

def zoom_helper(image):
    result = image.reshape(28,28)
    image = ndimage.zoom(result,(1.2,1.2), prefilter = False)
    return crop_center(image,28,28).reshape(28*28)

def zoom(X):
    return np.apply_along_axis(zoom_helper, 1, X)

def rotate_helper(image):
    result = image.reshape(28,28)
    image = transform.rotate(result,20)
    return image.reshape(28*28)

def rotate(X):
    return np.apply_along_axis(rotate_helper,1, X)

def noise_helper(image):
    result = image.reshape(28,28)
    image = util.random_noise(result)
    return image.reshape(28*28)

def noise(X):
    return np.apply_along_axis(noise_helper,1, X)

def augmentation(X, y):
    ind_translation = np.random.permutation(2500)
    X_translation = translation(X[ind_translation])
    y_translation = y[ind_translation]
    
    ind_zoom = np.random.permutation(2500)
    X_zoom = zoom(X[ind_zoom])
    y_zoom = y[ind_zoom]
    
    ind_rotate = np.random.permutation(2500)
    X_rotate = rotate(X[ind_rotate])
    y_rotate = y[ind_rotate]
    
    ind_noise = np.random.permutation(2500)
    X_noise = noise(X[ind_noise])
    y_noise = y[ind_noise]
    
    X_result = np.vstack((X,X_translation,X_zoom,X_rotate,X_noise))
    y_result = np.vstack((y,y_translation,y_zoom,y_rotate,y_noise))
    
    return X_result, y_result

if __name__ == "__main__":
    X_train = np.load("small_mnist_train_images.npy")
    X_test = np.load("small_mnist_test_images.npy")
    y_train = np.load("small_mnist_train_labels.npy")
    y_test = np.load("small_mnist_test_labels.npy")
    
    #Part 1
    print("PART I")
    w = gradientDescent(X_train,y_train,100,100)
    Xtilde_train = reshapeAndAppend1s(X_train)
    yhat_train = softmax(Xtilde_train,w).T

    #Training
    print("Training")
    print("Accuracy: {}".format(fPC(yhat_train,y_train)))
    print("Cross Entropy: {} \n".format(fCE(yhat_train, y_train)))

    Xtilde_test = reshapeAndAppend1s(X_test)
    yhat_test = softmax(Xtilde_test,w).T

    #Testing
    print("Testing")
    print("Accuracy: {}".format(fPC(yhat_test,y_test)))
    print("Cross Entropy: {} \n".format(fCE(yhat_test, y_test)))

    #Part 2
    print("PART II")
    X_train_aug, y_train_aug = augmentation(X_train, y_train)
    w_aug = gradientDescent(X_train_aug,y_train_aug,100,100)

    Xtilde_train_aug = reshapeAndAppend1s(X_train_aug)
    yhat_train_aug = softmax(Xtilde_train_aug,w_aug).T

    #Training
    print("Training")
    print("Accuracy: {}".format(fPC(yhat_train_aug,y_train_aug)))
    print("Cross Entropy: {} \n".format(fCE(yhat_train_aug, y_train_aug)))

    Xtilde_test = reshapeAndAppend1s(X_test)
    yhat_test = softmax(Xtilde_test,w_aug).T

    #Testing
    print("Testing")
    print("Accuracy: {}".format(fPC(yhat_test,y_test)))
    print("Cross Entropy: {}".format(fCE(yhat_test, y_test)))