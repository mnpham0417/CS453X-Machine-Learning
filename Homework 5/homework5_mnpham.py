import numpy as np
import matplotlib.pyplot as plt
import math

#From homework 1
def helper(A, k):
    index = np.linalg.eigh(A)[0].argsort()[::-1][:k]
    return np.linalg.eigh(A)[1][:,index]

class PCA453X ():
    def __init__ (self, n_components=2):
        self.n_components = n_components

    def fit (self, X):
        mean_vector = np.mean(X, axis = 0)
        Xtilde = (X - mean_vector).T
        X_pca = X.dot(helper(Xtilde.dot(Xtilde.T),self.n_components))
        self.pca = X_pca
        
    def transform (self, x):
        return self.pca
    
    '''
    I did not actually use this function but added it anyway
    to make it similar to sklearn's API.
    '''
    def fit_transform (self, X):
        mean_vector = np.mean(X, axis = 0)
        Xtilde = (X - mean_vector).T
        X_pca = X.dot(helper(Xtilde.dot(Xtilde.T),self.n_components))
        self.pca = X_pca
        return self.pca

if __name__ == "__main__":
    #Load Data
    X_test = np.load("small_mnist_test_images.npy")
    y_test = np.load("small_mnist_test_labels.npy")

    pca_custom = PCA453X(2)
    pca_custom.fit(X_test)
    pca = pca_custom.transform(X_test)

    color = ["gold", "salmon","r","cyan","pink","crimson","blue","green"
         ,"aqua","maroon"]
    label_Xte = np.argmax(y_test, axis = 1)
    for i in range(len(np.unique(label_Xte))):
        index = np.where(label_Xte == i)[0]
        plt.scatter(pca[index,0],pca[index,1], s = 1, c = color[i])
    plt.show()

    '''
    Uncomment this part so you can compare the plot with sklearn PCA
    '''
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2)
    # pca.fit(X_test)
    # pca_sklearn = pca.transform(X_test)

    # for i in range(len(np.unique(label_Xte))):
    #     index = np.where(label_Xte == i)[0]
    #     plt.scatter(-pca_sklearn[index,0],pca_sklearn[index,1], s = 1, c = color[i])
    # plt.show()