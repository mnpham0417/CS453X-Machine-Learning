import numpy as np
import matplotlib.pyplot as plt

def reshapeAndAppend1s (faces):
    faces_reshaped = faces.reshape(faces.shape[0],48*48).T
    faces_ones = np.ones(faces_reshaped.shape[1])
    return np.vstack((faces_reshaped,faces_ones))

def fMSE (w, Xtilde, y):
    n = len(y)
    print("Size: {}".format(n))
    yhat = (Xtilde.T).dot(w)
    return np.sum((yhat-y)**2)/(2*n)

def gradfMSE (w, Xtilde, y, alpha = 0.):
    n = len(Xtilde)
    return Xtilde.dot(np.dot(Xtilde.T,w)-y)/n

def method1 (Xtilde, y):
    A = np.dot(Xtilde, Xtilde.T)
    b = Xtilde.dot(y)
    return np.linalg.solve(A,b)

def method2 (Xtilde, y):
    return gradientDescent(Xtilde,y)

def method3 (Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde,y,ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 0.001  # Step size aka learning rate
    T = 5000
    #Initialize w
    w = np.random.rand(Xtilde.shape[0])
    w[-1] = 0
    for itr in range(T):
        w = w - EPSILON*(gradfMSE(w,Xtilde,y) - (alpha/len(y))*w)
    return w


if __name__ == "__main__":
    X_train = np.load("age_regression_Xtr.npy")
    X_test = np.load("age_regression_Xte.npy")
    y_train = np.load("age_regression_ytr.npy")
    y_test = np.load("age_regression_yte.npy")

    Xtilde = reshapeAndAppend1s(X_train)
    w_1 = method1(Xtilde, y_train)
    w_2 = method2(Xtilde, y_train)
    w_3 = method3(Xtilde, y_train)

    print("Method 1")
    print("Training MSE: {}".format(fMSE(w_1,Xtilde,y_train)))
    print("Testing MSE: {}".format(fMSE(w_1,reshapeAndAppend1s(X_test),y_test)))
    print()
    print("Method 2")
    print("Training MSE: {}".format(fMSE(w_2,Xtilde,y_train)))
    print("Testing MSE: {}".format(fMSE(w_2,reshapeAndAppend1s(X_test),y_test)))
    print()
    print("Method 3")
    print("Training MSE: {}".format(fMSE(w_3,Xtilde,y_train)))
    print("Testing MSE: {}".format(fMSE(w_3,reshapeAndAppend1s(X_test),y_test)))

	#Display weights
    plt.imshow(w_1[:-1].reshape(48,48))
    plt.show()
    plt.imshow(w_2[:-1].reshape(48,48))
    plt.show()
    plt.imshow(w_3[:-1].reshape(48,48))
    plt.show()

	#Show top 5 worse error
    Xtilde_test = reshapeAndAppend1s(X_test)
    y_3_test_pred = Xtilde_test.T.dot(w_3)
    distance = abs(y_3_test_pred - y_test).argsort()[-5:][::-1]
    for i in range(5):
    	print("Image at index: {}".format(distance[i]))
    	print("Predicted age: {}".format(y_3_test_pred[distance[i]]))
    	print("Actual age: {}".format(y_test[distance[i]]))
    	plt.imshow(X_test[distance[i]])
    	plt.show()
    	print()







