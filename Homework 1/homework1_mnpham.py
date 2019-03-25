'''
Author: Minh Pham
Username: mnpham
'''

import numpy as np

#Problem 1
def problem1(A, B):
    return A + B

#Problem 2
def problem2(A, B, C):
    return A.dot(B) - C

#Problem 3
def problem3(A, B, C):
    return A*B + C.T

#Problem 4
def problem4(x, y):
    x_transpose = x.T
    return x_transpose.dot(y)

#Problem 5
def problem5(A):
    return np.zeros(A.shape)

#Problem 6
def problem6(A):
    return np.ones(A.shape[0]).reshape(A.shape[0], 1)

#Problem 7
def problem7(A, scalar):
    return A + scalar*np.eye(A.shape[0])

#Problem 8
def problem8(A, i, j):
    return A[i,j]

#Problem 9
def problem9(A, i):
    return np.sum(A[i])

#Problem 10
def problem10(A, c, d):
    result = A[np.nonzero((A >= c) & (A <= d))]
    return np.sum(result)/len(result)

#Problem 11
def problem11(A, k):
    index = np.linalg.eig(A)[0].argsort()[::-1][:k]
    return np.linalg.eig(A)[1][:,index]

#Problem 12
def problem12(A, x):
    return np.linalg.solve(A, x)

#Problem 13
def problem13 (A, x):
    ytranspose = np.linalg.solve(A.T, x.T)
    return ytranspose.T
    