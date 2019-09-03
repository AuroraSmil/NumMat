# Arrary and stuff
import numpy as np
# Linear algebra solvers from scipy
import scipy.linalg as la
# Basic plotting routines from the matplotlib library
import matplotlib.pyplot as plt

import random

def forward_sub(L, b):
    #print("l", L, "b", b)
    n = len(b)
    y=np.zeros(n)
    for i in range(n):
        #print("b", b[i])
        y_temp = b[i]
        for j in range(i):
            y_temp-= L[i][j]*y[j]

        y[i] =y_temp
    return y


def backward_sub(U, y):
    """Given a lower triangular matrix U and right-side vector y,
    compute the solution vector x solving Ux = y."""
    # ...
    n = len(y)
    x = np.zeros(n)
    for i in range(n):
        x_temp = y[n-i-1]

        for j in range(i):
            x_temp -= U[n - i-1][n - j - 1] * x[n - j-1]
        x_temp =x_temp / U[n - i-1][n - i-1]
        x[n - i - 1] = x_temp


    return x

def lu(A: np.ndarray):
    m, n = A.shape
    Y = np.zeros(A.shape)
    assert m == n, "Only square systems"

    for k in range(n - 1):
        if A[k, k] == 0:
            raise Exception("Null pivot element")
        A[k+1:n, k] = A[k+1:n, k] / A[k, k]
        for j in range(k + 1, n):
            i = np.arange(k + 1, n)
            A[i, j] = A[i, j] - A[i, k] * A[k, j]


    U = np.triu(A, k=0)
    L = np.tril(A, k=-1) + np.identity(m)

    return L, U

def lu_test(A: np.ndarray):
    m, n = A.shape
    Y = np.zeros(A.shape)
    assert m == n, "Only square systems"

    for j in range(n):
        if A[j, j] == 0:
            raise Exception("Null pivot element")
        for k in range(j-1):
            i = np.arange(k+1,n)
            A[i, j] = A[i, j] - A[i, k] * A[k, j]

        i = np.arange(j+1, n)
        A[i, j] = A[i, j] / A[j, j]

    U = np.triu(A, k=0)
    L = np.tril(A, k=-1) + np.identity(m)

    return L, U

def lu_forelesnign(A):
    n = len(A)
    L = np.identity(n)
    U = np.zeros(A.shape)


    for i in range(n):
        if A[i, i] == 0:
            raise Exception("Null pivot element")
        for j in range(i):
            L[i][j] = (1/(U[j][j]))*(A[i][j])
            for k in range(j):
                L[i][j] -= L[i][k]*U[k][j]/U[j][j]

        for j in range(i,n):
            U[i][j] = A[i][j]
            for k in range(i):
                U[i][j] -= L[i][k]*U[k][j]


    return L,U


A = np.random.randint(0,20, (5,5))
print("A correct")
print(A)


A_test = np.array(
[[19, 13,  6, 9, 16],
 [ 2, 18,  1,  1,  8],
 [12, 17, 12, 10,  6],
 [ 5,  4,  6, 18, 11],
 [16,  7,  1, 17,  4]])
A_hil = la.hilbert(5)

#print(A_test)
L, U = lu_forelesnign(A)

print("L")
print(L)
print("U")
print(U)
print( "LU")
print(np.matmul(L, U))




'''
L = np.array([[1,0,0], [2,1,0], [3,4,1]])
b = [1,4,14]
y = forward_sub(L, b)


U = [[2,2,3], [0,1,4], [0,0,2]]
b_1 = [15,14,6]

x = backward_sub(U, b_1)
print(x)
'''

def lu_solve(L, U, b):



    # Step 1: Solve Uy = b using forward substitution

    # Step 2: Solve Lx = y using backward substitution

    return x

#def linear_solve(A, b):
    L, U = LU(A)
    #print("l", L, "u", U)
    x = lu_solve(L, U, b)
    return x


A = a =np.array([[9,8,7], [7,6,5], [5,4,3]]) #np.matrix('1,2,3; 4,5,6; 7,8,9')

x = [10,11,12]


''''
print("b", b)
print(A[0][0])

y = linear_solve(A, b)
print(y)
'''