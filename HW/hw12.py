import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import math


def prob2():

    A = np.array([
        [12, 10, 4],
        [10, 8, -5],
        [4, -5, 3]
    ], dtype=float)


    T, Q = tridiag(A)
    print("Tridiagonal Matrix")
    print()
    print(T)
    print("\n")
    print("Householder Transformation Matrix:")
    print()
    print(Q)

    # check results
    check_A = Q @ T @ Q.T
    print("\n")
    print("Solution Check:")
    print()
    print(check_A)

    return





def tridiag(A):

    n = A.shape[0]
    T = A.copy()
    Q_total = np.eye(n)

    for i in range(n-2):

        x = T[i+1:, i]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x) * (1 if x[0] >= 0 else -1)
        u = x + e
        v = u / np.linalg.norm(u)

        H_i = np.eye(n)
        H_i[i+1:, i+1:] -= 2.0 * np.outer(v, v)
        T = H_i @ T @ H_i
        Q_total = Q_total @ H_i

    return T, Q_total


def create_hilbert(n):
    
    H = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)

    return H



def power_method(A, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    x = np.random.rand(n) #initial guess
    x = x / np.linalg.norm(x)

    eigen_curr = 0 #initialize eigenvalue estimate
    eigenvals = []
    for i in range(max_iter):

        x_new = A @ x
        x = x_new / np.linalg.norm(x_new)
        eigen_old = eigen_curr
        eigen_curr = x @ A @ x
        eigenvals.append(eigen_curr)

        if abs(eigen_curr - eigen_old) < tol:
            return eigen_curr, x, i+1, eigenvals
        
    return eigen_curr, x, max_iter, eigenvals



prob2()