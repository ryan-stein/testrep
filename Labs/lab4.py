import numpy as np


# pre lab modifications to fixed point method

def fixedpt(f,x0,tol,Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    x_vals = np.zeros((Nmax, 1))
    x_vals[0] = x0

    count = 0
    while (count <Nmax):
        count = count +1
        x1 = f(x0)
        x_vals[count] = x1

        if (abs(x1-x0) <tol):
            return x_vals[:count], 0
        
        x0 = x1

    return x_vals, 1