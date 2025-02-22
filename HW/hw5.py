import numpy as np
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
import math

def driver():

    print("HW 5:")
    print("\n")


    # Problem 1a.)

    #call oneA_iterator and print results with tolerance to 8 digits of accuracy

    [xstar_1a, ier_1a, its_1a] = oneA_iterator(1, 1, 1e-10, 100)

    print("Problem 1 a.):")
    print()
    print("the approximate root is: ", xstar_1a)
    print("the error message reads: ", ier_1a)
    print("the total number of iterations used = ", its_1a)
    

    # Problem 1c.): Newton method

    print("\n")
    
    x0_1c = np.array([1,1])

    [xstar_1c, ier_1c, its_1c] = Newton(x0_1c, 1e-10, 100)

    print("Problem 1 c.): Newton Method")
    print()
    print("the approximate root is: ", xstar_1c)
    print("the error message reads: ", ier_1c)
    print("the total number of iterations used = ", its_1c)

    
    






# Defining routines

#f and g for problem 1a
def f_1(x, y):
        return 3*x**2 - y**2
    
def g_1(x, y):
        return 3*x*y**2 - x**3 -1

# I'm going to combine evaluating f and g into one funtion for the newton method so it is more similar to the class example. I did not originally
# do this for 1 a.) so the inputs and style of oneA_iterator are slightly different

def evalJ_1(x):
     J = np.array([[6*x[0], -2*x[1]], 
                   [3*x[1]**2 - 3*x[0]**2, 6*x[0]*x[1]]])
     return J

def evalF_1(x):
     F = np.zeros(2)

     F[0] = 3*x[0]**2 - x[1]**2
     F[1] = 3*x[0]*x[1]**2 - x[0]**3 -1

     return F

def oneA_iterator(x0, y0, tol, Nmax):

    """
    1a iterator 
    * x0: initial guess for x
    * y0: initial guess for y
    * tol: convergence tolerance
    * Nmax: maximum number of iterations
    
    Returns: [xstar, ier, its]
    - xstar: approximate root
    - ier: 0 if converged, 1 if not converged after Nmax
    - its: the iteration count at exit
    """
    

    # matrix used to update [x,y]
    M = np.array([[1/6, 1/18], 
                  [0, 1/6]])

    for its in range(Nmax):

        F_val = f_1(x0, y0)
        G_val = g_1(x0, y0)
        FG_vec = np.array([F_val, G_val])

        xy_vec = np.array([x0,y0])

        update = M @ FG_vec #perfrom matrix multiplaction on FG vector

        xy_new = xy_vec - update #create [x1, y1]

        #check for convergence:
        if np.linalg.norm(xy_new - xy_vec, ord=2) < tol:
            xstar = xy_new
            ier = 0
            return [xstar, ier, its + 1]
        
        #update x0, and y0 values
        x0, y0 = xy_new

    xstar = xy_new
    ier = 1
    return [xstar, ier, Nmax]


def Newton(x0,tol,Nmax):
    ''' inputs: x0 = initial guess, tol = tolerance, Nmax = max its'''
    ''' Outputs: xstar= approx root, ier = error message, its = num its'''

    for its in range(Nmax):
        J = evalJ_1(x0)
        F = evalF_1(x0)
       
       # like explained in class im not going to compute the inverse of J and will instead use np.linalg.solve
        p = np.linalg.solve(J, -F)
        x1 = x0 + p

        if (np.linalg.norm(x1-x0, ord=2) < tol):
            xstar = x1
            ier =0
            return[xstar, ier, its]
        
        x0 = x1

    xstar = x1
    ier = 1
    return[xstar,ier,its]




driver()