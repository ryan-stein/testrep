import numpy as np
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt

def driver():

    print("HW 5:")
    print("\n")


    # Problem 1a.)

    #call oneA_iterator and print results with tolerance to 8 digits of accuracy

    [xstar, ier, its] = oneA_iterator(1, 1, 1e-8, 200)

    print("Problem 1 a.):")
    print()
    print("the approximate root is: ", xstar)
    print("the error message reads: ", ier)
    print("the total number of iterations used = ", its)
    
    
    






# Defining routines

#f and g for problem 1
def f_1(x, y):
        return 3*x**2 - y**2
    
def g_1(x, y):
        return 3*x*y**2 - x**3 -1

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




driver()