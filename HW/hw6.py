import numpy as np
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
import math



def driver():

    print("HW #6:")
    print("\n")

    # initial guess vectors:
    x0_i = np.array([1, 1])
    x0_ii = np.array([1, -1])
    x0_iii = np.array([0, 0])


    #perform each method for each initial guess:

    #Newton
    [xstar_N_1, ier_N_1, its_N_1] = Newton(x0_i, 10e-8, 100)
    [xstar_N_2, ier_N_2, its_N_2] = Newton(x0_ii, 10e-8, 100)
    #[xstar_N_3, ier_N_3, its_N_3] = Newton(x0_iii, 10e-8, 100)

    #Broyden
    [xstar_B_1, ier_B_1, its_B_1] = broyden_method(x0_i, 10e-8, 100)
    [xstar_B_2, ier_B_2, its_B_2] = broyden_method(x0_ii, 10e-8, 100)
    #[xstar_B_3, ier_B_3, its_B_3] = broyden_method(x0_iii, 10e-8, 100)

    #Lazy Newton
    [xstar_L_1, ier_L_1, its_L_1] = lazyNewton(x0_i, 10e-8, 100)
    [xstar_L_2, ier_L_2, its_L_2] = lazyNewton(x0_ii, 10e-8, 100)
    #[xstar_L_3, ier_L_3, its_L_3] = lazyNewton(x0_iii, 10e-8, 100)








    print("Problem 1:")
    print()
    print("For initial guess [1, 1]:")
    print()
    print("Using Newton's Method")
    print("the approximate root is: ", xstar_N_1)
    print("the error message reads: ", ier_N_1)
    print("the total number of iterations used = ", its_N_1)
    print()
    print("Using Broyden's Method")
    print("the approximate root is: ", xstar_B_1)
    print("the error message reads: ", ier_B_1)
    print("the total number of iterations used = ", its_B_1)
    print()
    print("Using Lazy Newton's Method")
    print("the approximate root is: ", xstar_L_1)
    print("the error message reads: ", ier_L_1)
    print("the total number of iterations used = ", its_L_1)

    print("\n")

    print("For initial guess [1, -1]:")
    print()
    print("Using Newton's Method")
    print("the approximate root is: ", xstar_N_2)
    print("the error message reads: ", ier_N_2)
    print("the total number of iterations used = ", its_N_2)
    print()
    print("Using Broyden's Method")
    print("the approximate root is: ", xstar_B_2)
    print("the error message reads: ", ier_B_2)
    print("the total number of iterations used = ", its_B_2)
    print()
    print("Using Lazy Newton's Method")
    print("the approximate root is: ", xstar_L_2)
    print("the error message reads: ", ier_L_2)
    print("the total number of iterations used = ", its_L_2)

    print("\n")

    print("For initial guess [0, 0]:")
    print("The Jacobian evaluated at (0,0) is singular so none of the methods will work.")

    """
    print("Using Newton's Method")
    print("the approximate root is: ", xstar_N_3)
    print("the error message reads: ", ier_N_3)
    print("the total number of iterations used = ", its_N_3)
    print()
    print("Using Broyden's Method")
    print("the approximate root is: ", xstar_B_3)
    print("the error message reads: ", ier_B_3)
    print("the total number of iterations used = ", its_B_3)
    print()
    print("Using Lazy Newton's Method")
    print("the approximate root is: ", xstar_L_3)
    print("the error message reads: ", ier_L_3)
    print("the total number of iterations used = ", its_L_3)

    """
    


#Defining routines:
def evalJ_1(x):
     J = np.array([[2*x[0], 2*x[1]], 
                   [np.exp(x[0]), 1]])
     return J

def evalF_1(x):
     F = np.zeros(2)

     F[0] = x[0]**2 + x[1]**2 - 4
     F[1] = np.exp(x[0]) + x[1] - 1

     return F


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



import numpy as np

def broyden_method(x0, tol, Nmax):
    """
    Broyden's method for solving f(x) = 0.
    
    Parameters:
      x0   : initial guess (numpy array)
      tol  : tolerance for convergence (based on step size)
      Nmax : maximum number of iterations
    
    Returns:
      [x_star, ier, its]
         x_star: approximate solution
         ier   : error flag (0 if converged, 1 if not)
         its   : number of iterations performed
    """
    #initialize Broyden matrix
    B = evalJ_1(x0)
    
    
    for its in range(Nmax):
        F = evalF_1(x0)
        
        p = np.linalg.solve(B, -F) # note this is equivalent to delta x so it is what I'll use in the Broyden updates
        x1 = x0 + p
        
        # Check convergence 
        if (np.linalg.norm(x1-x0, ord=2) < tol):
            xstar = x1
            ier =0
            return[xstar, ier, its]
        
        # create vectors for Broyden update
        r = (evalF_1(x1) - F) - B @ p
        denominator = np.dot(p, p)
        
        # Safeguard: denominator should not be zero
        if abs(denominator) < 1e-14:
            return [x1, 1, its + 1]
        
        u = r / denominator

        # Broyden update 

        B1 = B + np.outer(u, p)
        
        # Update x for next iteration
        x0 = x1
        B = B1
        
    # If we exit the loop without converging, return with ier=1
    return [x1, 1, Nmax]




def lazyNewton(x0, tol, Nmax):
    """
    Slacker Newton 
    * x0: initial guess (NumPy array)
    * tol: convergence tolerance
    * Nmax: maximum number of iterations
    
    Returns: [xstar, ier, its]
    - xstar: approximate root
    - ier: 0 if converged, 1 if not converged after Nmax
    - its: the iteration count at exit
    """
    
    # Compute the Jacobian once at the start
    J0 = evalJ_1(x0)
    
    for its in range(Nmax):
        F0 = evalF_1(x0)   # <-- your function for computing F(x0)
        
        # Solve J(x0)*p = -F(x0) for the Newton step p
        p = np.linalg.solve(J0, -F0)
        
        # Update the estimate
        x1 = x0 + p

        # Check convergence using step size
        if np.linalg.norm(x1 - x0) < tol:
            xstar = x1
            ier = 0
            return [xstar, ier, its]
        
        x0 = x1 
        

    xstar = x1
    ier = 1
    return [xstar, ier, its]



driver()






























