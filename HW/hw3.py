# import libraries
import numpy as np
import matplotlib.pyplot as plt

def driver():

# finding root for problem 1 part c.)

    f_1 = lambda x: 2*x - 1 - np.sin(x)
    a_1 = 0
    b_1 = np.pi / 2
    tol_1 = 1e-8

    [astar,ier,n_it] = bisection(f_1,a_1,b_1,tol_1)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('the total number of iterations used = ', n_it)




# define routines
def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success
#      n_it  - number of iterations used

#     first verify there is a root we can find in the interval 

    n_it = 0  # initialize iterations to 0

    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier, n_it]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier, n_it]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier, n_it]

    
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      n_it += 1
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier, n_it]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)

      
    astar = d
    ier = 0
    return [astar, ier, n_it]
      
driver()               

