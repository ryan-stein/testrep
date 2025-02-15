# import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy

def driver():

    # evaluating erf at different values to determine good guess for x_bar
    g1 = scipy.special.erf(1)
    g2 = scipy.special.erf(1/2)
    g3 = scipy.special.erf(1/4)
    g4 = scipy.special.erf(1/8)

    print("erf tests: ", g1, g2, g3, g4)


    # problem 1a f(x)
    T_i = 20
    T_s = -15
    alpha = 0.138e-6
    t = 60 * 24 * 60 * 60

    f_1 = lambda x: T_s + (T_i - T_s)*scipy.special.erf(x/(2*np.sqrt(alpha*t)))

    x_vals = np.linspace(0, 0.85, 300)
    y_vals = f_1(x_vals)

    #problem 1a plot

    plt.figure()
    plt.plot(x_vals, y_vals)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')  # horizontal line y = 0
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Plot of $f(x)$')
    plt.grid(True)
    plt.legend()
    plt.show()

    return






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



def fixedpt(f,x0,tol,Nmax):
    ''' x0 = initial guess'''
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''
    count = 0
    while (count <Nmax):
      count = count +1
      x1 = f(x0)
      if (x1 > 0): # insures relative error does not divide by zero
        rel_err = abs(x1 - x0) / abs(x1)
      else:
         rel_err = abs(x1 - x0)

      if (rel_err <tol):
        xstar = x1
        ier = 0
        return [xstar,ier]
      x0 = x1

    xstar = x1
    ier = 1
    return [xstar, ier, count]


      
driver()               

























