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


    #problem 1b (bisection)

    [astar_1b,ier_1b,n_it_1b] = bisection(f_1, 0, 0.85, 10e-13)

    print('Problem 1 b.):')
    print('the approximate root is: ',astar_1b)
    print('the error message reads: ',ier_1b)
    print('the total number of iterations used = ', n_it_1b)


    #problem 1c (newton)
    
    #define f'
    f_1p = lambda x: ((T_i - T_s)/np.sqrt(alpha*t*np.pi)) * np.exp(-1* (x/(2*np.sqrt(alpha*t)))**2)

    #plug into newton method using the two values of x_0:
    [p_c1,pstar_c1,info_c1,it_c1] = newton(f_1, f_1p, 0.01, 10e-13, 200) #x_0 = 0.01  
    [p_c2,pstar_c2,info_c2,it_c2] = newton(f_1, f_1p, 0.85, 10e-13, 200) #x_0 = x_bar = 0.85


    print('Problem 1 c.):')
    print('using x_0 = 0.01')
    print('the approximate root is: ',pstar_c1)
    print('the error message reads: ',info_c1)
    print('the total number of iterations used = ', it_c1)
    print('')
    print('using x_0 = x_bar = 0.85')
    print('the approximate root is: ',pstar_c2)
    print('the error message reads: ',info_c2)
    print('the total number of iterations used = ', it_c2)








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



def newton(f,fp,p0,tol,Nmax):
    """
    Newton iteration.
    Inputs:
        f,fp - function and derivative
        p0 - initial guess for root
        tol - iteration stops when p_n,p_{n+1} are within tol
        Nmax - max number of iterations
    Returns:
        p - an array of the iterates
        pstar - the last iterate
        info - success message
            - 0 if we met tol
            - 1 if we hit Nmax iterations (fail)
    """
    p = np.zeros(Nmax+1)
    p[0] = p0
    for it in range(Nmax):
        p1 = p0-f(p0)/fp(p0)
        p[it+1] = p1
        if (abs(p1-p0) < tol):
            pstar = p1
            info = 0
            return [p,pstar,info,it]
        p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]


      
driver()               

























