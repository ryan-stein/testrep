# import libraries
import numpy as np

def driver():

# problem 1 function    
    f_1 = lambda x: x**2 *(x-1)

    #bounds for part a
    a_1 = 0.5
    b_1 = 2

    # bounds for part b
    a_2 = -1
    b_2 = 0.5

    # bounds for part c
    a_3 = -1
    b_3 = 2


#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-7

# output for part a
    [astar_1,ier_1] = bisection(f_1,a_1,b_1,tol)
    
    print('part a.):  (a,b) = (0.5, 2)')
    print('the approximate root is',astar_1)
    print('the error message reads:',ier_1)
    print('f(astar) =', f_1(astar_1))

# output for part b
    [astar_2,ier_2] = bisection(f_1,a_2,b_2,tol)
    
    print('part b.):  (a,b) = (-1, 0.5)')
    print('the approximate root is',astar_2)
    print('the error message reads:',ier_2)
    print('f(astar) =', f_1(astar_2))

# output for part c
    [astar_3,ier_3] = bisection(f_1,a_3,b_3,tol)
    
    print('part b.):  (a,b) = (-1, 0.5)')
    print('the approximate root is',astar_3)
    print('the error message reads:',ier_3)
    print('f(astar) =', f_1(astar_3))






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

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b)
    if (fa*fb>0):
       ier = 1
       astar = a
       return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier]
      
driver()               

