# import libraries
import numpy as np
import matplotlib.pyplot as plt
import sympy

def driver():

# finding root for problem 1 part c.)

    f_1 = lambda x: 2*x - 1 - np.sin(x)
    a_1 = 0
    b_1 = np.pi / 2
    tol_1 = 1e-8

    [astar,ier,n_it] = bisection(f_1,a_1,b_1,tol_1)
    print('Problem 1 c.):')
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('the total number of iterations used = ', n_it)



# problem 2 a.)

    f_2a = lambda x: (x - 5)**9
    a_2 = 4.82
    b_2 = 5.2
    tol_2 = 1e-4

    [astar_2a,ier_2a,n_it_2a] = bisection(f_2a,a_2,b_2,tol_2)
    print('Problem 2 a.): f(x) = (x-5)^9')
    print('the approximate root is',astar_2a)
    print('the error message reads:',ier_2a)
    print('the total number of iterations used = ', n_it_2a)

# problem 2 b.)

    # get expanded form using sympy library
    x = sympy.Symbol('x', real=True)
    f_2_sym = (x - 5)**9
    f_2_expanded_sym = sympy.expand(f_2_sym)

    # converted expanded sympy function to usable function in bisection:
    f_2_expanded = sympy.lambdify(x, f_2_expanded_sym, 'numpy')

    [astar_2b,ier_2b,n_it_2b] = bisection(f_2_expanded,a_2,b_2,tol_2)
    print('Problem 2 b.): expanded f')
    print('the approximate root is',astar_2b)
    print('the error message reads:',ier_2b)
    print('the total number of iterations used = ', n_it_2b)



# problem 3 b.)

    f_3 = lambda x: x**3 + x - 4
    a_3 = 1
    b_3 = 4
    tol_3 = 1e-3

    [astar,ier,n_it] = bisection(f_3,a_3,b_3,tol_3)
    print('Problem 3 b.):')
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('the total number of iterations used = ', n_it)     


# problem 5 a.)

f_5 = lambda x: x - 4*np.sin(2*x) - 3

x_vals = np.linspace(-2, 8, 1000) #further explaination for this interval in the hw
y_vals = f_5(x_vals)

# create figure
plt.figure()
plt.plot(x_vals, y_vals, label=r'$f(x) = x - 4\sin(2x) - 3$')
plt.axhline(0, color='black', linewidth=1, linestyle='--')  # horizontal line y = 0
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of $f(x) = x - 4\\sin(2x) - 3$')
plt.grid(True)
plt.legend()
plt.show()





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

