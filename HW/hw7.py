import numpy as np
from scipy import io, integrate, linalg, signal
import matplotlib.pyplot as plt
import math


def prob1():

    # Problem 1 b.)

    print("Problem 1 b.)")

    # define f
    f = lambda x: 1/(1 + (10*x)**2)

    # compute exact f(x) for comparison
    x_exact = np.linspace(-1,1,1001)
    y_exact = f(x_exact)

    
    # I am going to compute and plot the function interpolation for N = 2, 3, ... , 20
    for i in range(2, 21):

        #create interpolation nodes:
        N = i 
        h = 2/(N-1)
        xint = np.zeros(N)

        for j in range(1, N+1):
            xint[j-1] = -1 + (j-1)*h

        yint = f(xint)

        #create vandermonde matrix and inverse:
        V = Vandermonde(xint, N)

        #compute coefficent vector
        c = np.linalg.solve(V,yint)

        #evaluate polynomial interpolation
        Neval = 1000    
        xeval = np.linspace(-1,1, Neval)
        yeval = eval_monomial(xeval,c,N,Neval)


        #create plot:

        plt.figure()
        plt.scatter(xint, yint, color = 'black', marker = 'o', label = 'Interpolation Nodes')
        plt.plot(x_exact, y_exact, color = 'red', label = 'Exact Function')
        plt.plot(xeval,yeval, color = 'blue', label = f'Polynomial Interpolation with {N} Nodes')
        plt.legend()
        plt.show()

    
def prob2():
   
    print("Problem 2: Lagrange Interpolation")

    # define f
    f = lambda x: 1/(1 + (10*x)**2)

    # compute exact f(x) for comparison
    x_exact = np.linspace(-1,1,1001)
    y_exact = f(x_exact)

    
    # I am going to compute and plot the function interpolation for N = 2, 3, ... , 20
    for i in range(2, 21):

        #create interpolation nodes:
        N = i 
        h = 2/(N-1)
        xint = np.zeros(N)

        for j in range(1, N+1):
            xint[j-1] = -1 + (j-1)*h

        yint = f(xint)

        
        Neval = 1000    
        xeval = np.linspace(-1,1, Neval)

        # evalaute p(x) using lagrange polynomials
        yeval = np.zeros(Neval)

        for i in range(Neval):
           yeval[i] = p(xeval[i], xint, yint, N)


        #create plot:

        plt.figure()
        plt.scatter(xint, yint, color = 'black', marker = 'o', label = 'Interpolation Nodes')
        plt.plot(x_exact, y_exact, color = 'red', label = 'Exact Function')
        plt.plot(xeval,yeval, color = 'blue', label = f'Lagrange Interpolation with {N} Nodes')
        plt.legend()
        plt.show()






# defining routines

def  eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval)
    
    for j in range(1, N):
      for i in range(Neval):
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval


def Vandermonde(xint,N):

    V = np.zeros((N,N))
    
    ''' fill the first column'''
    for j in range(N):
       V[j][0] = 1.0

    for i in range(1,N):
        for j in range(N):
           V[j][i] = xint[j]**i

    return V


# Problem 2 functions:

def phi(x, xint, N):
   
    prod = 1

    for i in range(N):
      prod *= (x - xint[i])

    return prod


def wj(j, x_j, xint, N):
   
   denominator = 1

   for i in range(N):
      
      #check i != j
      if i != j:
         denominator *= (x_j - xint[i])

   return 1 / denominator


def p(x, xint, yint, N):

    # If x is exactly one of the nodes, return the known value
    for j in range(N):
        if np.isclose(x, xint[j]):
            return yint[j]
        
    # otherwise compute the normal formula
   
    phi_n = phi(x, xint, N)

   #initialize sum:
    sum = 0

    for j in range(N):
      
        # evaluate x_j
        x_j = xint[j]

        #evaluate w_j
        w_j = wj(j, x_j, xint, N)
         
        sum += (w_j * yint[j])/(x-x_j)

    return phi_n * sum





# prob1()
prob2()
