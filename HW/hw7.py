import numpy as np
from scipy import io, integrate, linalg, signal
import matplotlib.pyplot as plt
import math


def driver():

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



driver()