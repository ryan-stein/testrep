import numpy as np
import math
import numpy.linalg as la
from numpy.linalg import inv
from numpy.linalg import norm



def driver():

    # define f
    f = lambda x: 1/(1 + (10*x)**2)


    #create interpolation nodes:

    N = 2 # note I am going to save the plot for each N but I am just going to edit this line of code for each N = 2, 3, 4 ...
    h = 2/(N-1)
    xint = np.zeros(N)

    for j in range(0, N+1):
       xint[j] = -1 + (j-1)*h

    yint = f(xint)

    
    # via monomial expansion:

    #create vandermonde matrix and inverse:
    V = Vandermonde(xint, N)
    Vinv = inv(V)

    #compute coefficent vector
    coef = Vinv @ yint


    Neval = 1000    
    xeval = np.linspace(-1,1, Neval + 1)
    yeval = eval_monomial(xeval,coef,N,Neval)

    



    



# defining routines

def  eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    
#    print('yeval = ', yeval)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
#        print('yeval[i] = ', yeval[i])
#        print('a[j] = ', a[j])
#        print('i = ', i)
#        print('xeval[i] = ', xeval[i])
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval


def Vandermonde(xint,N):

    V = np.zeros((N+1,N+1))
    
    ''' fill the first column'''
    for j in range(N+1):
       V[j][0] = 1.0

    for i in range(1,N+1):
        for j in range(N+1):
           V[j][i] = xint[j]**i

    return V


def  eval_monomial(xeval,coef,N,Neval):

    yeval = coef[0]*np.ones(Neval+1)
    
#    print('yeval = ', yeval)
    
    for j in range(1,N+1):
      for i in range(Neval+1):
#        print('yeval[i] = ', yeval[i])
#        print('a[j] = ', a[j])
#        print('i = ', i)
#        print('xeval[i] = ', xeval[i])
        yeval[i] = yeval[i] + coef[j]*xeval[i]**j

    return yeval







driver()

