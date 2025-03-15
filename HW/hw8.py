import numpy as np
from scipy import io, integrate, linalg, signal
import matplotlib.pyplot as plt
import math


def prob1():


    print("Problem 1")

    # define f
    f = lambda x: 1/(1 + x**2)

    # compute exact f(x) for comparison
    x_exact = np.linspace(-1,1,1001)
    y_exact = f(x_exact)

    
    # compute interpolation for n = 5, 10, 15, and 20 nodes
    for i in range(5, 21, 5):

        #create  N equidistant interpolation nodes on [-5,5]:
        N = i 
        h = 10/(N-1)
        xint = np.zeros(N)

        for j in range(1, N+1):
            xint[j-1] = -5 + (j-1)*h

        yint = f(xint)

        Neval = 1000    
        xeval = np.linspace(-1,1, Neval)

        # evalaute p(x) using barycentric lagrange polynomials
        yeval_lagrange = np.zeros(Neval)

        for i in range(Neval):
           yeval_lagrange[i] = p_lagrange(xeval[i], xint, yint, N)


        # evaluate p(x) using hermite-lagrange interpolation

        

    
def prob2():
   
    print("Problem 2: Lagrange Interpolation")

    # define f
    f = lambda x: 1/(1 + x**2)

    # define f'
    f_prime = lambda x: (-2*x)/((1 + x**2)**2)

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
        y_prime_int = f_prime(xint)

        
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



def prob3():
   
    print("Problem 3: Chebyshev Nodes")

    # define f
    f = lambda x: 1/(1 + (10*x)**2)

    # compute exact f(x) for comparison
    x_exact = np.linspace(-1,1,1001)
    y_exact = f(x_exact)

    
    # I am going to compute and plot the function interpolation for N = 2, 3, ... , 20
    for i in range(2, 21):

        #create interpolation nodes:
        N = i 
        xint = np.zeros(N)

        for j in range(1, N+1):
            xint[j-1] = np.cos(((2*j - 1)*np.pi)/(2*N)) #modified code for chebyshev nodes

        yint = f(xint)

        
        Neval = 1000    
        xeval = np.linspace(-1,1, Neval)

        # evalaute p(x) using lagrange polynomials
        yeval = np.zeros(Neval)

        for i in range(Neval):
           yeval[i] = p(xeval[i], xint, yint, N)


        #create plot:

        plt.figure()
        plt.scatter(xint, yint, color = 'black', marker = 'o', label = 'Interpolation Nodes (Chebyshev)')
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


# Barycentric-Lagrange functions:

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


def p_lagrange(x, xint, yint, N):

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


# Hermite Lagrange funtions:

def L_j(x, j, xint, N):

    #initialize numerator and denominator
    numerator = 1
    denominator = 1

    # loop to find products
    for i in range(N):

        if i != j: 
            
            numerator *= (x - xint[i])
            denominator *= (xint[j] - xint[i])
    
    return numerator/denominator


def eta_j(j, xint, N):

    #initialize sum
    sum = 0

    for i in range(N):

        if i != j:

            sum += 1/(xint[j] - xint[i])

    return sum

def H_j(x, j, xint, N):

    eta = eta_j(j, xint, N)
    L = L_j(x, j, xint, N)

    return (1 - 2*eta * (x - xint[j])) * (L)**2

def K_j(x, j, xint, N):

    L = L_j(x, j, xint, N)

    return (x - xint[j]) * (L)**2


def p_hermite(x, xint, yint, y_prime_int, N):

    sum = 0

    for j in range(N):

        f = yint[j]
        fprime = y_prime_int[j]

        H = H_j(x, j, xint, N)
        K = K_j(x, j, xint, N)

        sum += f*H + fprime*K

    return sum








# prob1()
# prob2()
# prob3()
