import numpy as np
from scipy import io, integrate, linalg, signal
import matplotlib.pyplot as plt
import math


def prob1():


    print("Problem 1")

    # define f
    f = lambda x: 1/(1 + x**2)

    # define f'
    f_prime = lambda x: (-2*x)/((1 + x**2)**2)
    f_prime_left = f_prime(-5)
    f_prime_right = f_prime(5)

    # compute exact f(x) for comparison
    x_exact = np.linspace(-5,5,1001)
    y_exact = f(x_exact)

    
    # compute interpolation for n = 5, 10, 15, and 20 nodes
    for nNodes in range(5, 21, 5):

        #create  N equidistant interpolation nodes on [-5,5]:
        N = nNodes
        h = 10/(N-1)
        xint = np.zeros(N)

        for j in range(1, N+1):
            xint[j-1] = -5 + (j-1)*h

        yint = f(xint)
        y_prime_int = f_prime(xint)

        Neval = 1000    
        xeval = np.linspace(-5,5, Neval)

        # evalaute p(x) using barycentric lagrange polynomials
        yeval_lagrange = np.zeros(Neval)

        for i in range(Neval):
           yeval_lagrange[i] = p_lagrange(xeval[i], xint, yint, N)


        # evaluate p(x) using hermite-lagrange interpolation
        yeval_hermite = np.zeros(Neval)

        for i in range(Neval):
            yeval_hermite[i] = p_hermite(xeval[i], xint, yint, y_prime_int, N)


        # evaluate using natural cubic splines
        (M_n,C_n,D_n) = create_natural_spline(yint,xint,N-1)

        yeval_natural = eval_cubic_spline(xeval,Neval,xint,N-1,M_n,C_n,D_n)



        # evaluate p(x) using clamped cubic splines
        (M_c,C_c,D_c) = create_clamped_spline(yint, xint, N-1, f_prime_left, f_prime_right)

        yeval_clamped = eval_cubic_spline(xeval,Neval,xint,N-1,M_c,C_c,D_c)


        # Plot all four methods in a single 2×2 figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Comparison of Interpolation Methods with N = {N} Nodes')

        # Top-left: Barycentric Lagrange
        axes[0, 0].scatter(xint, yint, color='black', marker='o', label='Interpolation Nodes')
        axes[0, 0].plot(x_exact, y_exact, color='red', label='Exact Function')
        axes[0, 0].plot(xeval, yeval_lagrange, color='blue', label='Barycentric Lagrange')
        axes[0, 0].legend()
        axes[0, 0].set_title('Barycentric Lagrange')

        # Top-right: Hermite-Lagrange
        axes[0, 1].scatter(xint, yint, color='black', marker='o', label='Interpolation Nodes')
        axes[0, 1].plot(x_exact, y_exact, color='red', label='Exact Function')
        axes[0, 1].plot(xeval, yeval_hermite, color='blue', label='Hermite-Lagrange')
        axes[0, 1].legend()
        axes[0, 1].set_title('Hermite-Lagrange')

        # Bottom-left: Natural Spline
        axes[1, 0].scatter(xint, yint, color='black', marker='o', label='Interpolation Nodes')
        axes[1, 0].plot(x_exact, y_exact, color='red', label='Exact Function')
        axes[1, 0].plot(xeval, yeval_natural, color='blue', label='Natural Spline')
        axes[1, 0].legend()
        axes[1, 0].set_title('Natural Cubic Spline')

        # Bottom-right: Clamped Spline
        axes[1, 1].scatter(xint, yint, color='black', marker='o', label='Interpolation Nodes')
        axes[1, 1].plot(x_exact, y_exact, color='red', label='Exact Function')
        axes[1, 1].plot(xeval, yeval_clamped, color='blue', label='Clamped Spline')
        axes[1, 1].legend()
        axes[1, 1].set_title('Clamped Cubic Spline')

        # Layout and show
        plt.tight_layout()
        plt.show()
        

   
    

# defining routines:

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

# cubic spline functions

def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    A[0][0] = 1.0
    for j in range(1,N):
       A[j][j-1] = h[j-1]/6
       A[j][j] = (h[j]+h[j-1])/3 
       A[j][j+1] = h[j]/6
    A[N][N] = 1
    
    M  = np.linalg.solve(A,b)

#  Create the linear coefficients
    C = np.zeros(N)
    D = np.zeros(N)
    for j in range(N):
       C[j] = yint[j]/h[j]-h[j]*M[j]/6
       D[j] = yint[j+1]/h[j]-h[j]*M[j+1]/6
    return(M,C,D)
       
def eval_local_spline(xeval,xi,xip,Mi,Mip,C,D):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Mip = M_{i+1}; Mi = M_i

    hi = xip-xi
    yeval = (Mi*(xip-xeval)**3 +(xeval-xi)**3*Mip)/(6*hi) \
            + C*(xip-xeval) + D*(xeval-xi)
    return yeval 
    
    
def  eval_cubic_spline(xeval,Neval,xint,Nint,M,C,D):
    
    yeval = np.zeros(Neval)
    
    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j]
        btmp= xint[j+1]
        
#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,M[j],M[j+1],C[j],D[j])
#        print('yloc = ', yloc)
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)
                        

def create_clamped_spline(yint, xint, N, fprime_left, fprime_right):

    #    create the right  hand side for the linear system
    b = np.zeros(N+1)
#  vector values
    h = np.zeros(N+1)  
    for i in range(1,N):
       hi = xint[i]-xint[i-1]
       hip = xint[i+1] - xint[i]
       b[i] = (yint[i+1]-yint[i])/hip - (yint[i]-yint[i-1])/hi
       h[i-1] = hi
       h[i] = hip

#  create matrix so you can solve for the M values
# This is made by filling one row at a time 
    A = np.zeros((N+1,N+1))
    for j in range(1,N):
       A[j][j-1] = h[j-1]/6
       A[j][j] = (h[j]+h[j-1])/3 
       A[j][j+1] = h[j]/6


    # clamp left boundary
    h0 = h[0]
    A[0, 0] = -h0/6.0
    A[0, 1] =  h0/6.0
    b[0]     = fprime_left - (yint[1] - yint[0]) / h0

    # clamp right boundary
    hN_1 = h[N-1]
    A[N,   N-1] =  hN_1 / 6.0
    A[N,   N  ] = -hN_1 / 6.0
    b[N]        = fprime_right - (yint[N] - yint[N-1]) / hN_1

    # solve for M
    M = np.linalg.solve(A, b)

    # Build C and D
    C = np.zeros(N)
    D = np.zeros(N)
    for i in range(N):
        C[i] = (yint[i]/h[i]) - (h[i]*M[i])/6.0
        D[i] = (yint[i+1]/h[i]) - (h[i]*M[i+1])/6.0

    return (M, C, D)


prob1()
