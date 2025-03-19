import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy.linalg as la
import math

def driver():
    # function you want to approximate
    f = lambda x: math.exp(x)
    # Interval of interest
    a = -1
    b = 1
    # weight function
    w = lambda x: 1.
    # order of approximation
    n = 2
    # Number of points you want to sample in [a,b]
    N = 1000
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
        pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
    
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])

    plt.figure()
    plt.plot(xeval,fex,'ro-', label= 'f(x)')
    plt.plot(xeval,pval,'bs--',label= 'Expansion')
    plt.legend()
    plt.show()

    err = abs(pval-fex)
    plt.semilogy(xeval,err,'ro--',label='error')
    plt.legend()
    plt.show()

    return


def eval_legendre(n, x):

    # Initialize output array
    p = np.zeros(n+1)

    # Base cases:
    if n >= 0:
        p[0] = 1.0         # phi_0(x) = 1
    if n >= 1:
        p[1] = x           # phi_1(x) = x

    # Apply recursion for k = 1..n-1
    for k in range(1, n):
        p[k+1] = ((2*k + 1)*x*p[k] - k*p[k-1]) / (k+1)

    return p


def eval_legendre_expansion(f,a,b,w,n,x):

# This subroutine evaluates the Legendre expansion

# Evaluate all the Legendre polynomials at x that are needed
# by calling your code from prelab
    p = eval_legendre(n, x)
# initialize the sum to 0
    pval = 0.0
    for j in range(0,n+1):
        # make a function handle for evaluating phi_j(x)
        phi_j = lambda x: eval_legendre(n, x)[j]
        # make a function handle for evaluating phi_j^2(x)*w(x)
        phi_j_sq = lambda x: (phi_j(x))**2 * w(x)
        # use the quad function from scipy to evaluate normalizations
        norm_fac,err = quad(phi_j_sq, a, b)
        # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        func_j = lambda x: f(x)*phi_j(x)*w(x) / norm_fac
        # use the quad function from scipy to evaluate coeffs
        aj,err = quad(func_j, a, b)
        # accumulate into pval
        pval = pval+aj*p[j]

    return pval


driver()