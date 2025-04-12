import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import math

def prob1():

    f = lambda s: 1/(1+s**2)

    # Evaluate integral using Trapezoidal and M found in part b.)
    I_trap, X_trap, f_calls_trap = eval_composite_trap(1291, -5, 5, f)

    # Evaluate integral using Simpson's and M found in part b.)
    I_simp, X_simp, f_calls_simp = eval_composite_simpsons(108, -5, 5, f)

    # Evaluate integral using quad and default tolerance
    val1, err1, info1 = quad(f, -5, 5, full_output=True)
    f_calls_quad1 = info1['neval']

    # Evaluate integral using quad and tolerance 10^-4
    val2, err2, info2 = quad(f, -5, 5, epsrel=1e-4, epsabs=1e-4, full_output=True)
    f_calls_quad2 = info2['neval']

    # printing comparisons:

    #comparisons to default tolerance of quad
    print("Comparison to default tolerance of quad: \n")
    print(f"quad evaluated this integral to be {val1}, and used {f_calls_quad1} function calls to do so. \n")
    print(f"Composite Trapezoidal rule evaluated the integral to be {I_trap}, giving it an error of: {abs(I_trap-val1)}")
    print(f"It computed this using {f_calls_trap} function evaluations. \n")
    print(f"Composite Simpson's rule evaluated the integral to be {I_simp}, giving it an error of: {abs(I_simp-val1)}")
    print(f"It computed this using {f_calls_simp} function evaluations. \n")

    #comparisons to quad with tolerance 10^-4
    print("Comparison to quad with tolerance 10^-4: \n")
    print(f"quad evaluated this integral to be {val2}, and used {f_calls_quad2} function calls to do so. \n")
    print(f"Composite Trapezoidal rule evaluated the integral to be {I_trap}, giving it an error of: {abs(I_trap-val2)}")
    print(f"It computed this using {f_calls_trap} function evaluations. \n")
    print(f"Composite Simpson's rule evaluated the integral to be {I_simp}, giving it an error of: {abs(I_simp-val2)}")
    print(f"It computed this using {f_calls_simp} function evaluations. \n")

    return


def prob2():

    # define f(t) to account for 1/t
    def f(t):
        if t == 0.0:
            return 0.0
        else:
            return t*np.cos(1/t)

    # computing integral using simpsons with 5 nodes (M = 4):
    I_simp, X_simp, f_calls_simp = eval_composite_simpsons(4, 0, 1, f)

    print(f"The integral evaluates to approximately: {I_simp}")

    return








def eval_composite_trap(M,a,b,f):
    
    # Track function calls for part c.)
    func_calls = 0

    # Step size
    h = (b - a) / M
    
    # Build the array of subinterval endpoints
    x = np.linspace(a, b, M+1)
    
    # Initialize the integral accumulator
    I_hat = 0.0
    
    # Sum the area of each trapezoid
    for k in range(M):
        xk    = x[k]
        xk1   = x[k+1]
        fmean = 0.5 * (f(xk) + f(xk1)) 
        func_calls += 2 # I could modify this code so that is only calls the function once per iteration, but it's running fine now and I don't want to mess with it
        I_hat += fmean * (xk1 - xk)    
    
    return I_hat, x, func_calls

def eval_composite_simpsons(M,a,b,f):
    
    #Check for even intervsals
    if M % 2 != 0:
        raise ValueError("M must be even for composite Simpson's rule.")
    
    # Track function calls for part c.)
    func_calls = 0

    # Step size
    h = (b - a) / M

    # Generate all subinterval endpoints
    x = np.linspace(a, b, M+1)

    # Initialize accumulator
    I_hat = 0.0

    # perform simposons
    for i in range(0, M, 2):
        x0 = x[i]
        x1 = x[i+1]
        x2 = x[i+2]
        I_hat += (x2 - x0)/6.0 * ( f(x0) + 4.0*f(x1) + f(x2) )
        func_calls += 3
    
    return I_hat, x, func_calls


prob1()
prob2()