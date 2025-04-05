import numpy as np
from scipy import io, integrate, linalg, signal
import matplotlib.pyplot as plt
import math

def prob1():

    # part a.)

    # solving for b1, b2, b3

    M = np.array([[-1/6, 0, 1],[0, 1/6, 0], [1/120, 0, -1/6]])
    a = np.array([0, 1/120, 0])

    b = np.linalg.solve(M, a)

    print(b)

    #defining all of the functions for plotting and error analysis:

    f = lambda x: np.sin(x) #actual function for f
    T = lambda x: x - (x**3)/6 + (x**5)/120  # 6th degree Taylor polynomial of sin(x) centered around 0

    p1 = lambda x: x - (7*x**3)/60 # numerator for part a, (ceofficients solved for in written portion)
    q1 = lambda x: 1 + (x**2)/20  # denominator for part a, (coefficients solved for in written portion)

    r1 = lambda x: p1(x)/q1(x) # Pade approximation function, p(x)/q(x)

    # evaluating func and different approxiamtion methods for plotting and error comparison
    xeval = np.linspace(0,5, 1000)

    # fucntion evalutations
    feval = f(xeval)
    Teval = T(xeval)
    reval = r1(xeval)

    # error evaluations
    err_Taylor = abs(feval - Teval)
    err_Pale = abs(feval - reval)

    # plot of true funtion and approximations
    plt.figure()
    plt.plot(xeval, feval, label="sin(x)", color="black")
    plt.plot(xeval, Teval, label="Taylor Approximation", color="blue")
    plt.plot(xeval, reval, label="Pale Approximation", color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Comparison of sin(x) and Approximations")
    plt.legend()
    plt.show()

    # plot of absolute error for each method:
    plt.figure()
    plt.plot(xeval, err_Taylor, label="Taylor Approximation", color="blue")
    plt.plot(xeval, err_Pale, label="Pale Approximation", color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Comparison of Absolute Error Between Approximation Methods")
    plt.legend()
    plt.show()




    return


prob1()










