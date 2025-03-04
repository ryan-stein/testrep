import numpy as np
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
import math


def driver():

    # define f and interval
    f = lambda x: 1/(1 + (10*x)**2)
    a = -1
    b = 1

    #create points you want to evaluate at
    Neval = 1000
    xeval = np.linspace(a, b, Neval)

    # number of intervals
    Nint = 10

    # evaluate linear spline
    yeval = eval_lin_spline(xeval, Neval, a, b, f, Nint)

    # evaluate actual f at evaluation points
    f_real = f(xeval)

    # I am going to plot only the interpolation nodes as points so that the first plot is more clear, but I will include the xeval nodes for the error plot
    x_nodes = np.linspace(a, b, Nint+1)
    y_nodes = f(x_nodes)


    # plot both to compare
    plt.figure()
    plt.scatter(x_nodes, y_nodes, color = 'black', marker = 'o', label = 'Interpolation Nodes')
    plt.plot(xeval,f_real, color = 'red', label = 'Exact Function')
    plt.plot(xeval,yeval, color = 'blue', label = 'Linear Spline: (11 interpolation nodes)')
    plt.legend()
    plt.show()

    err = abs(yeval-f_real)
    plt.figure()
    plt.plot(xeval,err,'ro-')
    plt.title('Absolute Error Using Linear Splines')
    plt.show()






#prelab:

def line_eval(x0, x1, f0, f1, alpha):

    return f0 + (alpha - x0) * ((f1-f0)/(x1-x0))\
    

# linear splines method

def eval_lin_spline(xeval,Neval,a,b,f,Nint):
    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)

    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval)

    
    for jint in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''

        atmp = xint[jint]
        btmp= xint[jint+1]
        
    # find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp))
        xloc = xeval[ind]
        n = len(xloc)

        '''temporarily store your info for creating a line in the interval of
        interest'''
        fa = f(atmp)
        fb = f(btmp)

        yloc = np.zeros(len(xloc))
        for kk in range(n):
        #use your line evaluator to evaluate the spline at each location
            yloc[kk] = line_eval(atmp, btmp, fa, fb, xloc[kk])

        # Copy yloc into the final vector
        yeval[ind] = yloc

    return yeval



driver()
           
           


