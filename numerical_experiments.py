import numpy as np;
import matplotlib.pyplot as plt;
from scipy.special import legendre;
from numpy.polynomial import chebyshev;
from scipy.special import eval_chebyt;
from scipy.special import legendre;
from scipy.integrate import quad;
from scipy.fftpack import dct
rng = np.random.default_rng(seed=27)


############################################################################################################################################
# My general strategy for the experiments is to essentially gather as many different combinations of functions, type of noise, noise-level,
# number of data points, etc. Then we can pick and choose which combinations we want to graph/discuss and modify that part of the code. I'm
# going to include what I think will be generally relevant, but I figured we can ask Corona about it when we meet with him.
############################################################################################################################################



def driver():

    ##############################################################################################################################
    # To test the strengths and weaknesses of each method we will test them on the data sets created from the following functions:
    ##############################################################################################################################

    # linear functions:
    lin_ex1 = lambda x: x
    lin_ex2 = lambda x: .1*x + 3
    lin_ex3 = lambda x: 8*x - 4

    # polynomial functions:
    poly_ex1 = lambda x: x**2  # quadratic
    poly_ex2 = lambda x: .5*(x**3 - (x+3)**2 - x + 10)  # degree-3 polynomial
    poly_ex3 = lambda x: .1*(x+3)*((x-2)**2)*((x+1)**3) # degree-6 polynomial

    # exponential function:
    exp_ex = lambda x: x*np.exp(1-x**2)



    #######################################################################################################################
    # For each function we will take a set number of data points, generate a different type of noise for those data points,
    # and apply the appropriate LS implementatation 
    #######################################################################################################################

    # list used to iterate differing number of data points
    n_points = [10, 50, 100, 1000]

    # initialize lists to store exact evaluations for each function
    lin_ex1_evals = []
    lin_ex2_evals = []
    lin_ex3_evals = []

    poly_ex1_evals = []
    poly_ex2_evals = []
    poly_ex3_evals = []

    exp_ex_evals = []

    # initialize lists to store different levels of noise
    normal_small = []
    normal_med = []
    normal_large = []

    uni_small = []
    uni_med = []
    uni_large = []

    # evaluate each function using various numbers of data points and store the evaluation as a list of lists, and create random noise vectors
    for point in n_points:

        xvals = np.linspace(-3, 3, point) # I've chosen functions that are "interesting" between [-3,3] but this can easily be changed

        #exact function evaliations:
        lin_ex1_evals.append(lin_ex1(xvals))
        lin_ex2_evals.append(lin_ex2(xvals))
        lin_ex3_evals.append(lin_ex3(xvals))

        poly_ex1_evals.append(poly_ex1(xvals))
        poly_ex2_evals.append(poly_ex2(xvals))
        poly_ex3_evals.append(poly_ex3(xvals))

        exp_ex_evals.append(exp_ex(xvals))

        # random noise vectors:
        normal_small.append(rng.normal(0, 1, size=point))
        normal_med.append(rng.normal(0, 2, size=point))
        normal_large.append(rng.normal(0, 3, size=point))

        uni_small.append(rng.uniform(-1, 1, size=point))
        uni_med.append(rng.uniform(-2, 2, size=point))
        uni_large.append(rng.uniform(-3, 3, size=point))

    
    

    
    




        










    return




# Linear LS using QR factorization of M as recommended to reduce condtioning
def LSqr_linfit(xi,yi): 
    m = len(xi)
    M = np.ones((m,2))
    M[:,1] = xi

    # Compute QR
    Q,R = np.linalg.qr(M, mode='reduced')
    QT = np.transpose(Q)

    # Solve equivalent system Rx = Q^T y
    a = np.linalg.solve(R,QT@yi)

    return a

# Polynomial LS again using QR factorization to reduce conditioning of M^TM

def LSqr_polyfit(xi,yi,k):
    m = len(xi)
    M = np.ones((m,k+1))
    for i in np.arange(1,k+1):
        M[:,i] = xi**i

    # Define matrix and rhs for normal eqs
    Q,R = np.linalg.qr(M, mode='reduced')
    QT = np.transpose(Q)

    # Solve normal eqs
    a = np.linalg.solve(R,QT@yi)

    return (a,M)



driver()



