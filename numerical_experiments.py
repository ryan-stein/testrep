import numpy as np;
import matplotlib.pyplot as plt;
from scipy.special import legendre;
from numpy.polynomial import chebyshev;
from scipy.special import eval_chebyt;
from scipy.special import legendre;
from scipy.integrate import quad;
from scipy.fftpack import dct
rng = np.random.default_rng(seed=27)


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
    # The following tests can be modified to match our exact needs for the project but I've provided some basic comparisons
    # to start with (will ask Corona exactly what he wants). 
    #######################################################################################################################


    # perform all tests using 10, 50, and 250 data points
    for i in range(3):

        # xvals for graphing
        xvals = np.linspace(-3, 3, 1000) #the example functions I chose are "interesting" between -3, and 3 but we could change these bounds

        # exact evaluations of functions for graphing
        lin1_exact = lin_ex1(xvals)
        lin2_exact = lin_ex2(xvals)
        lin3_exact = lin_ex3(xvals)
        poly1_exact = poly_ex1(xvals)
        poly2_exact = poly_ex2(xvals)
        poly3_exact = poly_ex3(xvals)
        exp_exact = exp_ex(xvals)

        #############################################################
        # linear examples with normal, medium noise
        # Shows how the effects of constant scaled noise depend on the
        # function
        #############################################################


        # generate noisy data points
        [x_noise, lin1_norm_m] = genNoisyFunc(lin_ex1, 'n', 'm', -3, 3, 10*(5**i))
        [x_noise, lin2_norm_m] = genNoisyFunc(lin_ex2, 'n', 'm', -3, 3, 10*(5**i))
        [x_noise, lin3_norm_m] = genNoisyFunc(lin_ex3, 'n', 'm', -3, 3, 10*(5**i))

        # Apply linear LS fit to noisy data
        alin1 = LSqr_linfit(x_noise, lin1_norm_m)
        alin2 = LSqr_linfit(x_noise, lin2_norm_m)
        alin3 = LSqr_linfit(x_noise, lin3_norm_m)

        #plot results
        fig, axes = plt.subplots(1, 3, figsize=(12, 8))
        fig.suptitle(f'Comparison of Linear LS using {10*(5**i)} nodes, and Normally Distributed Noise')

        axes[0].scatter(x_noise, lin1_norm_m, color='black', marker='o', label='Noisy Data')
        axes[0].plot(xvals, lin1_exact, color='blue', label='Exact Function')
        axes[0].plot(x_noise, alin1[0]+alin1[1]*x_noise, color='red', label='LS Approximation')
        axes[0].legend()
        axes[0].set_title(r"$f(x) = x$")

        axes[1].scatter(x_noise, lin2_norm_m, color='black', marker='o', label='Noisy Data')
        axes[1].plot(xvals, lin2_exact, color='blue', label='Exact Function')
        axes[1].plot(x_noise, alin2[0]+alin2[1]*x_noise, color='red', label='LS Approximation')
        axes[1].legend()
        axes[1].set_title(r"$f(x) = \frac{1}{10} x + 3$")

        axes[2].scatter(x_noise, lin3_norm_m, color='black', marker='o', label='Noisy Data')
        axes[2].plot(xvals, lin3_exact, color='blue', label='Exact Function')
        axes[2].plot(x_noise, alin3[0]+alin3[1]*x_noise, color='red', label='LS Approximation')
        axes[2].legend()
        axes[2].set_title(r"$f(x) =  8x - 4$")

        plt.tight_layout()
        plt.show()


        ##########################################################################################################
        # Linear LS fit using normally distributed noise scaled by a constant factor, vs. noise proportional to x
        ##########################################################################################################

        # evaluate exact function on [0,10]
        xvals2 = np.linspace(0,10,1000)
        lin1_exact2 = lin_ex1(xvals2)

        # generate noisy data points: (small normal and proportional normal)
        [x_noise, lin1_norm_s] = genNoisyFunc(lin_ex1, 'n', 's', 0, 10, 10*(5**i))
        [x_noise, lin1_norm_p] = genNoisyFunc(lin_ex1, 'n', 'p', 0, 10, 10*(5**i))

        # Apply LS fit
        alin4 = LSqr_linfit(x_noise, lin1_norm_s)
        alin5 = LSqr_linfit(x_noise, lin1_norm_p)

        # plot results
        fig, axes = plt.subplots(1, 2, figsize=(12, 8))
        fig.suptitle(f'Comparison of Noise Scaleling')

        axes[0].scatter(x_noise, lin1_norm_s, color='black', marker='o', label='Noisy Data')
        axes[0].plot(xvals2, lin1_exact2, color='blue', label=r"Exact Function: $f(x) = x$")
        axes[0].plot(x_noise, alin4[0]+alin4[1]*x_noise, color='red', label=f"LS Approximation: $f(x) = {alin4[1]:.2f}x + {alin4[0]:.2f}$")
        axes[0].legend()
        axes[0].set_title("Noise using distribution N~(0,1)")

        axes[1].scatter(x_noise, lin1_norm_p, color='black', marker='o', label='Noisy Data')
        axes[1].plot(xvals2, lin1_exact2, color='blue', label=r"Exact Function: $f(x) = x$")
        axes[1].plot(x_noise, alin5[0]+alin5[1]*x_noise, color='red', label=f"LS Approximation: $f(x) = {alin5[1]:.2f}x + {alin5[0]:.2f}$")
        axes[1].legend()
        axes[1].set_title("Noise using distribution x*N~(0,1)")

        plt.tight_layout()
        plt.show()
        


        #############################################################################
        # Example of underfit LS approximation (linear to approximate quadratic data)
        #############################################################################

        # evaluate exact function on [0,5]
        xvals3 = np.linspace(0,5,1000)
        poly1_exact2 = poly_ex1(xvals3)

        # generate noisy data points
        [x_noise, poly1_norm_s] = genNoisyFunc(poly_ex1, 'n', 's', 0, 5, 10*(5**i))

        # Apply LS linear fit
        alin6 = LSqr_linfit(x_noise, poly1_norm_s)

        # plot results
        plt.figure()
        plt.title("Applying Linear LS to Non-Linear Data")
        plt.scatter(x_noise, poly1_norm_s, color='black', marker='o', label='Noisy Data')
        plt.plot(xvals3, poly1_exact2, color='blue', label=r"Exact Function: $f(x) = x^2$")
        plt.plot(x_noise, alin6[0]+alin6[1]*x_noise, color='red', label=f"LS Approximation: $f(x) = {alin6[1]:.2f}x + {alin6[0]:.2f}$")
        plt.legend()
        plt.tight_layout()
        plt.show()


        #################################################################################################
        # Polynomial LS fit using underfit, correctly fit, and overfit degree (using degree 3 polynomial)
        #################################################################################################

        # generate noisy data
        [x_noise, poly2_norm_s] = genNoisyFunc(poly_ex2, 'n', 's', 0, 5, 10*(5**i))

        # Apply LS polynomial fit with varying degree
        (ap2,M2) = LSqr_polyfit(x_noise,poly2_norm_s,2)
        (ap3,M3) = LSqr_polyfit(x_noise,poly2_norm_s,3)
        (ap6,M6) = LSqr_polyfit(x_noise,poly2_norm_s,4)

        # plot results
        fig, axes = plt.subplots(1, 3, figsize=(12, 8))
        fig.suptitle(f'Choosing Correct fit for Polynomial LS')

        axes[0].scatter(x_noise, poly2_norm_s, color='black', marker='o', label='Noisy Data')
        axes[0].plot(xvals, poly2_exact, color='blue', label=f"Exact Function: $\frac{1}{2}(x^3 - (x+3)^2 - x + 10)$")
        axes[0].plot(x_noise, M2@ap2, color='red', label='Quadratic LS Approximation')
        axes[0].legend()
        axes[0].set_title(r"$Under$")

        axes[1].scatter(x_noise, lin2_norm_m, color='black', marker='o', label='Noisy Data')
        axes[1].plot(xvals, lin2_exact, color='blue', label='Exact Function')
        axes[1].plot(x_noise, alin2[0]+alin2[1]*x_noise, color='red', label='LS Approximation')
        axes[1].legend()
        axes[1].set_title(r"$f(x) = \frac{1}{10} x + 3$")

        axes[2].scatter(x_noise, lin3_norm_m, color='black', marker='o', label='Noisy Data')
        axes[2].plot(xvals, lin3_exact, color='blue', label='Exact Function')
        axes[2].plot(x_noise, alin3[0]+alin3[1]*x_noise, color='red', label='LS Approximation')
        axes[2].legend()
        axes[2].set_title(r"$f(x) =  8x - 4$")

        plt.tight_layout()
        plt.show()



    
    




        










    return

# Linear LS solving normal equations directly
def LS_linfit(xi,yi):
    m = len(xi)
    M = np.ones((m,2))
    M[:,1] = xi

    # Define matrix and rhs for normal eqs
    N = np.transpose(M)@M
    b = np.transpose(M)@yi

    # Solve normal eqs
    a = np.linalg.solve(N,b)

    # Solving for condition number:

    #perform SVD of N (M^T*M)
    U, sig, V = np.linalg.svd(N)

    # take ratio of largest and smallest singular values
    cond = (sig[0,0]) / (sig[1,1])

    return a, cond



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

    # Solving for condition number:

    #perform SVD of R
    U, sig, V = np.linalg.svd(R)

    # take ratio of largest and smallest singular values
    cond = (sig[0,0]) / (sig[1,1])

    return a, cond


# Polynomial least squares solving the normal equations directly
def LS_polyfit(xi,yi,k):
    m = len(xi)
    M = np.ones((m,k+1))
    for i in np.arange(1,k+1):
        M[:,i] = xi**i

    # Define matrix and rhs for normal eqs
    N = np.transpose(M)@M
    b = np.transpose(M)@yi

    # Solve normal eqs
    a = np.linalg.solve(N,b)

    # Solving for condition number:

    #perform SVD of N (M^T*M)
    U, sig, V = np.linalg.svd(N)

    # take ratio of largest and smallest singular values
    cond = (sig[0,0]) / (sig[k-1,k-1])

    return (a,M, cond)

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

    # Solving for condition number:

    #perform SVD of N (M^T*M)
    U, sig, V = np.linalg.svd(R)

    # take ratio of largest and smallest singular values
    cond = (sig[0,0]) / (sig[k-1,k-1])

    return (a,M, cond)


def genNoisyFunc(f, noise_type, noise_level, left_bound, right_bound, num_points):

    """
    Generate noisy function 
    * f: original function to generate noise from
    * noise_type: desired noise type to be applied to that function ('u' for uniform, 'n' for normal)
    * noise_level: scale of noise to be applied (constant scale: 's', 'm', 'l', or proportional scaling 'p')
    * left_bound: left bound of where you are taking data points from
    * right_bound: right bound of where you are taking data points from
    * num_points: number of "noisy" data points you want to consider (equispaced from left to right bound)
    
    Returns: [xvals, yvals]
    - xvals: the xvalues the function was evaluated at
    - yvals: the corresponding y values with random noise added
    """

    # create xvals
    xvals = np.linspace(left_bound, right_bound, num_points)

    # evalutate true function values
    fevals = f(xvals)

    # create array of random noise values based on chosen type and level
    if noise_type == 'n': #normal noise

        #check for noise level (changing variance for 's', 'm', 'l')
        if noise_level == 's':
            noise = rng.normal(0, 1, num_points)
        elif noise_level == 'm':
            noise = rng.normal(0, 2, num_points)
        elif noise_level == 'l':
            noise = rng.normal(0, 3, num_points)
        elif noise_level == 'p':
            # special case where the noise is scaled proportionally to the x value: start with 's' noise level then scale\
            noise = rng.normal(0, 1, num_points)
            noise *= xvals
        else:
            print("Invalid noise_level argument")
            return      
    elif noise_type == 'u': #uniform noise

        #check for noise level (changing variance for 's', 'm', 'l')
        if noise_level == 's':
            noise = rng.uniform(-1,1, num_points)
        elif noise_level == 'm':
            noise = rng.uniform(-2,2, num_points)
        elif noise_level == 'l':
            noise = rng.uniform(-3,3, num_points)
        elif noise_level == 'p':
            # special case where the noise is scaled proportionally to the x value: start with 's' noise level then scale\
            noise = rng.uniform(-1,1)
            noise *= xvals
        else:
            print("Invalid noise_level argument")
            return     
    else:
        print("Invalid noise_type argument")
        return
    
    # add noise values to corresponding function eval values
    yvals = fevals + noise

    return [xvals, yvals]





driver()



