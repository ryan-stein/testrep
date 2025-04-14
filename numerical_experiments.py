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



def linConditioningTests():

    #######################################################################################################################################
    # To analyze the conditioning of linear fit DLS we'll analyse the condition number of a Linear fit DLS model using an increasing amount
    # of data points with random noise added to each. We'll also compare the average error in the coeffiecients produced by DLS.
    ########################################################################################################################################

    lin_ex = lambda x: x + 3

    # arrays to store condition numbers and coefficient differences:
    reg_conditions = np.zeros(100)
    reg_coef_diff = np.zeros(100)
    QR_conditions = np.zeros(100)
    QR_coef_diff = np.zeros(100)

    for i in range(100): # iterator for different number of data points

        # generate noisy data
        [x_noise, lin_noisy] = genNoisyFunc(lin_ex, 'n', 'm', -50, 50, i*10 + 10)

        #perform standard DLS fit
        a_reg, k_reg = LS_linfit(x_noise, lin_noisy)

        #perform DLS QR fit
        a_QR, k_QR = LSqr_linfit(x_noise, lin_noisy)

        # add condition numbers to arrays
        reg_conditions[i] = k_reg
        QR_conditions[i] = k_QR

        # Store sampled coefficient differences to be averaged
        sample_reg_coef = np.zeros(100)
        sample_QR_coef = np.zeros(100)

        # compute average coefficient difference by sampling 100 times at current number of data points and computing the average value
        for j in range(100):

            # generate noisy data
            [sample_x_noise, sample_lin_noisy] = genNoisyFunc(lin_ex, 'n', 'm', -50, 50, i*10 + 10)

            #perform standard DLS fit
            sample_a_reg, _sample_k_reg = LS_linfit(sample_x_noise, sample_lin_noisy)

            #perform DLS QR fit
            sample_a_QR, sample_k_QR = LSqr_linfit(sample_x_noise, sample_lin_noisy)

            sample_reg_avg = (abs(sample_a_reg[0] - 3) + abs(sample_a_reg[1] - 1))/2
            sample_QR_avg = (abs(sample_a_QR[0] -3) + abs(sample_a_QR[1] -1 ))/2

            sample_reg_coef[j] = sample_reg_avg
            sample_QR_coef[j] = sample_QR_avg

        #compute total average of coefficient differences across all samples for current number of data points and add to arrays
        reg_coef_diff[i] = sum(sample_reg_coef)/100
        QR_coef_diff[i] = sum(sample_QR_coef)/100


    #create x values for plotting
    xvals = np.linspace(10, 1000, 100)

    # Standard DLS
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f"Standard Linear fit DLS on Noisy Function: $f(x) = (x + 3) + \\epsilon$")

    axes[0].scatter(xvals, reg_conditions, color='black', marker='o')
    axes[0].set_ylabel('Condition Number')
    axes[0].set_xlabel('Number of Data Points Used')
    axes[0].set_title('Condition Number of DLS by Data Points Used')

    axes[1].scatter(xvals, reg_coef_diff, color='black', marker='o')
    axes[1].set_ylabel('Average Difference Between DLS and Exact Coefficients')
    axes[1].set_xlabel('Number of Data Points Used')
    axes[1].set_title('Average Error in Coefficients Using DLS')

    plt.tight_layout()
    plt.show()

    # QR DLS
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f"QR Factored Linear fit DLS on Noisy Function: $f(x) = (x + 3) + \\epsilon$")

    axes[0].scatter(xvals, QR_conditions, color='black', marker='o')
    axes[0].set_ylabel('Condition Number')
    axes[0].set_xlabel('Number of Data Points Used')
    axes[0].set_title('Condition Number of DLS by Data Points Used')

    axes[1].scatter(xvals, QR_coef_diff, color='black', marker='o')
    axes[1].set_ylabel('Average Difference Between DLS and Exact Coefficients')
    axes[1].set_xlabel('Number of Data Points Used')
    axes[1].set_title('Average Error in Coefficients Using DLS')

    plt.tight_layout()
    plt.show()


    #################################################################################################################################
    # For the second part of our senesitivity analysis, we are going to focus on how the input data affects Linear DLS. We'll compare
    # different types of noise, levels of noise, and number of data points for a linear fit DLS model. For all of these we will use
    # the QR factorization model to reduce the impact of conditioning on the sensitvity so we can focus more on the data itself.
    #################################################################################################################################


    ############################################################################################
    # Comparing Uniform and Normal homoskedastic noise, both medium levels, and using 100 points
    ############################################################################################

    # generate both types of noise
    x_noise, lin_noisy_n = genNoisyFunc(lin_ex, 'n', 'm', -10, 10, 100)
    x_noise, lin_noisy_u = genNoisyFunc(lin_ex, 'u', 'm', -10, 10, 100)

    # perform DLS
    a_norm, k_norm = LSqr_linfit(x_noise, lin_noisy_n)
    a_uni, k_uni = LSqr_linfit(x_noise, lin_noisy_u)

    # define functions based off least squares results to allow for smooth error plot:
    DLS_norm = lambda x: a_norm[0] + a_norm[1]*x
    DLS_uni = lambda x: a_uni[0] + a_uni[1]*x

    # define error functions to allow for smooth error plotting
    err_norm = lambda x: abs(lin_ex(x) - DLS_norm(x))
    err_uni = lambda x: abs(lin_ex(x) - DLS_uni(x))

    # Evaluate functions for plotting
    xevals = np.linspace(-10, 10, 1000)

    feval = lin_ex(xevals)
    DLS_n_eval = DLS_norm(xevals)
    DLS_u_eval = DLS_uni(xevals)
    err_n_eval = err_norm(xevals)
    err_u_eval = err_uni(xevals)

    # Plot results:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Comparison of Normal and Uniform Noise')

    axes[0,0].scatter(x_noise, lin_noisy_n, color='black', marker='o', label='Noisy Data')
    axes[0,0].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = x + 3$")
    axes[0,0].plot(xevals, DLS_n_eval, color='red', label=f"LS Approximation: $f(x) = {a_norm[1]:.2f}x + {a_norm[0]:.2f}$")
    axes[0,0].legend()
    axes[0,0].set_xlabel("x")
    axes[0,0].set_ylabel("y")
    axes[0,0].set_title(f"$f(x) = (x + 3) + \\epsilon | \\epsilon~N(0, 4)$")

    axes[0,1].scatter(x_noise, lin_noisy_n, color='black', marker='o', label='Noisy Data')
    axes[0,1].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = x + 3$")
    axes[0,1].plot(xevals, DLS_u_eval, color='red', label=f"LS Approximation: $f(x) = {a_uni[1]:.2f}x + {a_uni[0]:.2f}$")
    axes[0,1].legend()
    axes[0,1].set_xlabel("x")
    axes[0,1].set_ylabel("y")
    axes[0,1].set_title(f"$f(x) = (x + 3) + \\epsilon | \\epsilon~Uniform(-2, 2)$")

    axes[1,0].plot(xevals, err_n_eval, color='red')
    axes[1,0].set_xlabel("x")
    axes[1,0].set_ylabel("Absolute Error")
    axes[1,0].set_title(f"Absoulte Error of DLS using Normal Noise")

    axes[1,1].plot(xevals, err_u_eval, color='red')
    axes[1,1].set_xlabel("x")
    axes[1,1].set_ylabel("Absolute Error")
    axes[1,1].set_title(f"Absoulte Error of DLS using Uniform Noise")

    plt.tight_layout()
    plt.show()










    return
    

def polyConditioningTests():

    #######################################################################################################################################
    # Even though the condition number for linear DLS is high, the numerical differences between QR and Standard DLS are not very large for 
    # linear data with homoskedastic noise. To better see the effects of a high condition number on the sensitivity of DLS we'll perform a 
    # similar experiment on noisy polynomial data of increasing degrees.
    #######################################################################################################################################

    # definging polynomials of varying degrees to test
    poly2 = lambda x: (x-1)**2
    poly5 = lambda x: (x-1)**5
    poly9 = lambda x: (x-1)**9

    # arrays to store condition numbers and coefficient differences:
    reg_conditions2 = np.zeros(100)
    reg_coef_diff2 = np.zeros(100)
    reg_conditions5 = np.zeros(100)
    reg_coef_diff5 = np.zeros(100)
    reg_conditions9 = np.zeros(100)
    reg_coef_diff9 = np.zeros(100)

    QR_conditions2 = np.zeros(100)
    QR_coef_diff2 = np.zeros(100)
    QR_conditions5 = np.zeros(100)
    QR_coef_diff5 = np.zeros(100)
    QR_conditions9 = np.zeros(100)
    QR_coef_diff9 = np.zeros(100)

    for i in range(100): # iterator for different number of data points

        # generate noisy data for each polynomial
        [x_noise2, poly_noisy2] = genNoisyFunc(poly2, 'n', 'm', -1, 1, i + 10)
        [x_noise5, poly_noisy5] = genNoisyFunc(poly5, 'n', 'm', -1, 1, i + 10)
        [x_noise9, poly_noisy9] = genNoisyFunc(poly9, 'n', 'm', -1, 1, i + 10)

        #perform standard DLS fit
        a_reg2, M_reg2 ,k_reg2 = LS_polyfit(x_noise2, poly_noisy2, 2)
        a_reg5, M_reg5 ,k_reg5 = LS_polyfit(x_noise5, poly_noisy5, 5)
        a_reg9, M_reg9 ,k_reg9 = LS_polyfit(x_noise9, poly_noisy9, 9)

        #perform DLS QR fit
        a_QR2, M_QR2, k_QR2 = LSqr_polyfit(x_noise2, poly_noisy2, 2)
        a_QR5, M_QR5, k_QR5 = LSqr_polyfit(x_noise5, poly_noisy5, 5)
        a_QR9, M_QR9, k_QR9 = LSqr_polyfit(x_noise9, poly_noisy9, 9)

        # add condition numbers to arrays
        reg_conditions2[i] = k_reg2
        reg_conditions5[i] = k_reg5
        reg_conditions9[i] = k_reg9

        QR_conditions2[i] = k_QR2
        QR_conditions5[i] = k_QR5
        QR_conditions9[i] = k_QR9

        # Store sampled coefficient differences to be averaged
        sample_reg_coef2 = np.zeros(100)
        sample_reg_coef5 = np.zeros(100)
        sample_reg_coef9 = np.zeros(100)

        sample_QR_coef2 = np.zeros(100)
        sample_QR_coef5 = np.zeros(100)
        sample_QR_coef9 = np.zeros(100)

        # compute average coefficient difference by sampling 100 times at current number of data points and computing the average value
        for j in range(100):

            # generate noisy data
            [sample_x_noise, sample_poly_noisy2] = genNoisyFunc(poly2, 'n', 'm', -1, 1, i + 10)
            [sample_x_noise, sample_poly_noisy5] = genNoisyFunc(poly5, 'n', 'm', -1, 1, i + 10)
            [sample_x_noise, sample_poly_noisy9] = genNoisyFunc(poly9, 'n', 'm', -1, 1, i + 10)

            #perform standard DLS fit
            sample_a_reg2, sample_reg_M2 ,sample_k_reg2 = LS_polyfit(sample_x_noise, sample_poly_noisy2, 2)
            sample_a_reg5, sample_reg_M5 ,sample_k_reg5 = LS_polyfit(sample_x_noise, sample_poly_noisy5, 5)
            sample_a_reg9, sample_reg_M9 ,sample_k_reg9 = LS_polyfit(sample_x_noise, sample_poly_noisy9, 9)

            #perform DLS QR fit
            sample_a_QR2, sample_QR_M2 ,sample_k_QR2 = LSqr_polyfit(sample_x_noise, sample_poly_noisy2, 2)
            sample_a_QR5, sample_QR_M5 ,sample_k_QR5 = LSqr_polyfit(sample_x_noise, sample_poly_noisy5, 5)
            sample_a_QR9, sample_QR_M9 ,sample_k_QR9 = LSqr_polyfit(sample_x_noise, sample_poly_noisy9, 9)

            # Compute average coefficient differences for each polynomial
            sample_reg_avg2 = (abs(sample_a_reg2[0] - 1) + abs(sample_a_reg2[1] + 2) + abs(sample_a_reg2[2] - 1))/3
            sample_reg_avg5 = (abs(sample_a_reg5[0] + 1) + abs(sample_a_reg5[1] - 5) + abs(sample_a_reg5[2] + 10) + abs(sample_a_reg5[3] - 10) + abs(sample_a_reg5[4] + 5) + abs(sample_a_reg5[5] - 1))/5
            sample_reg_avg9 = (abs(sample_a_reg9[0] + 1) + abs(sample_a_reg9[1] - 9) + abs(sample_a_reg9[2] + 36) + abs(sample_a_reg9[3] - 84) + abs(sample_a_reg9[4] + 126) + abs(sample_a_reg9[5] - 126) + abs(sample_a_reg9[6] + 84) + abs(sample_a_reg9[7] - 36) + abs(sample_a_reg9[8] + 9) + abs(sample_a_reg9[9] - 1))/9

            sample_QR_avg2 = (abs(sample_a_QR2[0] - 1) + abs(sample_a_QR2[1] - 1) + abs(sample_a_QR2[2] - 2))/3
            sample_QR_avg5 = (abs(sample_a_QR5[0] - 1) + abs(sample_a_QR5[1] - 1) + abs(sample_a_QR5[2] - 2) + abs(sample_a_QR5[3] - 3) + abs(sample_a_QR5[4] - 4) + abs(sample_a_QR5[5] - 5))/5
            sample_QR_avg9 = (abs(sample_a_QR9[0] + 1) + abs(sample_a_QR9[1] - 9) + abs(sample_a_QR9[2] + 36) + abs(sample_a_QR9[3] - 84) + abs(sample_a_QR9[4] + 126) + abs(sample_a_QR9[5] - 126) + abs(sample_a_QR9[6] + 84) + abs(sample_a_QR9[7] - 36) + abs(sample_a_QR9[8] + 9) + abs(sample_a_QR9[9] - 1))/9

            # add sampled averages to arrays
            sample_reg_coef2[j] = sample_reg_avg2
            sample_reg_coef5[j] = sample_reg_avg5
            sample_reg_coef9[j] = sample_reg_avg9

            sample_QR_coef2[j] = sample_QR_avg2
            sample_QR_coef5[j] = sample_QR_avg5
            sample_QR_coef9[j] = sample_QR_avg9

        #compute total average of coefficient differences across all samples for current number of data points and add to arrays
        reg_coef_diff2[i] = sum(sample_reg_coef2)/100
        reg_coef_diff5[i] = sum(sample_reg_coef5)/100
        reg_coef_diff9[i] = sum(sample_reg_coef9)/100

        QR_coef_diff2[i] = sum(sample_QR_coef2)/100
        QR_coef_diff5[i] = sum(sample_QR_coef5)/100
        QR_coef_diff9[i] = sum(sample_QR_coef9)/100


    #create x values for plotting
    xvals = np.linspace(10, 110, 100)

    # Standard DLS
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"Conditioning and Error for Standard Polynomial DLS of Varying Degrees")

    axes[0,0].scatter(xvals, reg_conditions2, color='black', marker='o')
    axes[0,0].set_ylabel('Condition Number')
    axes[0,0].set_xlabel('Number of Data Points Used')
    axes[0,0].set_title(f'Condition Number for Degree 2 Polynomial Fit')

    axes[1,0].scatter(xvals, reg_coef_diff2, color='black', marker='o')
    axes[1,0].set_ylabel('Average Error Between DLS and Exact Coefficients')
    axes[1,0].set_xlabel('Number of Data Points Used')
    axes[1,0].set_title('Average Error in Coefficients (Degree 2)')

    axes[0,1].scatter(xvals, reg_conditions5, color='black', marker='o')
    axes[0,1].set_ylabel('Condition Number')
    axes[0,1].set_xlabel('Number of Data Points Used')
    axes[0,1].set_title(f'Condition Number for Degree 5 Polynomial Fit')

    axes[1,1].scatter(xvals, reg_coef_diff5, color='black', marker='o')
    axes[1,1].set_ylabel('Average Error Between DLS and Exact Coefficients')
    axes[1,1].set_xlabel('Number of Data Points Used')
    axes[1,1].set_title('Average Error in Coefficients (Degree 5)')

    axes[0,2].scatter(xvals, reg_conditions9, color='black', marker='o')
    axes[0,2].set_ylabel('Condition Number')
    axes[0,2].set_xlabel('Number of Data Points Used')
    axes[0,2].set_title(f'Condition Number for Degree 9 Polynomial Fit')

    axes[1,2].scatter(xvals, reg_coef_diff9, color='black', marker='o')
    axes[1,2].set_ylabel('Average Error Between DLS and Exact Coefficients')
    axes[1,2].set_xlabel('Number of Data Points Used')
    axes[1,2].set_title('Average Error in Coefficients (Degree 9)')

    plt.tight_layout()
    plt.show()


    # QR factored DLS
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"Conditioning and Error for QR Factored Polynomial DLS of Varying Degrees")

    axes[0,0].scatter(xvals, QR_conditions2, color='black', marker='o')
    axes[0,0].set_ylabel('Condition Number')
    axes[0,0].set_xlabel('Number of Data Points Used')
    axes[0,0].set_title(f'Condition Number for Degree 2 Polynomial Fit')

    axes[1,0].scatter(xvals, QR_coef_diff2, color='black', marker='o')
    axes[1,0].set_ylabel('Average Error Between DLS and Exact Coefficients')
    axes[1,0].set_xlabel('Number of Data Points Used')
    axes[1,0].set_title('Average Error in Coefficients (Degree 2)')

    axes[0,1].scatter(xvals, QR_conditions5, color='black', marker='o')
    axes[0,1].set_ylabel('Condition Number')
    axes[0,1].set_xlabel('Number of Data Points Used')
    axes[0,1].set_title(f'Condition Number for Degree 5 Polynomial Fit')

    axes[1,1].scatter(xvals, QR_coef_diff5, color='black', marker='o')
    axes[1,1].set_ylabel('Average Error Between DLS and Exact Coefficients')
    axes[1,1].set_xlabel('Number of Data Points Used')
    axes[1,1].set_title('Average Error in Coefficients (Degree 5)')

    axes[0,2].scatter(xvals, QR_conditions9, color='black', marker='o')
    axes[0,2].set_ylabel('Condition Number')
    axes[0,2].set_xlabel('Number of Data Points Used')
    axes[0,2].set_title(f'Condition Number for Degree 9 Polynomial Fit')

    axes[1,2].scatter(xvals, QR_coef_diff9, color='black', marker='o')
    axes[1,2].set_ylabel('Average Error Between DLS and Exact Coefficients')
    axes[1,2].set_xlabel('Number of Data Points Used')
    axes[1,2].set_title('Average Error in Coefficients (Degree 9)')

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
    cond = (sig[0]) / (sig[-1])

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
    cond = (sig[0]) / (sig[-1])

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
    cond = (sig[0]) / (sig[-1])

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
    cond = (sig[0]) / (sig[-1])

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





#driver()
linConditioningTests()
#polyConditioningTests()



