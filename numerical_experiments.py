import numpy as np;
import matplotlib.pyplot as plt;
from scipy.special import legendre;
from numpy.polynomial import chebyshev;
from scipy.special import eval_chebyt;
from scipy.special import legendre;
from scipy.integrate import quad;
from scipy.fftpack import dct
rng = np.random.default_rng(seed=27)
from matplotlib.ticker import ScalarFormatter, FuncFormatter


# Global Plotting Style:

plt.rcParams.update({
    "font.family"      : "serif",
    "font.serif"       : ["Times New Roman", "Times", "DejaVu Serif"],  # graceful fallback
    "font.size"        : 13,
    "axes.titlesize"   : 19,
    "axes.labelsize"   : 17,
    "figure.titlesize" : 24,
    "xtick.labelsize"  : 15,
    "ytick.labelsize"  : 15,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

def driver():  # Note I started working with this originally but then changed my strategy but didn't want to delete code in case it could be useful

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

    coef = np.array([3, 1])

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

            sample_reg_avg = np.linalg.norm(coef - sample_a_reg)
            sample_QR_avg = np.linalg.norm(coef - sample_a_QR)

            sample_reg_coef[j] = sample_reg_avg
            sample_QR_coef[j] = sample_QR_avg

        #compute total average of coefficient differences across all samples for current number of data points and add to arrays
        reg_coef_diff[i] = sum(sample_reg_coef)/100
        QR_coef_diff[i] = sum(sample_QR_coef)/100


    #create x values for plotting
    xvals = np.linspace(10, 1000, 100)

    #####################
    # Standard DLS plot
    #####################

    with plt.rc_context(changeFontSize(10)): # make font size bigger
        # figure skeleton
        fig, axs = plt.subplots(
            1, 2,
            figsize=(6.8, 3), 
            sharex=True,
            sharey=False,
            constrained_layout=True
        )

        fig.set_constrained_layout_pads(
            w_pad  = 0.35,  # space to the figure edge (left/right)
            h_pad  = 0.20,  # space to the figure edge (top/bottom)
            wspace = 0.20,  # space between the two columns
            hspace = 0.25   # not used (only one row), but harmless
        )

        fig.suptitle("Conditioning & Coefficient Error for Standard Linear DLS")
        

        lin_cond_helper(axs[0], xvals, reg_conditions,
            y_label="condition number",
            title="Condition Number vs. n")

        lin_cond_helper(axs[1], xvals, reg_coef_diff,
            y_label=r"mean $\| c_{true} - c_{est}\|_2$",
            title="Coefficient Error vs. n")
        
        fig.savefig("lin_cond.pdf", bbox_inches="tight")
        
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
    x_noise, lin_noisy_n = genNoisyFunc(lin_ex, 'n', 'l', -5, 5, 50)
    x_noise, lin_noisy_u = genNoisyFunc(lin_ex, 'u', 'l', -5, 5, 50)

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
    xevals = np.linspace(-5, 5, 1000)

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
    axes[0,0].plot(xevals, DLS_n_eval, color='red', label=f"LS Approx.: $f(x) = {a_norm[1]:.2f}x + {a_norm[0]:.2f}$")
    axes[0,0].legend(loc="upper left", bbox_to_anchor=(0.55,0.4))
    axes[0,0].set_xlabel("x")
    axes[0,0].set_ylabel("y")
    axes[0,0].set_title(f"$f(x) = (x + 3) + \\epsilon | \\epsilon~N(0, 4)$")

    axes[0,1].scatter(x_noise, lin_noisy_u, color='black', marker='o', label='Noisy Data')
    axes[0,1].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = x + 3$")
    axes[0,1].plot(xevals, DLS_u_eval, color='red', label=f"LS Approx.: $f(x) = {a_uni[1]:.2f}x + {a_uni[0]:.2f}$")
    axes[0,1].legend(loc="upper left", bbox_to_anchor=(0.55,0.4))
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

    fig.savefig("noise_comparison_linear.pdf", bbox_inches="tight")

    plt.tight_layout()
    plt.show()

    

    ########################################################
    # Comparing Uniform noise at three levels and 100 points
    ########################################################

    # generate noisy data at each level:
    x_noise, lin_noisy_s = genNoisyFunc(lin_ex, 'u', 's', -10, 10, 100)
    x_noise, lin_noisy_m = genNoisyFunc(lin_ex, 'u', 'm', -10, 10, 100)
    x_noise, lin_noisy_l = genNoisyFunc(lin_ex, 'u', 'l', -10, 10, 100)

    # Perform DLS
    a_s, k_s = LSqr_linfit(x_noise, lin_noisy_s)
    a_m, k_m = LSqr_linfit(x_noise, lin_noisy_m)
    a_l, k_l = LSqr_linfit(x_noise, lin_noisy_l)

    # define functions based off least squares results to allow for smooth error plot:
    DLS_s = lambda x: a_s[0] + a_s[1]*x
    DLS_m = lambda x: a_m[0] + a_m[1]*x
    DLS_l = lambda x: a_l[0] + a_l[1]*x

    # define error functions to allow for smooth error plotting
    err_s = lambda x: abs(lin_ex(x) - DLS_s(x))
    err_m = lambda x: abs(lin_ex(x) - DLS_m(x))
    err_l = lambda x: abs(lin_ex(x) - DLS_l(x))
    

    # Evaluate functions for plotting
    DLS_s_eval = DLS_s(xevals)
    DLS_m_eval = DLS_m(xevals)
    DLS_l_eval = DLS_l(xevals)
    err_s_eval = err_s(xevals)
    err_m_eval = err_m(xevals)
    err_l_eval = err_l(xevals)

    # Plot results:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'Comparison of Uniform Noise at Different Levels')

    axes[0,0].scatter(x_noise, lin_noisy_s, color='black', marker='o', label='Noisy Data')
    axes[0,0].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = x + 3$")
    axes[0,0].plot(xevals, DLS_s_eval, color='red', label=f"LS Approximation: $f(x) = {a_s[1]:.2f}x + {a_s[0]:.2f}$")
    axes[0,0].legend()
    axes[0,0].set_xlabel("x")
    axes[0,0].set_ylabel("y")
    axes[0,0].set_title(f"$f(x) = (x + 3) + \\epsilon | \\epsilon~Uniform(-1,1)$")

    axes[0,1].scatter(x_noise, lin_noisy_m, color='black', marker='o', label='Noisy Data')
    axes[0,1].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = x + 3$")
    axes[0,1].plot(xevals, DLS_m_eval, color='red', label=f"LS Approximation: $f(x) = {a_m[1]:.2f}x + {a_m[0]:.2f}$")
    axes[0,1].legend()
    axes[0,1].set_xlabel("x")
    axes[0,1].set_ylabel("y")
    axes[0,1].set_title(f"$f(x) = (x + 3) + \\epsilon | \\epsilon~Uniform(-2,2)$")

    axes[0,2].scatter(x_noise, lin_noisy_l, color='black', marker='o', label='Noisy Data')
    axes[0,2].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = x + 3$")
    axes[0,2].plot(xevals, DLS_l_eval, color='red', label=f"LS Approximation: $f(x) = {a_l[1]:.2f}x + {a_l[0]:.2f}$")
    axes[0,2].legend()
    axes[0,2].set_xlabel("x")
    axes[0,2].set_ylabel("y")
    axes[0,2].set_title(f"$f(x) = (x + 3) + \\epsilon | \\epsilon~Uniform(-3,3)$")

    axes[1,0].plot(xevals, err_s_eval, color='red')
    axes[1,0].set_xlabel("x")
    axes[1,0].set_ylabel("Absolute Error")
    axes[1,0].set_title(f"Abs Err using 'Small' Uniform Noise")

    axes[1,1].plot(xevals, err_m_eval, color='red')
    axes[1,1].set_xlabel("x")
    axes[1,1].set_ylabel("Absolute Error")
    axes[1,1].set_title(f"Abs Err using 'Medium' Uniform Noise")

    axes[1,2].plot(xevals, err_l_eval, color='red')
    axes[1,2].set_xlabel("x")
    axes[1,2].set_ylabel("Absolute Error")
    axes[1,2].set_title(f"Abs Err using 'Large' Uniform Noise")

    plt.tight_layout()
    plt.show()


    ###################################################################################
    # Comparison of DLS using different numbers of data points and Uniform Medium Noise
    ###################################################################################

    # Generate noisy data:
    x_noise5, lin_noisy5 = genNoisyFunc(lin_ex, 'u', 'm', -10, 10, 5)
    x_noise25, lin_noisy25 = genNoisyFunc(lin_ex, 'u', 'm', -10, 10, 25)
    x_noise125, lin_noisy125 = genNoisyFunc(lin_ex, 'u', 'm', -10, 10, 125)

    # Perform DLS
    a_5, k_5 = LSqr_linfit(x_noise5, lin_noisy5)
    a_25, k_25 = LSqr_linfit(x_noise25, lin_noisy25)
    a_125, k_125= LSqr_linfit(x_noise125, lin_noisy125)

    # define functions based off least squares results to allow for smooth error plot:
    DLS_5 = lambda x: a_5[0] + a_5[1]*x
    DLS_25 = lambda x: a_25[0] + a_25[1]*x
    DLS_125 = lambda x: a_125[0] + a_125[1]*x

    # define error functions to allow for smooth error plotting
    err_5 = lambda x: abs(lin_ex(x) - DLS_5(x))
    err_25 = lambda x: abs(lin_ex(x) - DLS_25(x))
    err_125 = lambda x: abs(lin_ex(x) - DLS_125(x))
    
    # evaluate functions for plotting
    DLS_5_eval = DLS_5(xevals)
    DLS_25_eval = DLS_25(xevals)
    DLS_125_eval = DLS_125(xevals)
    err_5_eval = err_5(xevals)
    err_25_eval = err_25(xevals)
    err_125_eval = err_125(xevals)

    # Plot results:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'Comparison of DLS with Increasing Data Points')

    axes[0,0].scatter(x_noise5, lin_noisy5, color='black', marker='o', label='Noisy Data')
    axes[0,0].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = x + 3$")
    axes[0,0].plot(xevals, DLS_5_eval, color='red', label=f"LS Approximation: $f(x) = {a_5[1]:.2f}x + {a_5[0]:.2f}$")
    axes[0,0].legend()
    axes[0,0].set_xlabel("x")
    axes[0,0].set_ylabel("y")
    axes[0,0].set_title(f"Linear DLS Using 5 Nodes")

    axes[0,1].scatter(x_noise25, lin_noisy25, color='black', marker='o', label='Noisy Data')
    axes[0,1].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = x + 3$")
    axes[0,1].plot(xevals, DLS_25_eval, color='red', label=f"LS Approximation: $f(x) = {a_25[1]:.2f}x + {a_25[0]:.2f}$")
    axes[0,1].legend()
    axes[0,1].set_xlabel("x")
    axes[0,1].set_ylabel("y")
    axes[0,1].set_title(f"Linear DLS Using 25 Nodes")

    axes[0,2].scatter(x_noise125, lin_noisy125, color='black', marker='o', label='Noisy Data')
    axes[0,2].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = x + 3$")
    axes[0,2].plot(xevals, DLS_125_eval, color='red', label=f"LS Approximation: $f(x) = {a_125[1]:.2f}x + {a_125[0]:.2f}$")
    axes[0,2].legend()
    axes[0,2].set_xlabel("x")
    axes[0,2].set_ylabel("y")
    axes[0,2].set_title(f"Linear DLS Using 125 Nodes")

    axes[1,0].plot(xevals, err_5_eval, color='red')
    axes[1,0].set_xlabel("x")
    axes[1,0].set_ylabel("Absolute Error")
    axes[1,0].set_title(f"Absoulte Error of DLS using 5 Nodes")

    axes[1,1].plot(xevals, err_25_eval, color='red')
    axes[1,1].set_xlabel("x")
    axes[1,1].set_ylabel("Absolute Error")
    axes[1,1].set_title(f"Absoulte Error of DLS using 25 Nodes")

    axes[1,2].plot(xevals, err_125_eval, color='red')
    axes[1,2].set_xlabel("x")
    axes[1,2].set_ylabel("Absolute Error")
    axes[1,2].set_title(f"Absoulte Error of DLS using 125 Nodes")

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

    # real coefficient vectors
    coef2 = np.array([1, -2, 1], dtype=float)
    coef5 = np.array([-1, 5, -10, 10, -5, 1], dtype=float)
    coef9 = np.array([-1, 9, -36, 84, -126, 126, -84, 36, -9, 1], dtype=float)

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
            sample_reg_avg2 = np.linalg.norm(coef2 - sample_a_reg2)
            sample_reg_avg5 = np.linalg.norm(coef5 - sample_a_reg5)
            sample_reg_avg9 = np.linalg.norm(coef9 - sample_a_reg9)

            sample_QR_avg2 = np.linalg.norm(coef2 - sample_a_QR2)
            sample_QR_avg5 = np.linalg.norm(coef5 - sample_a_QR5)
            sample_QR_avg9 = np.linalg.norm(coef9 - sample_a_QR9)

            '''
            sample_reg_avg2 = (abs(sample_a_reg2[0] - 1) + abs(sample_a_reg2[1] + 2) + abs(sample_a_reg2[2] - 1))/3
            sample_reg_avg5 = (abs(sample_a_reg5[0] + 1) + abs(sample_a_reg5[1] - 5) + abs(sample_a_reg5[2] + 10) + abs(sample_a_reg5[3] - 10) + abs(sample_a_reg5[4] + 5) + abs(sample_a_reg5[5] - 1))/5
            sample_reg_avg9 = (abs(sample_a_reg9[0] + 1) + abs(sample_a_reg9[1] - 9) + abs(sample_a_reg9[2] + 36) + abs(sample_a_reg9[3] - 84) + abs(sample_a_reg9[4] + 126) + abs(sample_a_reg9[5] - 126) + abs(sample_a_reg9[6] + 84) + abs(sample_a_reg9[7] - 36) + abs(sample_a_reg9[8] + 9) + abs(sample_a_reg9[9] - 1))/9

            sample_QR_avg2 = (abs(sample_a_QR2[0] - 1) + abs(sample_a_QR2[1] - 1) + abs(sample_a_QR2[2] - 2))/3
            sample_QR_avg5 = (abs(sample_a_QR5[0] - 1) + abs(sample_a_QR5[1] - 1) + abs(sample_a_QR5[2] - 2) + abs(sample_a_QR5[3] - 3) + abs(sample_a_QR5[4] - 4) + abs(sample_a_QR5[5] - 5))/5
            sample_QR_avg9 = (abs(sample_a_QR9[0] + 1) + abs(sample_a_QR9[1] - 9) + abs(sample_a_QR9[2] + 36) + abs(sample_a_QR9[3] - 84) + abs(sample_a_QR9[4] + 126) + abs(sample_a_QR9[5] - 126) + abs(sample_a_QR9[6] + 84) + abs(sample_a_QR9[7] - 36) + abs(sample_a_QR9[8] + 9) + abs(sample_a_QR9[9] - 1))/9
            '''

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

    #########################
    # Standard DLS Plot
    #########################

    with plt.rc_context(changeFontSize(6)):
        #figure skeleton
        fig, axs = plt.subplots(
            2, 3,
            figsize=(8.2, 4.3),
            sharex=True,           # x‑axis aligned
            sharey=False,          # each panel free to choose its own y‑scale
            constrained_layout=True
        )

        fig.set_constrained_layout_pads(
            w_pad   = 0.25,   # pad between figure edge and subplots (left/right)
            h_pad   = 0.20,   # pad between figure edge and subplots (top/bottom)
            wspace  = 0.2,   # horizontal space between columns
            hspace  = 0.25    # vertical space between rows
        )

        fig.suptitle("Conditioning & Coefficient Error for Standard Polynomial DLS")

        # Populate the 6 panels
        info_top = [("2", reg_conditions2),
                    ("5", reg_conditions5),
                    ("9", reg_conditions9)]
        info_bot = [("2", reg_coef_diff2),
                    ("5", reg_coef_diff5),
                    ("9", reg_coef_diff9)]
        
        for col, (deg, ydata) in enumerate(info_top):
            poly_cond_helper(axs[0, col], xvals, ydata,
                row_lbl="condition number", col_deg=deg, title_row=True)

        for col, (deg, ydata) in enumerate(info_bot):
            poly_cond_helper(axs[1, col], xvals, ydata,
                    row_lbl=r"mean $\| c_{true} - c_{est}\|_2$", col_deg=deg, title_row=False)
            
        
        fig.savefig("poly_cond.pdf", bbox_inches="tight")    
            
        plt.show()

    #########################
    # QR factored DLS Plot
    #########################

    with plt.rc_context(changeFontSize(6)):
        #figure skeleton
        fig, axs = plt.subplots(
            2, 3,
            figsize=(8.2, 4.3),
            sharex=True,           # x‑axis aligned
            sharey=False,          # each panel free to choose its own y‑scale
            constrained_layout=True
        )

        fig.set_constrained_layout_pads(
            w_pad   = 0.25,   # pad between figure edge and subplots (left/right)
            h_pad   = 0.20,   # pad between figure edge and subplots (top/bottom)
            wspace  = 0.2,   # horizontal space between columns
            hspace  = 0.25    # vertical space between rows
        )

        fig.suptitle("Conditioning & Coefficient Error for QR Factored Polynomial DLS")

        # Populate the 6 panels
        info_top_QR = [("2", QR_conditions2),
                    ("5", QR_conditions5),
                    ("9", QR_conditions9)]
        info_bot_QR = [("2", QR_coef_diff2),
                    ("5", QR_coef_diff5),
                    ("9", QR_coef_diff9)]
        
        for col, (deg, ydata) in enumerate(info_top_QR):
            poly_cond_helper(axs[0, col], xvals, ydata,
                row_lbl="Condition Number", col_deg=deg, title_row=True)
            

        for col, (deg, ydata) in enumerate(info_bot_QR):
            poly_cond_helper(axs[1, col], xvals, ydata,
                    row_lbl=r"mean $\| c_{true} - c_{est}\|_2$", col_deg=deg, title_row=False)
            

        fig.savefig("poly_cond_qr.pdf", bbox_inches="tight") 
        plt.show()

    

    #################################################################################################################################
    # For the second part of our senesitivity analysis, we are going to focus on how the input data affects Polynomial DLS. We'll compare
    # different types of noise, levels of noise, and number of data points for a linear fit DLS model. For all of these we will use
    # the QR factorization model to reduce the impact of conditioning on the sensitvity so we can focus more on the data itself.
    #################################################################################################################################

    # For these examples we'll use a degree 3 polynomial, so that the conditioning doesn't have too much effect on the sensitiviy
    poly3 = lambda x: .1*(x-1)**3

    ############################################################################################
    # Comparing Uniform and Normal homoskedastic noise, both large levels, and using 50 points
    ############################################################################################

    # generate both types of noise
    x_noise, poly_noisy_n = genNoisyFunc(poly3, 'n', 'm', -10, 10, 100)
    x_noise, poly_noisy_u = genNoisyFunc(poly3, 'u', 'm', -10, 10, 100)

    # perform DLS
    a_norm, M_norm, k_norm = LSqr_polyfit(x_noise, poly_noisy_n, 3)
    a_uni, M_uni, k_uni = LSqr_polyfit(x_noise, poly_noisy_u, 3)

    # define functions based off least squares results to allow for smooth error plot:
    DLS_norm = lambda x: a_norm[0] + a_norm[1]*x + a_norm[2]*x**2 + a_norm[3]*x**3
    DLS_uni = lambda x: a_uni[0] + a_uni[1]*x + a_uni[2]*x**2 + a_uni[3]*x**3

    # define error functions to allow for smooth error plotting
    err_norm = lambda x: abs(poly3(x) - DLS_norm(x))
    err_uni = lambda x: abs(poly3(x) - DLS_uni(x))

    # Evaluate functions for plotting
    xevals = np.linspace(-10, 10, 1000)

    feval = poly3(xevals)
    DLS_n_eval = DLS_norm(xevals)
    DLS_u_eval = DLS_uni(xevals)
    err_n_eval = err_norm(xevals)
    err_u_eval = err_uni(xevals)

    
    # Plot results:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Comparison of Normal and Uniform Noise')

    axes[0,0].scatter(x_noise, poly_noisy_n, color='black', marker='o', label='Noisy Data')
    axes[0,0].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = .1(x-1)^3$")
    axes[0,0].plot(xevals, DLS_n_eval, color='red', label=f"LS Approximation")
    axes[0,0].legend()
    axes[0,0].set_xlabel("x")
    axes[0,0].set_ylabel("y")
    axes[0,0].set_title(f"$f(x) = .1(x -1)^3 + \\epsilon | \\epsilon~N(0, 9)$")

    axes[0,1].scatter(x_noise, poly_noisy_u, color='black', marker='o', label='Noisy Data')
    axes[0,1].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = .1(x-1)^3$")
    axes[0,1].plot(xevals, DLS_u_eval, color='red', label=f"LS Approximation")
    axes[0,1].legend()
    axes[0,1].set_xlabel("x")
    axes[0,1].set_ylabel("y")
    axes[0,1].set_title(f"$f(x) = .1(x - 3)^3 + \\epsilon | \\epsilon~Uniform(-3, 3)$")

    axes[1,0].plot(xevals, err_n_eval, color='red')
    axes[1,0].set_xlabel("x")
    axes[1,0].set_ylabel("Absolute Error")
    axes[1,0].set_title(f"Absoulte Error of DLS using Normal Noise")

    axes[1,1].plot(xevals, err_u_eval, color='red')
    axes[1,1].set_xlabel("x")
    axes[1,1].set_ylabel("Absolute Error")
    axes[1,1].set_title(f"Absoulte Error of DLS using Uniform Noise")

    fig.savefig("noise_comparison_poly.pdf", bbox_inches="tight")

    plt.tight_layout()
    plt.show()

    
    ########################################################
    # Comparing Uniform noise at three levels and 50 points
    ########################################################

    # generate noisy data at each level:
    x_noise, poly_noisy_s = genNoisyFunc(poly3, 'u', 's', -5, 5, 50, 2)
    x_noise, poly_noisy_m = genNoisyFunc(poly3, 'u', 'm', -5, 5, 50, 2)
    x_noise, poly_noisy_l = genNoisyFunc(poly3, 'u', 'l', -5, 5, 50, 2)

    # Perform DLS
    a_s, Ms, k_s = LSqr_polyfit(x_noise, poly_noisy_s, 3)
    a_m, Mm, k_m = LSqr_polyfit(x_noise, poly_noisy_m, 3)
    a_l, Ml, k_l = LSqr_polyfit(x_noise, poly_noisy_l, 3)

    # define functions based off least squares results to allow for smooth error plot:
    DLS_s = lambda x: a_s[0] + a_s[1]*x + a_s[2]*x**2 + a_s[3]*x**3
    DLS_m = lambda x: a_m[0] + a_m[1]*x + a_m[2]*x**2 + a_m[3]*x**3
    DLS_l = lambda x: a_l[0] + a_l[1]*x + a_l[2]*x**2 + a_l[3]*x**3

    # define error functions to allow for smooth error plotting
    err_s = lambda x: abs(poly3(x) - DLS_s(x))
    err_m = lambda x: abs(poly3(x) - DLS_m(x))
    err_l = lambda x: abs(poly3(x) - DLS_l(x))
    

    # Evaluate functions for plotting
    DLS_s_eval = DLS_s(xevals)
    DLS_m_eval = DLS_m(xevals)
    DLS_l_eval = DLS_l(xevals)
    err_s_eval = err_s(xevals)
    err_m_eval = err_m(xevals)
    err_l_eval = err_l(xevals)

    # Plot results:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'Comparison of Uniform Noise at Different Levels')

    axes[0,0].scatter(x_noise, poly_noisy_s, color='black', marker='o', label='Noisy Data')
    axes[0,0].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = .1(x-1)^3$")
    axes[0,0].plot(xevals, DLS_s_eval, color='red', label=f"LS Approximation")
    axes[0,0].legend()
    axes[0,0].set_xlabel("x")
    axes[0,0].set_ylabel("y")
    axes[0,0].set_title(f"$f(x) = (x + 3) + \\epsilon | \\epsilon~Uniform(-2,2)$")

    axes[0,1].scatter(x_noise, poly_noisy_m, color='black', marker='o', label='Noisy Data')
    axes[0,1].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = .1(x-1)^3$")
    axes[0,1].plot(xevals, DLS_m_eval, color='red', label=f"LS Approximation")
    axes[0,1].legend()
    axes[0,1].set_xlabel("x")
    axes[0,1].set_ylabel("y")
    axes[0,1].set_title(f"$f(x) = (x + 3) + \\epsilon | \\epsilon~Uniform(-4,4)$")

    axes[0,2].scatter(x_noise, poly_noisy_l, color='black', marker='o', label='Noisy Data')
    axes[0,2].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = .1(x-1)^3$")
    axes[0,2].plot(xevals, DLS_l_eval, color='red', label=f"LS Approximation")
    axes[0,2].legend()
    axes[0,2].set_xlabel("x")
    axes[0,2].set_ylabel("y")
    axes[0,2].set_title(f"$f(x) = (x + 3) + \\epsilon | \\epsilon~Uniform(-6,6)$")

    axes[1,0].plot(xevals, err_s_eval, color='red')
    axes[1,0].set_xlabel("x")
    axes[1,0].set_ylabel("Absolute Error")
    axes[1,0].set_title(f"Abs Err using 'Small' Uniform Noise")

    axes[1,1].plot(xevals, err_m_eval, color='red')
    axes[1,1].set_xlabel("x")
    axes[1,1].set_ylabel("Absolute Error")
    axes[1,1].set_title(f"Abs Err using 'Medium' Uniform Noise")

    axes[1,2].plot(xevals, err_l_eval, color='red')
    axes[1,2].set_xlabel("x")
    axes[1,2].set_ylabel("Absolute Error")
    axes[1,2].set_title(f"Abs Errusing 'Large' Uniform Noise")

    plt.tight_layout()
    plt.show()

    #Plot to show how error extends to extrapolation:
    xevals2 = np.linspace(-50, 50, 1000)
    feval2 = poly3(xevals2)

    DLS_extrapolation = DLS_l(xevals2)
    err_l2_eval = err_l(xevals2)

    #plot of extrapolation and error:
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f'Comparison of Uniform Noise at Different Levels')

    axes[0].scatter(x_noise, poly_noisy_l, color='black', marker='o', label='Noisy Data')
    axes[0].plot(xevals2, feval2, color='blue', label=r"Exact Function: $f(x) = .1(x-1)^3$")
    axes[0].plot(xevals2, DLS_extrapolation, color='red', label=f"LS Extrapolation")
    axes[0].legend()
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"Extrapolated DLS Model")

    axes[1].plot(xevals2, err_l2_eval, color='red')
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Absolute Error")
    axes[1].set_title(f"Absoulte Error of Extrapolated DLS Model")

    plt.tight_layout()
    plt.show()



    ###################################################################################
    # Comparison of DLS using different numbers of data points and Uniform Large Noise
    ###################################################################################

    # Generate noisy data:
    x_noise5, poly_noisy5 = genNoisyFunc(poly3, 'u', 'm', -5, 5, 5, 2)
    x_noise25, poly_noisy25 = genNoisyFunc(poly3, 'u', 'm', -5, 5, 25, 2)
    x_noise125, poly_noisy125 = genNoisyFunc(poly3, 'u', 'm', -5, 5, 125, 2)

    # Perform DLS
    a_5, M5, k_5 = LSqr_polyfit(x_noise5, poly_noisy5, 3)
    a_25, M5, k_25 = LSqr_polyfit(x_noise25, poly_noisy25, 3)
    a_125, M5, k_125= LSqr_polyfit(x_noise125, poly_noisy125, 3)

    # define functions based off least squares results to allow for smooth error plot:
    DLS_5 = lambda x: a_5[0] + a_5[1]*x + a_5[2]*x**2 + a_5[3]*x**3
    DLS_25 = lambda x: a_25[0] + a_25[1]*x + a_25[2]*x**2 + a_25[3]*x**3
    DLS_125 = lambda x: a_125[0] + a_125[1]*x + a_125[2]*x**2 + a_125[3]*x**3

    # define error functions to allow for smooth error plotting
    err_5 = lambda x: abs(poly3(x) - DLS_5(x))
    err_25 = lambda x: abs(poly3(x) - DLS_25(x))
    err_125 = lambda x: abs(poly3(x) - DLS_125(x))
    
    # evaluate functions for plotting
    DLS_5_eval = DLS_5(xevals)
    DLS_25_eval = DLS_25(xevals)
    DLS_125_eval = DLS_125(xevals)
    err_5_eval = err_5(xevals)
    err_25_eval = err_25(xevals)
    err_125_eval = err_125(xevals)

    # Plot results:
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'Comparison of DLS with Increasing Data Points')

    axes[0,0].scatter(x_noise5, poly_noisy5, color='black', marker='o', label='Noisy Data')
    axes[0,0].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = .1(x-1)^3$")
    axes[0,0].plot(xevals, DLS_5_eval, color='red', label=f"LS Approximation")
    axes[0,0].legend()
    axes[0,0].set_xlabel("x")
    axes[0,0].set_ylabel("y")
    axes[0,0].set_title(f"Linear DLS Using 5 Nodes")

    axes[0,1].scatter(x_noise25, poly_noisy25, color='black', marker='o', label='Noisy Data')
    axes[0,1].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = .1(x-1)^3$")
    axes[0,1].plot(xevals, DLS_25_eval, color='red', label=f"LS Approximation")
    axes[0,1].legend()
    axes[0,1].set_xlabel("x")
    axes[0,1].set_ylabel("y")
    axes[0,1].set_title(f"Linear DLS Using 25 Nodes")

    axes[0,2].scatter(x_noise125, poly_noisy125, color='black', marker='o', label='Noisy Data')
    axes[0,2].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = .1(x-1)^3$")
    axes[0,2].plot(xevals, DLS_125_eval, color='red', label=f"LS Approximation")
    axes[0,2].legend()
    axes[0,2].set_xlabel("x")
    axes[0,2].set_ylabel("y")
    axes[0,2].set_title(f"Linear DLS Using 125 Nodes")

    axes[1,0].plot(xevals, err_5_eval, color='red')
    axes[1,0].set_xlabel("x")
    axes[1,0].set_ylabel("Absolute Error")
    axes[1,0].set_title(f"Absoulte Error of DLS using 5 Nodes")

    axes[1,1].plot(xevals, err_25_eval, color='red')
    axes[1,1].set_xlabel("x")
    axes[1,1].set_ylabel("Absolute Error")
    axes[1,1].set_title(f"Absoulte Error of DLS using 25 Nodes")

    axes[1,2].plot(xevals, err_125_eval, color='red')
    axes[1,2].set_xlabel("x")
    axes[1,2].set_ylabel("Absolute Error")
    axes[1,2].set_title(f"Absoulte Error of DLS using 125 Nodes")

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
    c = np.linalg.solve(N,b)

    ################################
    # Solving for condition number:
    ################################

    # Compute the singular values of M
    singular_values = np.linalg.svd(M, compute_uv=False)
    sigma1 = singular_values[0]        # largest singular value
    sigman = singular_values[-1]       # smallest singular value

    # Compute the residual vector r = y - M c
    residual = yi - M.dot(c)

    # Compute norms and condition number of M
    norm_resid = np.linalg.norm(residual)   # ||y - M c||
    norm_c     = np.linalg.norm(c)          # ||c||
    cond_M     = sigma1 / sigman            

    # Compute the sensitivity bound
    cond = cond_M * np.sqrt(1 + (norm_resid**2) / (sigma1**2 * norm_c**2))

    return c, cond



# Linear LS using QR factorization of M as recommended to reduce condtioning
def LSqr_linfit(xi,yi): 
    m = len(xi)
    M = np.ones((m,2))
    M[:,1] = xi

    # Compute QR
    Q,R = np.linalg.qr(M, mode='reduced')
    QT = np.transpose(Q)

    # Solve equivalent system Rx = Q^T y
    c = np.linalg.solve(R,QT@yi)

    ################################
    # Solving for condition number:
    ################################

    # Compute the singular values of M
    singular_values = np.linalg.svd(M, compute_uv=False)
    sigma1 = singular_values[0]        # largest singular value
    sigman = singular_values[-1]       # smallest singular value

    # Compute the residual vector r = y - M c
    residual = yi - M.dot(c)

    # Compute norms and condition number of M
    norm_resid = np.linalg.norm(residual)   # ||y - M c||
    norm_c     = np.linalg.norm(c)          # ||c||
    cond_M     = sigma1 / sigman            

    # Compute the sensitivity bound
    cond = cond_M * np.sqrt(1 + (norm_resid**2) / (sigma1**2 * norm_c**2))

    return c, cond


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
    c = np.linalg.solve(N,b)

    ################################
    # Solving for condition number:
    ################################

    # Compute the singular values of M
    singular_values = np.linalg.svd(M, compute_uv=False)
    sigma1 = singular_values[0]        # largest singular value
    sigman = singular_values[-1]       # smallest singular value

    # Compute the residual vector r = y - M c
    residual = yi - M.dot(c)

    # Compute norms and condition number of M
    norm_resid = np.linalg.norm(residual)   # ||y - M c||
    norm_c     = np.linalg.norm(c)          # ||c||
    cond_M     = sigma1 / sigman            

    # Compute the sensitivity bound
    cond = cond_M * np.sqrt(1 + (norm_resid**2) / (sigma1**2 * norm_c**2))

    return (c,M, cond)

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
    c = np.linalg.solve(R,QT@yi)

    ################################
    # Solving for condition number:
    ################################

    # Compute the singular values of M
    singular_values = np.linalg.svd(M, compute_uv=False)
    sigma1 = singular_values[0]        # largest singular value
    sigman = singular_values[-1]       # smallest singular value

    # Compute the residual vector r = y - M c
    residual = yi - M.dot(c)

    # Compute norms and condition number of M
    norm_resid = np.linalg.norm(residual)   # ||y - M c||
    norm_c     = np.linalg.norm(c)          # ||c||
    cond_M     = sigma1 / sigman            

    # Compute the sensitivity bound
    cond = cond_M * np.sqrt(1 + (norm_resid**2) / (sigma1**2 * norm_c**2))

    return (c,M, cond)


def genNoisyFunc(f, noise_type, noise_level, left_bound, right_bound, num_points, multiplier=1):

    """
    Generate noisy function 
    * f: original function to generate noise from
    * noise_type: desired noise type to be applied to that function ('u' for uniform, 'n' for normal)
    * noise_level: scale of noise to be applied (constant scale: 't' 's', 'm', 'l', or proportional scaling 'p')
    * left_bound: left bound of where you are taking data points from
    * right_bound: right bound of where you are taking data points from
    * num_points: number of "noisy" data points you want to consider (equispaced from left to right bound)
    * multiplier: can be used if standard "levels" are dramatic enough
    
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
            noise = rng.normal(0, 1*multiplier, num_points)
        elif noise_level == 'm':
            noise = rng.normal(0, 2*multiplier, num_points)
        elif noise_level == 'l':
            noise = rng.normal(0, 3*multiplier, num_points)
        elif noise_level == 't':  # Used for stability comparison between QR and Standard normal equations
            noise = rng.normal(0, 1e-8, num_points)
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
            noise = rng.uniform(-1*multiplier,1*multiplier, num_points)
        elif noise_level == 'm':
            noise = rng.uniform(-2*multiplier,2*multiplier, num_points)
        elif noise_level == 'l':
            noise = rng.uniform(-3*multiplier,3*multiplier, num_points)
        elif noise_level == 't':
            noise = rng.uniform(-1.7e-8, 1.7e-8, num_points)
        
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

################################
# Helper functions for plotting:
################################

def poly_cond_helper(ax, x, y, row_lbl, col_deg, title_row):
    """Scatter + cosmetics for one subplot."""
    ax.scatter(x, y, s=18, color="k")                # compact marker
    if title_row:                                    # only top row gets titles
        ax.set_title(f"Degree {col_deg}", pad=6)
    if col_deg == "2":                               # only left column gets y‑labels
        ax.set_ylabel(row_lbl, labelpad=10)
    ax.set_xlabel("Num of Data Points", labelpad=6)
    ax.margins(x=0.02, y=0.05)                       # small padding
    ax.grid(True, ls=":", lw=0.5, alpha=0.55)


def lin_cond_helper(ax, x, y, y_label, title):
    ax.scatter(x, y, s=18, color="k")
    ax.set_title(title, pad=6)
    ax.set_xlabel("Num of Data Points", labelpad=10)
    ax.set_ylabel(y_label,          labelpad=10)
    ax.margins(x=0.02, y=0.05)
    ax.grid(True, ls=":", lw=0.5, alpha=0.55)


def noiseDiffHelper(ax, x, y, *, title="", xlab="x", ylab="y"):
    ax.plot(x, y, color="red", lw=1.5)   # default line (override when needed)
    ax.set_title(title, pad=6)
    ax.set_xlabel(xlab, labelpad=10)
    ax.set_ylabel(ylab, labelpad=10)
    ax.margins(x=0.02, y=0.05)
    ax.grid(True, ls=":", lw=0.5, alpha=0.55)


def changeFontSize(delta):

    change = {
        "font.size"       : plt.rcParams["font.size"]        + delta,
        "axes.titlesize"  : plt.rcParams["axes.titlesize"]   + delta,
        "axes.labelsize"  : plt.rcParams["axes.labelsize"]   + delta,
        "figure.titlesize": plt.rcParams["figure.titlesize"] + delta,
        "xtick.labelsize" : plt.rcParams["xtick.labelsize"]  + delta,
        "ytick.labelsize" : plt.rcParams["ytick.labelsize"]  + delta,
    }

    return change

#driver()
linConditioningTests()
polyConditioningTests()



