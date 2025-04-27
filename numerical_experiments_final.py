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
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms


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


def conditioning_plots():

    #######################################################################################################################################
    # To analyze the conditioning DLS we'll analyse the condition number of a both Linear and Polynomial fit DLS models using an increasing 
    # number of data points for two functions over the same range. Each function will have randomly selected additive noise from a
    # distribution of constant variance, with independent trials.
    ########################################################################################################################################

    #defining functions
    lin_ex = lambda x: x + 3
    poly9 = lambda x: (x-1)**9

    # store exact actual coefficients
    lin_coef = np.array([3, 1])
    coef9 = np.array([-1, 9, -36, 84, -126, 126, -84, 36, -9, 1], dtype=float)

    # arrays to store condition numbers
    reg_conditions = np.zeros(100)
    reg_conditions9 = np.zeros(100)
    


    # compute condition numbers for linear fit
    for i in range(100): # iterator for different number of data points

        # generate noisy data
        [x_noise, lin_noisy] = genNoisyFunc(lin_ex, 'u', 't', -100, 100, i*10 + 10)

        #perform standard DLS fit
        a_reg, k_reg = LS_linfit(x_noise, lin_noisy)

        # add condition numbers to arrays
        reg_conditions[i] = k_reg

    
    # compute condition numbers for polynomial fit
    for i in range(100): # iterator for different number of data points

        # generate noisy data for each polynomial
        [x_noise9, poly_noisy9] = genNoisyFunc(poly9, 'n', 't', -100, 100, i*10 + 10)

        #perform standard DLS fit
        a_reg9, M_reg9 ,k_reg9 = LS_polyfit(x_noise9, poly_noisy9, 9)

        # add condition numbers to arrays
        reg_conditions9[i] = k_reg9


    # Plot results

    #create x values for plotting
    xvals = np.linspace(10, 1000, 100)

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

        fig.suptitle("Conditioning of DLS")
        

        two_plot_helper(axs[0], xvals, reg_conditions,
            y_label="Condition Number",
            title="Linear Fit Conditioning", padding=30)

        two_plot_helper(axs[1], xvals, reg_conditions9,
            y_label="Condition Number",
            title="Polynomial Fit Conditioning")
        
        plt.show()

    return


def linear_QR_comp():

    #######################################################################################################################################
    # To analyze the stability of solving the least squares problem by the normal equations and QR factorization, we'll compare the average
    # error in the coefficients between the two methods using [10, 1000] data points. For each set of "n" data points, we'll run 100
    # independent samples and average the results to remove the effect of outliers in the coefficient difference.
    ########################################################################################################################################

    lin_ex = lambda x: 7*x + 3

    coef = np.array([3., 7.])

    # arrays to store coefficient differences:
    reg_coef_diff = np.zeros(100)
    QR_coef_diff = np.zeros(100)

    for i in range(100): # iterator for different number of data points

        # Store sampled coefficient differences to be averaged
        sample_reg_coef = np.zeros(100)
        sample_QR_coef = np.zeros(100)

        # compute average coefficient difference by sampling 100 times at current number of data points and computing the average value
        for j in range(100):

            # generate noisy data
            [sample_x_noise, sample_lin_noisy] = genNoisyFunc(lin_ex, 'u', 't', 0, 1e8, i*10 + 10)

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


    # Plot results:

    #create x values for plotting
    xvals = np.linspace(10, 1000, 100)

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

        fig.suptitle("Linear DLS: Stability of Solution Methods")
        

        two_plot_helper(axs[0], xvals, reg_coef_diff,
            y_label=r"mean $\| c_{true} - c_{est}\|_2$",
            title="Normal Equations")

        two_plot_helper(axs[1], xvals, QR_coef_diff,
            y_label=r"mean $\| c_{true} - c_{est}\|_2$",
            title="QR Factorization")
        
        
        plt.show()

    return


def poly_QR_comp():

    #######################################################################################################################################
    # We'll run very simlar experiments for polynomial fit DLS as we did for linear, but we will be testing our coefficient error using 
    # a few different polynomials of increasing degree
    ########################################################################################################################################

    # definging polynomials of varying degrees to test
    poly2 = lambda x: (x-1)**2
    poly5 = lambda x: (x-1)**5
    poly9 = lambda x: (x-1)**9

    # real coefficient vectors
    coef2 = np.array([1, -2, 1], dtype=float)
    coef5 = np.array([-1, 5, -10, 10, -5, 1], dtype=float)
    coef9 = np.array([-1, 9, -36, 84, -126, 126, -84, 36, -9, 1], dtype=float)

    # arrays to store condition numbers and coefficient differences:
    reg_coef_diff2 = np.zeros(100)
    reg_coef_diff5 = np.zeros(100)
    reg_coef_diff9 = np.zeros(100)

    QR_coef_diff2 = np.zeros(100) 
    QR_coef_diff5 = np.zeros(100)
    QR_coef_diff9 = np.zeros(100)

    for i in range(100): # iterator for different number of data points

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
            [sample_x_noise, sample_poly_noisy2] = genNoisyFunc(poly2, 'n', 't', -100, 100, i*10 + 10)
            [sample_x_noise, sample_poly_noisy5] = genNoisyFunc(poly5, 'n', 't', -100, 100, i*10 + 10)
            [sample_x_noise, sample_poly_noisy9] = genNoisyFunc(poly9, 'n', 't', -100, 100, i*10 + 10)

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

    
    # Plot results:

    #create x values for plotting
    xvals = np.linspace(10, 1000, 100)

    with plt.rc_context(changeFontSize(3)):

        fig, axs = plt.subplots(
            2, 3, figsize=(8.2, 4.6),          # a touch taller (4.3 → 4.6)
            sharex=False, constrained_layout=True
        )
        fig.set_constrained_layout_pads(
            w_pad=0.30, h_pad=0.25, wspace=0.25, hspace=0.35
        )

        # ---------------- panel content -------------------------------
        fig.suptitle("Polynomial DLS: Stability of Solution Methods", y=0.97)      

        info_top = [("2", reg_coef_diff2),
                    ("5", reg_coef_diff5),
                    ("9", reg_coef_diff9)]
        info_bot = [("2", QR_coef_diff2),
                    ("5", QR_coef_diff5),
                    ("9", QR_coef_diff9)]

        for c, (deg, ydata) in enumerate(info_top):
            poly_QR_helper(axs[0, c], xvals, ydata,
                        row_lbl="Coef. Err.", col_deg=deg, title_row=False)
            

        for c, (deg, ydata) in enumerate(info_bot):
            poly_QR_helper(axs[1, c], xvals, ydata,
                        row_lbl="Coef. Err.", col_deg=deg, title_row=False)

        # ---------------- shrink panels slightly ----------------------
        for ax in axs.flat:
            pos = ax.get_position()
            shrink = 0.8
            ax.set_position([
                pos.x0 + pos.width*(1-shrink)/2,
                pos.y0 + pos.height*(1-shrink)/2,
                pos.width*shrink,
                pos.height*shrink
            ])

        # ---------------- frames that include tick labels -------------
        fig.canvas.draw()                              # need renderer
        renderer = fig.canvas.get_renderer()

        def row_bbox(row):
            b = None
            for ax in axs[row, :]:
                bb = ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
                b  = bb if b is None else mtransforms.Bbox.union([b, bb])
            return b

        bbox_top = row_bbox(0)    # includes tick labels
        bbox_bot = row_bbox(1)

        
        # Edit bounds of rectangles around rows
        bbox_top = mtransforms.Bbox.from_extents(
            bbox_top.x0 + 0.03,             
            bbox_top.y0 + 0.085,            
            bbox_top.x1 - 0.01,             
            bbox_top.y1 - 0.04)             

        bbox_bot = mtransforms.Bbox.from_extents(
            bbox_bot.x0 + 0.03,
            bbox_bot.y0 + 0.08,
            bbox_bot.x1 - 0.01,
            bbox_bot.y1 - 0.04)

        for bbox in (bbox_top, bbox_bot):
            rect = mpatches.FancyBboxPatch(
                (bbox.x0, bbox.y0), bbox.width, bbox.height,
                boxstyle="round,pad=0.02",
                edgecolor="gray", linewidth=0.7,
                facecolor="none", alpha=0.8,
                transform=fig.transFigure, zorder=0
            )
            fig.add_artist(rect)

        # ---------------- vertical row labels outside frames ----------
        fig.text(bbox_top.x0 - 0.035, (bbox_top.y0 + bbox_top.y1)/2,
                "Normal Equations", rotation=90,
                ha="center", va="center", fontsize=20, weight="bold")
        fig.text(bbox_bot.x0 - 0.035, (bbox_bot.y0 + bbox_bot.y1)/2,
                "QR Factorization", rotation=90,
                ha="center", va="center", fontsize=20, weight="bold")
        

        # ---------- column-wide “Degree n” labels -------------------
        # x-centres of the three columns (figure coords)
        col_centres = [(axs[0, c].get_position().x0 +
                        axs[0, c].get_position().x1)/2 for c in range(3)]
        degree_lbls = ["Degree 2", "Degree 5", "Degree 9"]

        for xc, txt in zip(col_centres, degree_lbls):
            fig.text(xc, bbox_top.y1+0.03, txt,         # just above the frame
                    ha="center", va="bottom",
                    fontsize=20, weight="bold")
                    
            
    plt.show()

    return
     
def poly_noise_graphs():

    poly3 = lambda x: .1*(x-1)**3

    xevals = np.linspace(-5, 15, 1000)
    feval = poly3(xevals)

    ############################################################
    # Comparing Uniform noise at low and high variance 50 points
    ############################################################

    # generate noisy data at each level:
    x_noise, poly_noisy_s = genNoisyFunc(poly3, 'u', 's', 0, 10, 25)
    x_noise, poly_noisy_l = genNoisyFunc(poly3, 'u', 'l', 0, 10, 25)

    # Perform DLS
    a_s, Ms, k_s = LSqr_polyfit(x_noise, poly_noisy_s, 3)
    a_l, Ml, k_l = LSqr_polyfit(x_noise, poly_noisy_l, 3)

    # define functions based off least squares results to allow for smooth error plot:
    DLS_s = lambda x: a_s[0] + a_s[1]*x + a_s[2]*x**2 + a_s[3]*x**3
    DLS_l = lambda x: a_l[0] + a_l[1]*x + a_l[2]*x**2 + a_l[3]*x**3

    # define error functions to allow for smooth error plotting
    err_s = lambda x: abs(poly3(x) - DLS_s(x))
    err_l = lambda x: abs(poly3(x) - DLS_l(x))
    

    # Evaluate functions for plotting
    DLS_s_eval = DLS_s(xevals)
    DLS_l_eval = DLS_l(xevals)
    err_s_eval = err_s(xevals)
    err_l_eval = err_l(xevals)

    # Plot results:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Comparison of Uniform Noise with Different Variance')

    axes[0,0].scatter(x_noise, poly_noisy_s, color='black', marker='o', label='Noisy Data')
    axes[0,0].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = .1(x-1)^3$")
    axes[0,0].plot(xevals, DLS_s_eval, color='red', label=f"LS Approximation")
    axes[0,0].legend()
    axes[0,0].set_xlabel("x")
    axes[0,0].set_ylabel("y")
    axes[0,0].set_title(r"$\epsilon \sim Uniform(-\sqrt{3},\sqrt{3}), \; Var(\epsilon) = 1$")

    axes[0,1].scatter(x_noise, poly_noisy_l, color='black', marker='o', label='Noisy Data')
    axes[0,1].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = .1(x-1)^3$")
    axes[0,1].plot(xevals, DLS_l_eval, color='red', label=f"LS Approximation")
    axes[0,1].legend()
    axes[0,1].set_xlabel("x")
    axes[0,1].set_ylabel("y")
    axes[0,1].set_title(r"$\epsilon \sim Uniform(-3\sqrt{3},3\sqrt{3}), \; Var(\epsilon) = 9$")

    axes[1,0].plot(xevals, err_s_eval, color='red')
    axes[1,0].set_xlabel("x")
    axes[1,0].set_ylabel("Absolute Error")
    axes[1,0].set_title(r"Abs. Err. of Polynomial DLS")

    axes[1,1].plot(xevals, err_l_eval, color='red')
    axes[1,1].set_xlabel("x")
    axes[1,1].set_ylabel("Absolute Error")
    axes[1,1].set_title(r"Abs. Err. of Polynomial DLS")

    plt.tight_layout()
    plt.show()

    



    ###################################################################################
    # Comparison of DLS using different numbers of data points and Uniform Large Noise
    ###################################################################################

    # Generate noisy data:
    x_noise5, poly_noisy5 = genNoisyFunc(poly3, 'u', 'l', 0, 10, 5)
    x_noise125, poly_noisy125 = genNoisyFunc(poly3, 'u', 'l', 0, 10, 125)

    # Perform DLS
    a_5, M5, k_5 = LSqr_polyfit(x_noise5, poly_noisy5, 3)
    a_125, M5, k_125= LSqr_polyfit(x_noise125, poly_noisy125, 3)

    # define functions based off least squares results to allow for smooth error plot:
    DLS_5 = lambda x: a_5[0] + a_5[1]*x + a_5[2]*x**2 + a_5[3]*x**3
    DLS_125 = lambda x: a_125[0] + a_125[1]*x + a_125[2]*x**2 + a_125[3]*x**3

    # define error functions to allow for smooth error plotting
    err_5 = lambda x: abs(poly3(x) - DLS_5(x))
    err_125 = lambda x: abs(poly3(x) - DLS_125(x))
    
    # evaluate functions for plotting
    DLS_5_eval = DLS_5(xevals)
    DLS_125_eval = DLS_125(xevals)

    err_5_eval = err_5(xevals)
    err_125_eval = err_125(xevals)

    # Plot results:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Accuracy of DLS with Different Sample Sizes')

    axes[0,0].scatter(x_noise5, poly_noisy5, color='black', marker='o', label='Noisy Data')
    axes[0,0].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = .1(x-1)^3$")
    axes[0,0].plot(xevals, DLS_5_eval, color='red', label=f"LS Approximation")
    axes[0,0].legend()
    axes[0,0].set_xlabel("x")
    axes[0,0].set_ylabel("y")
    axes[0,0].set_title(f"Polynomial DLS Using 5 Nodes")

    axes[0,1].scatter(x_noise125, poly_noisy125, color='black', marker='o', label='Noisy Data')
    axes[0,1].plot(xevals, feval, color='blue', label=r"Exact Function: $f(x) = .1(x-1)^3$")
    axes[0,1].plot(xevals, DLS_125_eval, color='red', label=f"LS Approximation")
    axes[0,1].legend()
    axes[0,1].set_xlabel("x")
    axes[0,1].set_ylabel("y")
    axes[0,1].set_title(f"Polynomial DLS Using 125 Nodes")

    axes[1,0].plot(xevals, err_5_eval, color='red')
    axes[1,0].set_xlabel("x")
    axes[1,0].set_ylabel("Absolute Error")
    axes[1,0].set_title(f"Absoulte Error of DLS using 5 Nodes")

    axes[1,1].plot(xevals, err_125_eval, color='red')
    axes[1,1].set_xlabel("x")
    axes[1,1].set_ylabel("Absolute Error")
    axes[1,1].set_title(f"Absoulte Error of DLS using 125 Nodes")

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

# Linear WLS with QR factorization
def WLSqr_linfit(xi,yi,v):

    # create design matrix 
    m = len(xi)
    M = np.ones((m,2))
    M[:,1] = xi

    # create weight matrix based on vector of variances v
    W = np.diag(1.0/v)

    # form weighted data
    W_sq_root = np.linalg.cholesky(W) # needed so data points are multiplied by intended weights w_i and not (w_i)^2
    M_w = W_sq_root @ M
    y_w = W_sq_root @ yi

    # Applying the weighting to the data first allows us to solve the minimization problem for the weighted data like ordinary LS

    # Compute QR
    Q,R = np.linalg.qr(M_w, mode='reduced')
    QT = np.transpose(Q)

    # Solve equivalent system Rx = Q^T y
    c = np.linalg.solve(R,QT@y_w)

    ################################
    # Solving for condition number:
    ################################

    # Compute the singular values of M_w
    singular_values = np.linalg.svd(M_w, compute_uv=False)
    sigma1 = singular_values[0]        # largest singular value
    sigman = singular_values[-1]       # smallest singular value

    # Compute the residual vector r = y - M c
    residual = y_w - M_w.dot(c)

    # Compute norms and condition number of M
    norm_resid = np.linalg.norm(residual)   # ||y - M c||
    norm_c     = np.linalg.norm(c)          # ||c||
    cond_M_w     = sigma1 / sigman            

    # Compute the sensitivity bound
    cond = cond_M_w * np.sqrt(1 + (norm_resid**2) / (sigma1**2 * norm_c**2))

    return c, cond


def WLSqr_polyfit(xi,yi,v,k):

    # create design matrix
    m = len(xi)
    M = np.ones((m,k+1))
    for i in np.arange(1,k+1):
        M[:,i] = xi**i

    # create weight matrix based on vector of variances v
    W = np.diag(1.0/v)

    # form weighted data
    W_sq_root = np.linalg.cholesky(W) # needed so data points are multiplied by intended weights w_i and not (w_i)^2
    M_w = W_sq_root @ M
    y_w = W_sq_root @ yi

    # Define matrix and rhs for normal eqs
    Q,R = np.linalg.qr(M_w, mode='reduced')
    QT = np.transpose(Q)

    # Solve normal eqs
    c = np.linalg.solve(R,QT@y_w)

    ################################
    # Solving for condition number:
    ################################

    # Compute the singular values of M
    singular_values = np.linalg.svd(M_w, compute_uv=False)
    sigma1 = singular_values[0]        # largest singular value
    sigman = singular_values[-1]       # smallest singular value

    # Compute the residual vector r = y - M c
    residual = y_w - M_w.dot(c)

    # Compute norms and condition number of M
    norm_resid = np.linalg.norm(residual)   # ||y - M c||
    norm_c     = np.linalg.norm(c)          # ||c||
    cond_M_w     = sigma1 / sigman            

    # Compute the sensitivity bound
    cond = cond_M_w * np.sqrt(1 + (norm_resid**2) / (sigma1**2 * norm_c**2))

    return (c, cond)




def genNoisyFunc(f, noise_type, noise_level, left_bound, right_bound, num_points, multiplier=1):

    """
    Generate noisy function 
    * f: original function to generate noise from
    * noise_type: desired noise type to be applied to that function ('u' for uniform, 'n' for normal)
    * noise_level: scale of noise to be applied (constant scale: 't' 's', 'm', 'l', 'a' (asymetric noise, mean !=0), or proportional scaling 'p')
    * left_bound: left bound of where you are taking data points from
    * right_bound: right bound of where you are taking data points from
    * num_points: number of "noisy" data points you want to consider (equispaced from left to right bound)
    * multiplier: can be used if standard "levels" aren't dramatic enough
    
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
            noise = rng.normal(0, 1*multiplier, num_points)  # var = 1
        elif noise_level == 'm':
            noise = rng.normal(0, 2*multiplier, num_points)  # var = 4
        elif noise_level == 'l':
            noise = rng.normal(0, 3*multiplier, num_points)  # var = 9
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
            noise = rng.uniform(-np.sqrt(3)*multiplier,np.sqrt(3)*multiplier, num_points)  # var = 1
        elif noise_level == 'm':
            noise = rng.uniform(-np.sqrt(12)*multiplier,np.sqrt(12)*multiplier, num_points) # var = 4
        elif noise_level == 'l':
            noise = rng.uniform(-np.sqrt(27)*multiplier,np.sqrt(27)*multiplier, num_points) # var = 9
        elif noise_level == 't':
            noise = rng.uniform(-1.7e-8, 1.7e-8, num_points)
        elif noise_level == 'a': # asymetric noise used to compare QR and normal equations so that conditioning takes over
            noise = rng.uniform(0, 1e-8, num_points)
        
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


def heteroNoise(f, left_bound, right_bound, num_points, constant, prop_to='x'):

    """
    Generate heteroskedastic noisy function 
    * f: original function to generate noise from
    * constant: constant variance is multiplied by, if input == 0, constant will be chosen randomly for each point
    * left_bound: left bound of where you are taking data points from
    * right_bound: right bound of where you are taking data points from
    * num_points: number of "noisy" data points you want to consider (equispaced from left to right bound)
    * prop_to: Determines which variable the variance is proportional to, 'x' or 'y'
    
    Returns: [xvals, yvals]
    - xvals: the xvalues the function was evaluated at
    - yvals: the corresponding y values with random noise added
    - var_list: the known or guessed variance at each data point; used to construct weight matrix
    """

    # create xvals
    xvals = np.linspace(left_bound, right_bound, num_points)

    # evalutate true function values
    fevals = f(xvals)

    # initialize noise vector
    noise = np.zeros(num_points)

    # check what variable variance is proportional to and apply noise respectively:
    if prop_to == 'x':

        # check if the constant applied to the variance is random:
        if constant == 0:

            for i in range(num_points):

                # compute random constant from [0.05, 1]
                c = rng.uniform(0.05, 1)

                # set variance to xvals_i * c, and compute corresponing uniform bounds
                var = abs(xvals[i] * c) # need absolute value b/c variance needs to be positive
                a = np.sqrt(3*var)

                # get noise value based on bounds [-a, a]
                noise[i] = rng.uniform(-1*a, a)

        else: #constant is given

            for i in range(num_points):

                # set variance to xvals_i * constant, and compute corresponing uniform bounds
                var = abs(xvals[i] * constant) # need absolute value b/c variance needs to be positive
                a = np.sqrt(3*var)

                # get noise value based on bounds [-a, a]
                noise[i] = rng.uniform(-1*a, a)

    else: # noise is proportional to y

        # check if the constant applied to the variance is random:
        if constant == 0:

            for i in range(num_points):

                # compute random constant from [0.05, 1]
                c = rng.uniform(0.05, 1)

                # set variance to xvals_i * c, and compute corresponing uniform bounds
                var = abs(fevals[i] * c) # need absolute value b/c variance needs to be positive
                a = np.sqrt(3*var)

                # get noise value based on bounds [-a, a]
                noise[i] = rng.uniform(-1*a, a)

        else: #constant is given

            for i in range(num_points):

                # set variance to xvals_i * constant, and compute corresponing uniform bounds
                var = abs(fevals[i] * constant) # need absolute value b/c variance needs to be positive
                a = np.sqrt(3*var)

                # get noise value based on bounds [-a, a]
                noise[i] = rng.uniform(-1*a, a)

    # add noise values to corresponding function eval values
    yvals = fevals + noise

    return [xvals, yvals]
    






################################
# Helper functions for plotting:
################################

def poly_QR_helper(ax, x, y, row_lbl, col_deg, title_row):
    ax.scatter(x, y, s=12, color="k")                 
    ax.set_ylabel(row_lbl, labelpad=8)
    ax.set_xlabel("Num of Data Points", labelpad=6)
    ax.margins(x=0.02, y=0.05)
    ax.grid(True, ls=":", lw=0.5, alpha=0.55)


def two_plot_helper(ax, x, y, y_label, title, padding=10):
    ax.scatter(x, y, s=18, color="k")
    ax.set_title(title, pad=padding)
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


#conditioning_plots()
#linear_QR_comp()
#poly_QR_comp()
poly_noise_graphs()