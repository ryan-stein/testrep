# get lgwts routine and numpy
import numpy as np


def driver():
   
    #defining f and bounds of integral
    a, b = 0.1, 2.0
    f = lambda x: np.sin(1.0/x)

    # Tolerance and # of nodes
    tol = 1e-3
    M   = 5   

    # Adaptive Composite Trapezoid
    I_trap, X_trap, splits_trap = adaptive_quad(a, b, f, tol, M, eval_composite_trap)

    # Adaptive Composite Simpson
    I_simp, X_simp, splits_simp = adaptive_quad(a, b, f, tol, M, eval_composite_simpsons)

    # Adaptive Gauss-Legendre
    I_gauss, X_gauss, splits_gauss = adaptive_quad(a, b, f, tol, M, eval_gauss_quad)


    #print results
    print("1) Composite Trapezoidal:")
    print(f"   Approx Value = {I_trap}")
    print(f"   # of splits  = {splits_trap}")
    print(f"   # of final nodes in the adapted mesh = {len(X_trap)}\n")

    print("2) Composite Simpson's:")
    print(f"   Approx Value = {I_simp}")
    print(f"   # of splits  = {splits_simp}")
    print(f"   # of final nodes in the adapted mesh = {len(X_simp)}\n")

    print("3) Gauss-Legendre:")
    print(f"   Approx Value = {I_gauss}")
    print(f"   # of splits  = {splits_gauss}")
    print(f"   # of final nodes in the adapted mesh = {len(X_gauss)}\n")

    return



# adaptive quad subroutines
# the following three can be passed
# as the method parameter to the main adaptive_quad() function

def eval_composite_trap(M,a,b,f):
    """
    put code from prelab with same returns as gauss_quad
    you can return None for the weights
    """
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
        I_hat += fmean * (xk1 - xk)    
    
    return I_hat, x, None

def eval_composite_simpsons(M,a,b,f):
    """
    put code from prelab with same returns as gauss_quad
    you can return None for the weights
    """
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
    
    return I_hat, x, None

def eval_gauss_quad(M,a,b,f):
  """
  Non-adaptive numerical integrator for \int_a^b f(x)w(x)dx
  Input:
    M - number of quadrature nodes
    a,b - interval [a,b]
    f - function to integrate
  
  Output:
    I_hat - approx integral
    x - quadrature nodes
    w - quadrature weights

  Currently uses Gauss-Legendre rule
  """
  x,w = lgwt(M,a,b)
  I_hat = np.sum(f(x)*w)
  return I_hat,x,w

def adaptive_quad(a,b,f,tol,M,method):
  """
  Adaptive numerical integrator for \int_a^b f(x)dx
  
  Input:
  a,b - interval [a,b]
  f - function to integrate
  tol - absolute accuracy goal
  M - number of quadrature nodes per bisected interval
  method - function handle for integrating on subinterval
         - eg) eval_gauss_quad, eval_composite_simpsons etc.
  
  Output: I - the approximate integral
          X - final adapted grid nodes
          nsplit - number of interval splits
  """
  # 1/2^50 ~ 1e-15
  maxit = 50
  left_p = np.zeros((maxit,))
  right_p = np.zeros((maxit,))
  s = np.zeros((maxit,1))
  left_p[0] = a; right_p[0] = b;
  # initial approx and grid
  s[0],x,_ = method(M,a,b,f);
  # save grid
  X = []
  X.append(x)
  j = 1;
  I = 0;
  nsplit = 1;
  while j < maxit:
    # get midpoint to split interval into left and right
    c = 0.5*(left_p[j-1]+right_p[j-1]);
    # compute integral on left and right spilt intervals
    s1,x,_ = method(M,left_p[j-1],c,f); X.append(x)
    s2,x,_ = method(M,c,right_p[j-1],f); X.append(x)
    if np.max(np.abs(s1+s2-s[j-1])) > tol:
      left_p[j] = left_p[j-1]
      right_p[j] = 0.5*(left_p[j-1]+right_p[j-1])
      s[j] = s1
      left_p[j-1] = 0.5*(left_p[j-1]+right_p[j-1])
      s[j-1] = s2
      j = j+1
      nsplit = nsplit+1
    else:
      I = I+s1+s2
      j = j-1
      if j == 0:
        j = maxit
  return I,np.unique(X),nsplit


# Note: my code wouldn't compile and I couldn't find a given script for "gauss_legendre" anywhere, so I used chatgpt to make this:
def lgwt(N, a, b):
    """
    Compute the Legendre-Gauss nodes and weights on an interval [a, b].
    Returns arrays x (nodes) and w (weights).
    
    Translated from the original MATLAB code by Greg von Winckel (2009).
    
    Usage: x, w = lgwt(N, a, b)
    """
    # N: number of nodes
    # a, b: interval

    # Initial guesses for roots
    # We'll do this on [-1,1] then shift to [a,b].
    eps = 1e-14
    x = np.cos(np.pi*(np.arange(N)+0.75)/(N+0.5))

    # Legendre-Gauss Vandermonde
    p1 = np.zeros(N)
    p2 = np.zeros(N)
    p3 = np.zeros(N)
    pp = np.zeros(N)

    # for Newton iteration
    for _ in range(100):
        p2[:] = 0.0
        p1[:] = 1.0
        for k in range(1, N+1):
            p3[:] = p2
            p2[:] = p1
            p1[:] = ((2*k - 1)*x*p2 - (k-1)*p3)/k
        # p1 is now P_n(x), p2 is P_{n-1}(x)
        # derivative via the formula: P'_n(x) = n/(1-x^2)*[P_{n-1}(x) - xP_n(x)]
        pp = N*(p2 - x*p1)/(1 - x**2)
        xold = x
        x = xold - p1/pp
        if np.max(np.abs(x - xold)) < eps:
            break

    # compute weights
    w = 2./((1 - x**2)*(pp**2))

    # Shift from [-1,1] to [a,b]
    # transform: y = (a+b)/2 + (b-a)/2 * x
    # Jacobian factor for the weights is (b-a)/2
    x = 0.5*(a+b) + 0.5*(b-a)*x
    w = 0.5*(b-a)*w

    return x, w


driver()