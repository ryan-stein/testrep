import numpy as np
import matplotlib.pyplot as plt
import sympy

# pre-lab: approximating derivatives

def f(x):
    return np.cos(x)

def forward_diff(f, s, h):
    return (f(s + h) - f(s)) / h

def centered_diff(f, s, h):
    return (f(s + h) - f(s - h)) / (2*h)

# True derivative of cos(x) at any x = -sin(x)
def true_derivative(s):
    return -np.sin(s)

# Point of interest
s = np.pi / 2

# Create array of h values
hs = 0.01 * 2.0 ** (-np.arange(0, 10))


#print a table of 
print("   h           Forward Diff    Error FD        Center Diff     Error CD")
print("--------------------------------------------------------------------------")
for h in hs:
    # Compute approximations
    fd_approx = forward_diff(f, s, h)
    cd_approx = centered_diff(f, s, h)
    
    # Compute errors relative to the exact derivative, -1
    fd_error = abs(fd_approx - true_derivative(s))
    cd_error = abs(cd_approx - true_derivative(s))
    
    print(f"{h:10.8f}  {fd_approx:14.8f}  {fd_error:14.8e}  {cd_approx:14.8f}  {cd_error:14.8e}")


#calculating order for each method: p = ln(e_i+1/e_i)/ln(h_i+1/h_i)
# to do this we'll start with our original h_0 of 0.01, and h_1 = 0.005

fd_approx_0 = forward_diff(f, s, 0.01)
fd_error_0 = abs(fd_approx_0 - true_derivative(s))

fd_approx_1 = forward_diff(f, s, 0.005)
fd_error_1 = abs(fd_approx_1 - true_derivative(s))

fd_p = (np.log(fd_error_1/fd_error_0))/(np.log(1/2))

print(f"The order for forward difference is p = {fd_p}")


cd_approx_0 = centered_diff(f, s, 0.01)
cd_error_0 = abs(cd_approx_0 - true_derivative(s))

cd_approx_1 = centered_diff(f, s, 0.005)
cd_error_1 = abs(cd_approx_1 - true_derivative(s))

cd_p = (np.log(cd_error_1/cd_error_0))/(np.log(1/2))

print(f"The order for centered difference is p = {cd_p}")


