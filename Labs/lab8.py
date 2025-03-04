import numpy as np
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
import math


#prelab:

def lin_interp(x0, x1, f0, f1, alpha):

    return f0 + (alpha - x0) * ((f1-f0)/(x1-x0))


