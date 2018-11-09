#1. MATHS

#calculate area under 6 sigma interval of the normal distribution
from math import exp, pi
from scipy.integrate import quad
import numpy as np
s = 1
m = 0
SIX_SIGMA_Q = quad(lambda x: 1/np.sqrt(2*pi*s**2)*exp(-(x-m)**2/(2*s**2)),-3*s, 3*s)[0]

#2. DISTRIBUTIONS

#IDs for distributions from stats_array
ID_NA = 0 #no distribution available
ID_LG = 2 #lognormal
ID_NM = 3 #normal
ID_TR = 5 #triangular