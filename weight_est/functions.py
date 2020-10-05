import numpy as np

def phi_x(x):
    return [1/(1+np.exp(x[0])),x[1]**3,np.exp(x[2]+1)]

def phi(x):
    return [x**3,np.exp(x)]

