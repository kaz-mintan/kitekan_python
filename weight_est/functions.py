import numpy as np

def func_ex(x):
    return 1.0/(1.0+np.exp(x))

def phi_x(x):
    return [func_ex(x[0]),x[1]**3,np.exp(x[2]+1)]

def phi_ans(x):
    return [func_ex(x[0]),x[1]**2,np.exp(x[2])]





def phi(x):
    return [x**3,np.exp(x)]

