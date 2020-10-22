import numpy as np
from kitekan import *

def func_ex(x):
    return 1.0/(1.0+np.exp(x))

def phi_x(x):
    return [func_ex(x[0]),x[1]**3,np.exp(x[2]+1),x[3]+1]

def phi_four(x):
    return [func_ex(x[0]),x[1]**2,np.exp(x[2]),x[3]+1]


def phi(x):
    return [x**3,np.exp(x)]

def func(x,i):
    if i == 0:
        return func_ex(x)
    elif i == 1:
        return x**3
    elif i == 2:
        return np.exp(x+1.0)
    elif i == 3:
        return x+1.0
    elif i == 4:
        return 1.0/(1.0+np.exp(x+1))

def phi_trial(x):
    return [func(x[i],i) for i in range(x.shape[0])]

def phi_quiz(x):
    m = 5.0
    return [quiz_func(inv_norm(i*4,x[i]/100.0),m,i*4) for i in range(x.shape[0])]
