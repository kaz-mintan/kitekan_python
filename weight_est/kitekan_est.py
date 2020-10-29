#https://gihyo.jp/dev/serial/01/machine-learning/0011?page=2

import numpy as np

#from functions import *
#from virtual_sub import *

def fit(phi_y,Y,t):
  if Y.ndim==1:
    PHI_X = np.array([phi_y(x) for x in Y])
  else:
    PHI_X = np.array([phi_y(Y[i,:]) for i in range(Y.shape[0])])

  w_x = np.linalg.solve(np.dot(PHI_X.T, PHI_X), np.dot(PHI_X.T, t))

  return w_x
