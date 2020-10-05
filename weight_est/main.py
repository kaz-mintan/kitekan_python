#https://gihyo.jp/dev/serial/01/machine-learning/0011?page=2
import numpy as np
import matplotlib.pyplot as plt

from functions import *
from virtual_sub import out_virtual_data

def fit(phi_y,Y,t):
  if Y.ndim==1:
    PHI_X = np.array([phi_y(x) for x in Y])
  else:
    PHI_X = np.array([phi_y(Y[:,i]) for i in range(Y.shape[1])])

  w_x = np.linalg.solve(np.dot(PHI_X.T, PHI_X), np.dot(PHI_X.T, t))

  ylist = [np.dot(w_x, phi_y(Y[:,i])) for i in range(Y.shape[1])]

  plt.plot(ylist,label="Y[0]")
  if Y.ndim==1:
    plt.plot(Y, t, 'o',label='t')
  else:
    #for d in range(Y.ndim):
    plt.plot(t, 'x',label='t')
  plt.legend()
  plt.show()

  return w_x

if __name__ == '__main__':
  factor = np.array([[0.01, 0.02, 0.1, 0.2, 0.3, 0.31, 0.32, 0.34, 0.35, 0.36],[0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99],[0.4, 0.92, 0.91, 0.7, 0.4, 0.31, 0.5, 0.4, 0.8, 0.2]])
  Y = np.array([0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99])

  #t = np.array([0.05, 0.87, 0.94, 0.92, 0.54, -0.11, -0.78, -0.79, -0.89, -0.04])
  weight = np.array([0.3,1,0.5])
  t = out_virtual_data(phi_x,weight)

#fit(phi_y,Y)
  print(fit(phi_x,factor,t))
