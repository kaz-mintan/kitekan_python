import numpy as np
import matplotlib.pyplot as plt

#def gradient_descent(func, X, M,learning_rate, max_iter):
from grad import gradient_descent
from func_new import *

N = 10
EMO_NUM = 1
FACT_NUM = 10

TOL = 1e-6
L_RATE = 0.5
MAX_ITER = 100


def f(w, factor):
  ret =np.zeros(EMO_NUM)
  for emo_num in range(EMO_NUM):
    for fact_num in range(FACT_NUM):
      ret[emo_num] += w[fact_num] * factor[t][fact_num]
  return w[0] + w[1]

def phi(x,m):
  return [1, x, x**2, x**3]

def function(w,factor,mental):
  ret = 0
  for t in range(len(mental)):
    for f_num in range(FACT_NUM):
      i = f_num*4
      ret += w[f_num]*func(inv_norm(i,factor[t][f_num]/100.0),mental[t],i)

def get_PHI(X,M):
  PHI = np.array([phi(x,m) for x,m in zip(X,M)])
  return PHI

def main(X,t,N):

  err = 100
  M = np.zeros(N)
  while err>TOL:
    PHI=get_PHI(X,M)

    w = np.linalg.solve(np.dot(PHI.T, PHI), np.dot(PHI.T, t)) 
    gradient_descent(function,w,X,M,L_RATE,MAX_ITER)

  xlist = np.arange(0, 1, 0.01)
  ylist = [f(w, x) for x in xlist]

  plt.plot(xlist, ylist)
  plt.plot(X, t, 'o')

  plt.show()


if __name__ == "__main__": 

  X = np.array([0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99])
  t = np.array([0.05, 0.87, 0.94, 0.92, 0.54, -0.11, -0.78, -0.89, -0.79, -0.04])
  N = len(X)
  main(X,t,N)
