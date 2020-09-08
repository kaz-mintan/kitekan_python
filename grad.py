import numpy as np
import matplotlib.pyplot as plt

from func_new import *

N = 10
EMO_NUM = 1
FACT_NUM = 10
TOL = 1e-6


def calc_gradient(func, M):
  h = 1e-4
  gradient = np.zeros_like(X)

  for i in range(M.size):
    store_M = M[:]

    # f(x+h)
    M[i] += h
    f_x_plus_h = func(M)
    M = store_M[:]

    # f(x-h)
    M[i] -= h
    f_x_minus_h = func(M)

    # 偏微分
    gradient[i] = (f_x_plus_h - f_x_minus_h) / (2 * h)

  return gradient

def gradient_descent(func, X, learning_rate, max_iter):
  for i in range(max_iter):
    X -= (learning_rate * calc_gradient(func, X))
    print("[{:3d}] X = {}, f(X) = {:.7f}".format(i, X, f(X)))
      
  return X

if __name__ == "__main__": 

  f = lambda X: X[0]**2 + X[1]**2
  X = np.array([3.0, 4.0])
  gradient_descent(f, X, learning_rate=.1, max_iter=100)
