import numpy as np
import matplotlib.pyplot as plt

N = 10
EMO_NUM = 1
FACT_NUM = 10
TOL = 1e-6


def calc_gradient(func, w,X, M):
  h = 1e-4
  gradient = np.zeros_like(M)

  for i in range(M.size):
    store_M = M[:]

    # f(m+h)
    M[i] += h
    f_m_plus_h = func(w,X,M)
    M = store_M[:]
    # f(m-h)
    M[i] -= h
    f_m_minus_h = func(w,X,M)

    gradient[i] = (f_m_plus_h - f_m_minus_h) / (2 * h)

  return gradient

def gradient_descent(func, w,X, M,learning_rate, max_iter):
  for i in range(max_iter):
    X -= (learning_rate * calc_gradient(func, w,X, M))
    print("[{:3d}] X = {}, f(X) = {:.7f}".format(i, X, f(w,X,M)))
      
  return X

if __name__ == "__main__": 

  def f(X,M):
    return M[0]*X[0]**2 + M[1]*X[1]**2
  #f = lambda X: X[0]**2 + X[1]**2
  X = np.array([3.0, 4.0])
  M = np.array([1.0, 1.0])
  gradient_descent(f, w, X, M,learning_rate=.5, max_iter=100)
