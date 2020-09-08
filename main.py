import numpy as np
import matplotlib.pyplot as plt

def f(w, x):
  return w[0] + w[1] * x + w[2] * (x ** 2) + w[3] * (x ** 3)

def phi(x):
  return [1, x, x**2, x**3]


def main(X,t):
  PHI = np.array([phi(x) for x in X])

  w = np.linalg.solve(np.dot(PHI.T, PHI), np.dot(PHI.T, t)) 

  xlist = np.arange(0, 1, 0.01)
  ylist = [f(w, x) for x in xlist]

  plt.plot(xlist, ylist)
  plt.plot(X, t, 'o')

  plt.show()


if __name__ == "__main__": 

  X = np.array([0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99])
  t = np.array([0.05, 0.87, 0.94, 0.92, 0.54, -0.11, -0.78, -0.89, -0.79, -0.04])
  main(X,t)
