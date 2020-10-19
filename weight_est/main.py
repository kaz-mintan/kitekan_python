#https://gihyo.jp/dev/serial/01/machine-learning/0011?page=2
import numpy as np
import matplotlib.pyplot as plt

from functions import *
from virtual_sub import out_ydata,out_virtual_data

def fit(phi_y,Y,t):
  if Y.ndim==1:
    PHI_X = np.array([phi_y(x) for x in Y])
  else:
    PHI_X = np.array([phi_y(Y[:,i]) for i in range(Y.shape[1])])

  w_x = np.linalg.solve(np.dot(PHI_X.T, PHI_X), np.dot(PHI_X.T, t))

  return w_x

if __name__ == '__main__':
  for i in range(50):
    dim = 3 
    ans_weight = np.array([0.3,1,0.5])
    ans_x = np.random.rand(dim,50)

    train_time_length = 3
    test_time_length = 10

    train_id = np.sort(np.random.randint(0,49,(train_time_length,)))

    train_factor = ans_x[:,train_id]
    train_face = out_virtual_data(phi_ans,ans_weight,train_factor)

    trained_w = fit(phi_x,train_factor,train_face)

    test_id = np.sort(np.random.randint(0,49,(test_time_length,)))
    test_factor = ans_x[:,test_id]

    estimated_ylist = out_ydata(phi_x,trained_w,test_factor)

    plt.scatter(test_id,estimated_ylist,label="estimated_y")
    plt.scatter(train_id,train_face,label="trained_y")
    plt.plot(range(50),out_ydata(phi_ans,ans_weight,ans_x),label="answer")

    print('tra x,tes x',train_factor,test_factor)
    print('weight compare',ans_weight,trained_w)
    tra=np.array([out_ydata(phi_ans,trained_w,test_factor)])
    ans=np.array([out_ydata(phi_x,ans_weight,test_factor)])
    print('diff sum',np.sum(ans-tra))

    plt.legend()
    plt.show()

  '''
  ff=train_factor
  noize = np.random.rand(*ff.shape)
  print(noize)
  test_factor = ff#+0.05*noize
  '''
