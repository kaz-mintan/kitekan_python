import numpy as np

from functions import *

def add_noiz(array,order):
    ff=np.array(array)
    noize = np.power(10,order)*np.random.rand(*ff.shape)
    noized = ff+0.05*noize
    return noized

def out_ydata(phi_y,weight,xdata):
    ydata = [np.dot(weight, phi_y(xdata[:,i])) for i in range(xdata.shape[1])]
    return ydata

def out_virtual_data(phi_y,weight,factor):
    return out_ydata(phi_y,weight,factor)

if __name__ == '__main__':
    dim = 3
    weight = np.array([0.3,0.1,0.5])
    test_x = np.random.rand(dim,5)
    train_y= out_virtual_data(phi_ans,weight,test_x)
    print('train_y',train_y)
    print('noized',add_noiz(train_y,-1.0))
    #print(out_virtual_data(phi_x,weight))

