import numpy as np

from functions import *

def out_ydata(phi_y,weight,xdata):
    ydata = [np.dot(weight, phi_y(xdata[:,i])) for i in range(xdata.shape[1])]
    return ydata

def out_virtual_data(phi_y,weight,factor):
    return out_ydata(phi_y,weight,factor)

if __name__ == '__main__':
    weight = np.array([0.3,0.1,0.5])
    #print(out_virtual_data(phi_x,weight))

