import numpy as np

from functions import *

def out_virtual_data(phi_y,weight):
    factor = np.array([[0.1, 0.02, 0.5, 0.2, 0.3, 0.31, 0.32, 0.34, 0.35, 0.36],[0.02, 0.12, 0.19, 0.27, 0.42, 0.51, 0.64, 0.84, 0.88, 0.99],[0.4, 0.92, 0.91, 0.7, 0.4, 0.31, 0.5, 0.4, 0.8, 0.2]])

    virtual_data = [np.dot(weight, phi_y(factor[:,i])) for i in range(factor.shape[1])]

    return virtual_data

if __name__ == '__main__':
    weight = np.array([0.3,0.1,0.5])
    #print(out_virtual_data(phi_x,weight))
