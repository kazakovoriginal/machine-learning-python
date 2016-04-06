__author__ = 'andreykazakov'

import numpy as np
from sklearn.metrics import roc_auc_score
import random
import sklearn
from sklearn.datasets.samples_generator import make_regression
import pylab
from scipy import stats
import pandas as pn

train_data = pn.read_csv("data.csv",header=None)
train_data_class = train_data[:][0]
train_data_params = train_data
train_data_params.drop(train_data_params.columns[[0]],axis = 1, inplace = True)

# print(train_data_class)
# print(train_data_params)

def gradient_descent(alpha, x, y, ep=0.00001, max_iter=10000):
    converged = False
    iter = 0
    m = x.shape[0] # number of samples

    t0 = 0
    t1 = 0

    # Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0/m * sum([y[i]*x[1][i]*(1-(1/(1+np.exp(-y[i]*(t0*x[1][i]+t1*x[2][i])))))for i in range(m)])
        grad1 = 1.0/m * sum([y[i]*x[2][i]*(1-(1/(1+np.exp(-y[i]*(t0*x[1][i]+t1*x[2][i])))))for i in range(m)])
        # update the theta_temp
        temp0 = t0 + alpha * grad0
        temp1 = t1 + alpha * grad1

        e = abs(((t0-temp0)**2+(t1-temp1)**2)**0.5)

        if abs (e <= ep):
            print 'Converged, iterations: ', iter, '!!!'
            converged = True


        t0 = temp0
        t1 = temp1

        iter += 1  # update iter

        if iter == max_iter:
            print 'Max interactions exceeded!'
            converged = True

    return t0,t1


def gradient_descent_L2(alpha, x, y, ep=0.00001, max_iter=10000):
    converged = False
    iter = 0
    m = x.shape[0] # number of samples

    t0 = 0
    t1 = 0

    # Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = 1.0/m * sum([y[i]*x[1][i]*(1-(1/(1+np.exp(-y[i]*(t0*x[1][i]+t1*x[2][i])))))for i in range(m)])
        grad1 = 1.0/m * sum([y[i]*x[2][i]*(1-(1/(1+np.exp(-y[i]*(t0*x[1][i]+t1*x[2][i])))))for i in range(m)])
        # update the theta_temp
        temp0 = t0 + alpha * grad0 - alpha*(10) * t0
        temp1 = t1 + alpha * grad1 - alpha*(10) * t1

        e = abs(((t0-temp0)**2+(t1-temp1)**2)**0.5)

        if abs (e <= ep):
            print 'Converged, iterations: ', iter, '!!!'
            converged = True


        t0 = temp0
        t1 = temp1



        iter += 1  # update iter

        if iter == max_iter:
            print 'Max interactions exceeded!'
            converged = True

    return t0,t1

alpha = 0.1 # learning rate
ep = 0.00001 # convergence criteria
y = train_data_class
x = train_data_params
w1, w2 = gradient_descent(alpha, x, y, ep, max_iter=10000)
print("e = 0.0001    w1: ",w1,"w2: ",w2)

a = []
for i in range(len(y)):
    a.append(1/(1 + np.exp(-w1*x[1][i]-w2*x[2][i])))
metrica = roc_auc_score(y,a)
print(metrica)

# w1 = 0.287811620472
# w2 = 0.0919833021593

w1_L2, w2_L2 = gradient_descent_L2(alpha, x, y, ep, max_iter=10000)
print("w1_L2: ",w1_L2,"w2_L2: ",w2_L2)
a_L2 = []
for i in range(len(y)):
    a_L2.append(1/(1 + np.exp(-w1_L2*x[1][i]-w2_L2*x[2][i])))
metrica_L2 = roc_auc_score(y,a_L2)
print(metrica_L2)

