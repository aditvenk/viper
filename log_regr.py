import numpy as np
import pandas as pd
import matplotlib as mp
import scipy.optimize as sc

from matplotlib import pylab as py
from mpl_toolkits.mplot3d import Axes3D

def feature_normalization(X):
    (m,n) = np.shape(X)
    for i in range(1,4):
        col = np.reshape(X[0::, i], (m,1))
        mean = np.mean(col)
        stdev = np.std(col)
        col = (col - mean)/stdev
        X[0::, i] = np.reshape(col, (1,m))
    return X

def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

sigmoid = np.vectorize(sigmoid)

def cost_function (theta, X, y):
    (m, n) = np.shape(X)
    h = sigmoid(np.dot(X, np.reshape(theta, (n,1))))
    l1 = np.log(h)
    l2 = np.log(1-h)
    t1 = np.multiply(y, l1)
    t2 = np.multiply(1-y, l2)
    J = (-1/m)*np.sum(t1+t2)
    return J

def gradients (theta, X, y):

    (m, n) = np.shape(X)
    #print(theta)

    h = sigmoid(np.dot(X, np.reshape(theta, (n,1))))

    grad = (1/m)* np.dot(np.reshape(X, (n,m)), (h-y))
    return np.ndarray.flatten(grad)

def minimize (theta, X, y):
    #opt_theta = sc.fmin_bfgs(cost_function, theta, args=(X, y))
    opt_theta = sc.fmin(cost_function, theta, args=(X,y))
    return (opt_theta, cost_function(opt_theta, X, y))

def get_predictions(theta, X):
    (m, n) = np.shape(X)
    h = sigmoid(np.dot(X, np.reshape(theta, (n,1))))
    predict = (h>=0.5)
    return predict

def predict_correct (theta, X, y):
    (m, n) = np.shape(X)
    predict = get_predictions(theta, X)
    wrong = np.count_nonzero(predict-y)
    return (1-wrong/m)

def predict_wrong (theta, X, y):
    return (1 - predict_correct(theta, X, y))
