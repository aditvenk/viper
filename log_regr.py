import numpy as np
import pandas as pd
import matplotlib as mp
import scipy.optimize as sc

from matplotlib import pylab as py
from mpl_toolkits.mplot3d import Axes3D

def feature_normalization(X):
    """
    Normalize input matrix on a per-column basis.
    Each element e is converted to (e - mean)/stdev, where
    mean and stdev are per-column statistics.

    inputs:  X (input feature matrix)    dim: mxn
    outputs: Normalized X                dim: mxn
    """
    (m,n) = np.shape(X)
    for i in range(0,n):
        col = np.reshape(X[0::, i], (m,1))
        mean = np.mean(col)
        stdev = np.std(col)
        col = (col - mean)/stdev
        X[0::, i] = np.reshape(col, (1,m))
    return X

def sig(x):
    """
    Return sigmoid function of scalar input x

    inputs:  X                          dim: 1x1
    outputs: sigmoid of X               dim: 1x1
    """
    return 1 / (1 + np.exp(-1*x))


def sigmoid(x):
    """
    Return sigmoid function of scalar/matrix input x

    inputs:  X                          dim: mxn
    outputs: Per-element sigmoid        dim: mxn
    """
    s = np.vectorize(sigmoid)
    return(s(x))

def cost_function (theta, X, y):
    """
    Return cost function for logistic regression.
    inputs: theta (parameters)          dim: nx1
            X (input feature matrix)    dim: mxn
            y (output vector)           dim: mx1
    output: cost value                  dim: 1x1
    """

    (m, n) = np.shape(X)
    h = sigmoid(np.dot(X, np.reshape(theta, (n,1))))
    l1 = np.log(h)
    l2 = np.log(1-h)
    t1 = np.multiply(y, l1)
    t2 = np.multiply(1-y, l2)
    J = (-1/m)*np.sum(t1+t2)
    return J

def gradients (theta, X, y):
    """
    Returns partial derivative gradients of cost function for logistic regression
    inputs: theta (parameters)          dim: nx1
            X (input feature matrix)    dim: mxn
            y (output vector)           dim: mx1
    output: partial derivates           dim: nx1
    """

    (m, n) = np.shape(X)
    #print(theta)

    h = sigmoid(np.dot(X, np.reshape(theta, (n,1))))

    grad = (1/m)* np.dot(np.reshape(X, (n,m)), (h-y))
    return np.ndarray.flatten(grad)

def minimize (theta, X, y):
    """
    Optimize cost-function of logistic regression.
    Returns optimum parameters and optimum cost

    inputs: theta (initial parameters)  dim: nx1
            X (input feature matrix)    dim: mxn
            y (output vector)           dim: mx1
    outputs:optimum theta               dim: nx1
            optimum cost                dim: 1x1
    """


    #opt_theta = sc.fmin_bfgs(cost_function, theta, args=(X, y))
    opt_theta = sc.fmin(cost_function, theta, args=(X,y))
    return (opt_theta, cost_function(opt_theta, X, y))

def get_predictions(theta, X, threshold=0.5):
    """
    Get predictions for logistic regression based on sigmoid((theta^T)*x) >= threshold

    inputs: theta (parameters)          dim: nx1
            X (input feature matrix)    dim: mxn
            threshold (output vector)   dim: 1x1
    outputs: binary predictions         dim: mx1
    """

    (m, n) = np.shape(X)
    h = sigmoid(np.dot(X, np.reshape(theta, (n,1))))
    predict = (h>=threshold)
    return predict

def predict_correct (theta, X, y):
    """
    Get predictions for logistic regression based on sigmoid((theta^T)*x) >= threshold
    and compare predictions against provided correct solutions y
    inputs: theta (parameters)          dim: nx1
            X (input feature matrix)    dim: mxn
            y (correct outputs)         dim: mx1
    outputs: Proportion of correct predictions
    """

    (m, n) = np.shape(X)
    predict = get_predictions(theta, X)
    wrong = np.count_nonzero(predict-y)
    return (1-wrong/m)

def predict_wrong (theta, X, y):
    """
    Get predictions for logistic regression based on sigmoid((theta^T)*x) >= threshold
    and compare predictions against provided correct solutions y
    inputs: theta (parameters)          dim: nx1
            X (input feature matrix)    dim: mxn
            y (correct outputs)         dim: mx1
    outputs: Proportion of incorrect predictions
    """

    return (1 - predict_correct(theta, X, y))
