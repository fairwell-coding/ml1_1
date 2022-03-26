import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

def pearson_coefficient(x, y):
    """
    :param x: Variable_1 (one-dimensional)
    :param y: Variable_2 (one-dimensional)
    :return: Pearson coefficient of correlation
    """
    # Implement it yourself, you are allowed to use np.mean, np.sqrt, np.sum.
    r = 0.5 # TODO: change me
    return r


def fit_predict_mse(x, y):
    """
    :param x: Variable 1 (Feature vector (one-dimensional))
    :param y: Variable_2 (one-dimensional), dependent variable
    :return: theta_star - optimal parameters found; mse - Mean Squared Error
    """
    # X =  TODO create a design matrix
    theta_star = [0, 0] # TODO calculate theta using pinv from numpy.linalg (already imported)

    # y_pred = # TODO predict the value of y
    mse = 0 # TODO calculate MSE
    return theta_star, mse


def scatterplot_and_line(x, y, theta, xlabel='x', ylabel='y', title='Title'):
    """
    :param x: Variable_1 (one-dimensional)
    :param y: Variable_2 (one-dimensional), dependent variable
    :param theta: Coefficients of line that fits the data
    :return:
    """
    # theta will be an array with two coefficients, representing the slope and intercept.
    # In which format is it stored in the theta array? Take care of that when plotting the line.
    # TODO
    pass




