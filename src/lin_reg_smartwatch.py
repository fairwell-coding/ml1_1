from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

def pearson_coefficient(x, y):
    """
    :param x: Variable_1 (one-dimensional)
    :param y: Variable_2 (one-dimensional)
    :return: Pearson coefficient of correlation
    """

    E_X = np.mean(x)  # expectation of random variable X
    E_Y = np.mean(y)  # expectation of random variable Y
    cov_X_Y = np.mean((x - E_X) * (y - E_Y))  # covariance of random variables X, Y

    E_X2 = np.mean(x**2)  # expectation of random variable X^2
    E_Y2 = np.mean(y**2)  # expectation of random variable Y^2
    var_X = E_X2 - E_X**2
    var_Y = E_Y2 - E_Y ** 2
    denominator = np.sqrt(var_X * var_Y)

    pearson_coef = cov_X_Y / denominator

    return pearson_coef


def __plot_scatter(x, y, x_label, y_label, theta: np.ndarray):
    plt.plot(x, y, 'o', color="forestgreen", markersize="4")

    # Fitted line
    fitted_line = theta[1] * x + theta[0]
    plt.plot(x, fitted_line, color="lightskyblue")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def perform_linear_regression(data: np.ndarray, column_to_id: Dict[str, int], independent_variable: str, dependent_variable: str, normalized=False, create_plot=False):
    x = data[:, column_to_id[independent_variable]]
    y = data[:, column_to_id[dependent_variable]]

    if normalized:
        x = __normalize_feature(x)
        y = __normalize_feature(y)

    theta, mse = fit_predict_mse(x, y)
    pearson_coef = pearson_coefficient(x, y)

    print('{0} -> {1}: theta = {2}, mse = {3}, pearson_coef= {4}'.format(independent_variable, dependent_variable, theta, mse, pearson_coef))
    print('Pearson coefficient calculated by built-in numpy function: r = {0}'.format(np.corrcoef(x, y)[1][0]))  # comparison to verify that our own implementation is correct

    if create_plot:
        __plot_scatter(x, y, independent_variable, dependent_variable, theta)


def __normalize_feature(feature):
    """ Convert feature dimension to standard distribution (mean = 0, deviation = 1).
    """

    shifted_data = feature - np.mean(feature)
    max = np.max(shifted_data)
    min = np.min(shifted_data)
    normalized_data = np.array([(x - min) / (max - min) for x in shifted_data])

    return normalized_data


def fit_predict_mse(x, y):
    """
    :param x: Variable 1 (Feature vector (one-dimensional))
    :param y: Variable_2 (one-dimensional), dependent variable
    :return: theta_star - optimal parameters found; mse - Mean Squared Error
    """

    # Create design matrix X
    X = np.empty((x.shape[0], 2))
    X[:, 0] = 1  # first column in matrix X corresponds to calculate the offset/bias b and therefore only contains values of 1
    X[:, 1] = x  # second column in matrix X corresponds to calculate the linear parameter a which is provided via vector x

    theta = np.matmul(pinv(X.T @ X) @ X.T, y)  # analytically calculated model parameters theta
    y_pred = X @ theta
    mse = 1 / x.shape[0] * np.sum((y - y_pred)**2)

    return theta, mse


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




