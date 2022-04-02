import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_eggholder_function(f):
    """
    Plotting the 3D surface of a given cost function f.
    :param f: The function to visualize
    :return:
    """

    n = 1000
    bounds = [-512, 512]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    x_ax = np.linspace(bounds[0], bounds[1], n)
    y_ax = np.linspace(bounds[0], bounds[1], n)
    XX, YY = np.meshgrid(x_ax, y_ax)

    ZZ = np.zeros(XX.shape)
    ZZ = f([XX, YY])

    ax.plot_surface(XX, YY, ZZ, cmap='jet')
    plt.show()


def gradient_descent(f, df, x, learning_rate, max_iter):
    """
    Find the optimal solution of the function f(x) using gradient descent:
    Until the max number of iteration is reached, decrease the parameter x by the gradient times the learning_rate.
    The function should return the minimal argument x and the list of errors at each iteration in a numpy array.

    :param f: function to minimize
    :param df: function representing the gradient of f
    :param x: vector, initial point
    :param learning_rate:
    :param max_iter: maximum number of iterations
    :return: x (solution, vector), E_list (array of errors over iterations)
    """

    E_list = np.zeros(max_iter)

    for i in range(max_iter):
        x = x - learning_rate * df(x, f)
        E_list[i] = f(x)

    return x, E_list


def eggholder(x):
    z = - (x[1] + 47) * np.sin(np.sqrt(np.abs(x[0] / 2 + (x[1] + 47)))) - x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))  # Eggholder cost function
    return z


def gradient_special(x: np.ndarray, y: np.ndarray, z: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray):
    pass


def gradient_eggholder(x, func=eggholder, epsilon=1e-4):
    sqrt_term_1 = x[0] / 2 + x[1] + 47
    sqrt_term_2 = x[0] - (x[1] + 47)

    if sqrt_term_1 == 0 or sqrt_term_2 == 0:
        grad_x = func([x[0] + epsilon, x[1]]) - func([x[0] - epsilon, x[1]])
        grad_y = func([x[0], x[1] + epsilon]) - func([x[0], x[1] - epsilon])
        return np.multiply([grad_x, grad_y], 1 / (2 * epsilon))

    grad_x = - (x[1] + 47) * np.cos(np.sqrt(np.abs(sqrt_term_1))) * 1 / (2 * np.sqrt(np.abs(sqrt_term_1))) * sqrt_term_1 / np.abs(sqrt_term_1) * 1 / 2 \
             - np.sin(np.sqrt(np.abs(sqrt_term_2))) - x[0] * np.cos(np.sqrt(np.abs(sqrt_term_2))) * 1 / (2 * np.sqrt(np.abs(sqrt_term_2))) * sqrt_term_2 / np.abs(sqrt_term_2)
    grad_y = - np.sin(np.sqrt(np.abs(sqrt_term_1))) - (x[1] + 47) * np.cos(np.sqrt(np.abs(sqrt_term_1)) * 1 / (2 * np.sqrt(np.abs(sqrt_term_1))) * sqrt_term_1 / np.abs(sqrt_term_1) \
             + x[0] * np.cos(np.sqrt(np.abs(sqrt_term_2))) * 1 / (2 * np.sqrt(np.abs(sqrt_term_2))) * sqrt_term_2 / np.abs(sqrt_term_2))

    return np.array([grad_x, grad_y])
