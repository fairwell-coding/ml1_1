import numpy as np
from numpy.linalg import pinv


def test_fit_line():
    x = np.array([0, 1, 2, 3])
    y = np.array([3, 4, 5, 6])
    a, b = _fit_line(x, y)

    expected_result = [1.0, 3.0]
    print(a, b)  # Should be: a = 1.0, b = 3.0

    assert np.allclose([a, b], expected_result), "Line fitting is working incorrectly. Instead of the correct values a={0} and b={1} the _fit_line method calculated a={2} and b={3}.".format(expected_result[0], expected_result[1], a, b)


def _fit_line(x, y):
    """
    :param x: x coordinates of data points
    :param y: y coordinates of data points
    :return: a, b - slope and intercept of the fitted line
    """
    
    assert x.shape[0] == y.shape[0], "Data dimension mismatch: Number of time dimensions must match number of corresponding function values."
    assert x.shape[0] >= 2, "At leasts two data points are required."

    X = np.empty((x.shape[0], 2))
    X[:, 0] = 1  # first column in matrix X corresponds to calculate the offset/bias b and therefore only contains values of 1
    X[:, 1] = x  # second column in matrix X corresponds to calculate the linear parameter a which is provided via vector x

    theta = np.matmul(pinv(X.T @ X) @ X.T, y)  # analytically calculated model parameters theta

    return theta[1], theta[0]  # a, b


def _intersection(a, b, c, d):
    """
    :param a: slope of the "left" line
    :param b: intercept of the "left" line
    :param c: slope of the "right" line
    :param d: intercept of the "right" line
    :return: x, y - corrdinates of the intersection of two lines
    """

    # Calculate line intersection by using analytically calculated result in report
    A = np.full((2, 2), [[a, -1], [c, -1]])
    offsets = np.full((2, ), [-b, -d])

    intersection = np.matmul(pinv(A), offsets)

    return intersection[0], intersection[1]


def check_if_improved(x_new, y_new, peak, time, signal):
    """
    :param x_new: x-coordinate of a new peak
    :param y_new: y-coordinate of a new peak
    :param peak: index of the peak that we were improving
    :param time: all x-coordinates for ecg signal
    :param signal: all y-coordinates of signal (i.e., ecg signal)
    :return: 1 - if new peak is improvment of the old peak, otherwise 0
    """

    if y_new > signal[peak] and time[peak-1] < x_new < time[peak + 1]:
        return 1
    return 0


def find_new_peak(peak, time, sig):
    """
    This function fits a line through points left of the peak, then another line through points right of the peak.
    Once the coefficients of both lines are obtained, the intersection point can be calculated, representing a new peak.

    :param peak: Index of the peak
    :param time: Time signal (the whole signal, 50 s)
    :param sig: ECG signal (the whole signal for 50 s)
    :return:
    """

    a, b = __fit_left_line(peak, sig, time)
    c, d = __fit_right_line(peak, sig, time)

    # find intersection point
    x_new, y_new = _intersection(a, b, c, d)
    return x_new, y_new, [(a, b), (c, d)]


def __fit_right_line(peak, sig, time):
    n_points = 2  # number of points excluding peak on the right line
    x = np.empty(n_points)
    y = np.empty(n_points)

    for i in range(n_points):  # define points on the left line in reverse order
        x[i] = time[peak + (i + 1)]
        y[i] = sig[peak + (i + 1)]

    c, d = _fit_line(x, y)

    return c, d


def __fit_left_line(peak, sig, time):
    n_points = 2  # number of points excluding peak on the left line
    x = np.empty(n_points)
    y = np.empty(n_points)

    for i in range(n_points):  # define points on the left line in reverse order
        x[i] = time[peak - (i + 1)]
        y[i] = sig[peak - (i + 1)]

    a, b = _fit_line(x, y)

    return a, b
