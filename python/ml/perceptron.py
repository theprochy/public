#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def perceptron(X, y, max_iterations):
    """
    w, b = perceptron(X, y, max_iterations)

    Perceptron algorithm.
    Implements the perceptron algorithm
    (http://en.wikipedia.org/wiki/Perceptron)

    :param X:               d-dimensional observations, (d, number_of_observations) np array
    :param y:               labels of the observations (0 or 1), (n,) np array
    :param max_iterations:  number of algorithm iterations (scalar)
    :return w:              w - weights, (d,) np array
    :return b:              b - bias, python float
    """
    d = X.shape[0]
    N = X.shape[1]
    X = np.multiply(np.vstack([X, np.ones((1, N))]), -2 * y + 1)
    v = np.zeros(d + 1)
    found = False
    for i in range(max_iterations):
        updated = False
        for i in range(N):
            if np.dot(v, X[:, i]) <= 0:
                v = v + X[:, i]
                updated = True
                break
        if not updated:
            found = True
            break
    if not found:
        return np.nan, np.nan
    return v[0:d], v[d]


def lift_dimension(X):
    """
    Z = lift_dimension(X)

    Lifts the dimensionality of the feature space from 2 to 5 dimensions

    :param X:   observations in the original space
                2-dimensional observations, (2, number_of_observations) np array
    :return Z:  observations in the lifted feature space, (5, number_of_observations) np array
    """
    return np.vstack([X, np.square(X[0]), np.multiply(X[0], X[1]), np.square(X[1])])


def classif_quadrat_perc(tst, model):
    """
    K = classif_quadrat_perc(tst, model)

    Classifies test samples using the quadratic discriminative function

    :param tst:     2-dimensional observations, (2, n) np array
    :param model:   dictionary with the trained perceptron classifier (parameters of the discriminative function)
                        model['w'] - weights vector, np array (d, )
                        model['b'] - bias term, python float
    :return:        Y - classification result (contains either 0 or 1), (n,) np array
    """
    return (np.sum(np.array([model['w']]).T * lift_dimension(tst), axis=0) + model['b'] < 0).astype(int)


def kozinec(X, y, max_iterations):
    """
    w, b = kozinec(X, y, max_iterations)

    Kozinec algorithm.
    Implements the Kozinec algorithm

    :param X:               d-dimensional observations, (d, number_of_observations) np array
    :param y:               labels of the observations (0 or 1), (n,) np array
    :param max_iterations:  number of algorithm iterations (scalar)
    :return w:              w - weights, (d,) np array
    :return b:              b - bias, python float
    """
    d = X.shape[0]
    N = X.shape[1]
    X = np.multiply(np.vstack([X, np.ones((1, N))]), -2 * y + 1)
    v = X[:, np.random.randint(N)]
    found = False
    for i in range(max_iterations):
        updated = False
        for i in range(N):
            if v @ X[:, i] <= 0:
                w = v - X[:, i]
                v = v - (v @ w / np.linalg.norm(w) ** 2) * w
                updated = True
                break
        if not updated:
            found = True
            break
    if not found:
        return np.nan, np.nan
    return v[0:d], v[d]


################################################################################
#####                                                                      #####
#####             Below this line are already prepared methods             #####
#####                                                                      #####
################################################################################

def pboundary(X, y, model, figsize=None, style_0='bx', style_1='r+'):
    """
    pboundary(X, y, model)

    Plot boundaries for perceptron decision strategy

    :param X:       d-dimensional observations, (d, number_of_observations) np array
    :param y:       labels of the observations (0 or 1), (n,) np array
    :param model:   dictionary with the trained perceptron classifier (parameters of the discriminative function)
                        model['w'] - weights vector, np array (d, )
                        model['b'] - bias term, python float
    """

    plt.figure(figsize=figsize)
    plt.plot(X[0, y == 0], X[1, y == 0], style_0, ms=10)
    plt.plot(X[0, y == 1], X[1, y == 1], style_1, ms=10)

    minx, maxx = plt.xlim()
    miny, maxy = plt.ylim()

    epsilon = 0.1 * np.maximum(np.abs(maxx - minx), np.abs(maxy - miny))

    x_space = np.linspace(minx - epsilon, maxx + epsilon, 1000)
    y_space = np.linspace(miny - epsilon, maxy + epsilon, 1000)
    x_grid, y_grid = np.meshgrid(x_space, y_space)

    x_grid_fl = x_grid.reshape([1, -1])
    y_grid_fl = y_grid.reshape([1, -1])

    X_grid = np.concatenate([x_grid_fl, y_grid_fl], axis=0)
    Y_grid = classif_quadrat_perc(X_grid, model)
    Y_grid = Y_grid.reshape([1000, 1000])

    blurred_Y_grid = ndimage.gaussian_filter(Y_grid, sigma=0)

    plt.contour(x_grid, y_grid, blurred_Y_grid, colors=['black'])
    plt.xlim(minx, maxx)
    plt.ylim(miny, maxy)
