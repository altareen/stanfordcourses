###
#-------------------------------------------------------------------------------
# test_problem1.py
#-------------------------------------------------------------------------------
#
# Author:       Alwin Tareen
# Created:      Feb 04, 2022
# Execution:    pytest -v
#
# This program is the pytest test bench for the problem1.py code.
#
##

from linearRegression import gradientDescent
import numpy as np
import pytest

@pytest.mark.parametrize('X, y, theta, alpha, iterations, expected', [
    (np.array([[1, 5], [1, 2], [1, 4], [1, 5]]), np.array([[1], [6], [4], [2]]), np.array([[0.0], [0.0]]), 0.01, 1000, (5.2148, -0.5733)),
    (np.array([[1, 5], [1, 2]]), np.array([[1], [6]]), np.array([[0.5], [0.5]]), 0.1, 10, (1.7099, 0.1923)),
])

def test_gradientDescent(X, y, theta, alpha, iterations, expected):
    actual = gradientDescent(X, y, theta, alpha, iterations)
    assert actual == expected

