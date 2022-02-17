###
#-------------------------------------------------------------------------------
# linearRegressionMulti.py
#-------------------------------------------------------------------------------
#
# Author:       Alwin Tareen
# Created:      Feb 15, 2022
# Execution:    python3 linearRegressionMulti.py
#
# This program fits the linear regression parameters theta to the dataset.
#
##

import csv
import numpy as np

def normalEquation(X, y):
    invert = np.linalg.inv(np.matmul(X.transpose(), X))
    product = np.matmul(X.transpose(), y)
    theta = np.matmul(invert, product)
    return theta

def linearRegressionMulti():
    with open('ex1data2.txt', 'r') as f:
        entries = list(csv.reader(f))
        data = [[int(row[0]), int(row[1])] for row in entries]
        X = np.array(data)
        y = np.array([int(row[2]) for row in entries])
        y = np.expand_dims(y, axis=1)

    # determine the number of training examples
    m = y.shape[0]

    # add a column of ones to X
    ones = np.ones(m)
    ones = np.expand_dims(ones, axis=1)
    X = np.hstack((ones, X))

    # calculate theta using the normal equation
    theta = normalEquation(X, y)
    return theta

def main():
    theta = linearRegressionMulti()
    print(f'theta = {theta}')
    predict = np.matmul([1, 1650, 3], theta)
    print(f'predicted price for a 1650 sq-ft, 3 bedroom house = {predict}')

if __name__ == '__main__':
    main()

