###
#-------------------------------------------------------------------------------
# linearRegression.py
#-------------------------------------------------------------------------------
#
# Author:       Alwin Tareen
# Created:      Feb 14, 2022
# Execution:    python3 linearRegression.py
#
# This program fits the linear regression parameters theta to the dataset.
#
##

import csv
import matplotlib.pyplot as plt
import numpy as np

def producePlot(theta):
    with open('ex1data1.txt', 'r') as f:
        entries = list(csv.reader(f))
        population = [float(row[0]) for row in entries]
        profit = [float(row[1]) for row in entries]

    linear_x = range(int(min(population)), int(max(population))+1)
    linear_y = [theta[1]*x + theta[0] for x in linear_x]
    plt.plot(linear_x, linear_y, label='Training data')
    plt.scatter(population, profit, marker='+', color='r')
    plt.xlabel('City Population in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Food Truck Profit vs. City Population')
    plt.legend(loc = 'upper left')
    plt.show()

def computeCost(X, y, theta):
    m = y.shape[0]
    h = np.matmul(X, theta)
    J = 1 / (2*m) * ((h - y)**2).sum()
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m = y.shape[0]
    for trial in range(iterations):
        h = np.matmul(X, theta)
        theta -= (alpha / m) * np.matmul(X.transpose(), (h - y))
    return theta[0][0], theta[1][0]

def linearRegression():
    with open('ex1data1.txt', 'r') as f:
        entries = list(csv.reader(f))
        X = np.array([float(row[0]) for row in entries])
        X = np.expand_dims(X, axis=1)
        y = np.array([float(row[1]) for row in entries])
        y = np.expand_dims(y, axis=1)

    # determine the number of training examples
    m = y.shape[0]
    J = []
    iterations = 1500
    alpha = 0.01

    # add a column of ones to X
    ones = np.ones(m)
    ones = np.expand_dims(ones, axis=1)
    X = np.hstack((ones, X))

    # initialize the theta fitting parameters and compute cost J
    theta = np.array([0.0, 0.0])
    theta = np.expand_dims(theta, axis=1)
    J.append(round(computeCost(X, y, theta), 2))
    
    theta = np.array([-1.0, 2.0])
    theta = np.expand_dims(theta, axis=1)
    J.append(round(computeCost(X, y, theta), 2))
    
    # calculate the theta found by gradient descent
    theta = np.array([0.0, 0.0])
    theta = np.expand_dims(theta, axis=1)
    theta = gradientDescent(X, y, theta, alpha, iterations)
    
    # predict values for population sizes of 35,000 and 70,000
    pop35 = np.array([1.0, 3.5])
    predict35 = np.matmul(pop35, theta) * 10000
    pop70 = np.array([1.0, 7.0])
    predict70 = np.matmul(pop70, theta) * 10000
    
    return J, theta, predict35, predict70

def main():
    result = linearRegression()
    print(f'J, theta, predict35, predict70 = {result}')
    theta = result[1]
    #producePlot(theta)

if __name__ == '__main__':
    main()

