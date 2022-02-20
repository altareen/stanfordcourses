###
#-------------------------------------------------------------------------------
# logisticRegression.py
#-------------------------------------------------------------------------------
#
# Author:       Alwin Tareen
# Created:      Feb 18, 2022
# Execution:    python3 logisticRegression.py
#
# This program fits the logistic regression parameters theta to the dataset.
#
##

import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

def producePlot(theta):
    with open('ex2data1.txt', 'r') as f:
        entries = list(csv.reader(f))
        exam1_admit = [float(row[0]) for row in entries if int(row[2]) == 1]
        exam2_admit = [float(row[1]) for row in entries if int(row[2]) == 1]
        exam1_decline = [float(row[0]) for row in entries if int(row[2]) == 0]
        exam2_decline = [float(row[1]) for row in entries if int(row[2]) == 0]

    # determine the x range from the min and max of the exam scores
    linear_x = range(int(min(exam1_admit+exam1_decline)), int(max(exam2_admit+exam2_decline))+1)
    
    # determine the decision boundary using calculations from the assigment
    #linear_y = [(-1/theta[2]) * (theta[1]*x + theta[0]) for x in linear_x]
    #plt.plot(linear_x, linear_y, label='Decision Boundary')

    # determine the decision boundary using equation: x_2 = (-theta1*x1 - theta0)/theta2
    slope = -theta[1]/theta[2]
    yint = -theta[0]/theta[2]
    f_line = lambda x: slope*x + yint
    plt.plot(linear_x, f_line(linear_x), label='Decision Boundary')

    plt.scatter(exam1_admit, exam2_admit, marker='+', color='k', label='Admitted')
    plt.scatter(exam1_decline, exam2_decline, marker='o', color='y', label='Not Admitted')
    plt.xlabel('Exam 1 Score')
    plt.ylabel('Exam 2 Score')
    plt.title('College Admissions Decisions Based on Two Exam Scores')
    plt.legend(loc = 'lower left')
    plt.show()

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def computeCost(theta, X, y):
    theta = theta.reshape(-1, 1) # scipy.optimize expects theta to be a 2-d column array
    m = y.shape[0]
    h = np.matmul(X, theta)
    first = np.matmul(-y.transpose(), np.log(sigmoid(h)))
    second = np.matmul((1 - y).transpose(), np.log(1 - sigmoid(h)))
    J = (1 / m) * (first - second)
    return J[0][0]

def computeGrad(theta, X, y):
    theta = theta.reshape(-1, 1) # scipy.optimize expects theta to be a 2-d column array
    m = y.shape[0]
    h = np.matmul(X, theta)
    grad = (1 / m) * np.matmul(X.transpose(), (sigmoid(h) - y))
    return grad.ravel() # scipy.optimize expects grad to be a simple array

def predict(theta, X):
    h = sigmoid(np.matmul(X, theta))
    p = h >= 0.5
    return p

def logisticRegression():
    with open('ex2data1.txt', 'r') as f:
        entries = list(csv.reader(f))
        data = [[float(row[0]), float(row[1])] for row in entries]
        X = np.array(data)
        y = np.array([int(row[2]) for row in entries])
        y = np.expand_dims(y, axis=1)

    # initialize lists for the cost and gradient solutions
    J = []
    grad = []

    # determine the number of training examples
    m = y.shape[0]
    n = X.shape[1]

    # add a column of ones to X
    ones = np.ones(m)
    ones = np.expand_dims(ones, axis=1)
    X = np.hstack((ones, X))

    # initialize theta to a column vector of zeros
    initial_theta = np.zeros(n + 1)
    initial_theta = np.expand_dims(initial_theta, axis=1)

    # calculate the cost and gradient
    J.append(computeCost(initial_theta, X, y))
    grad.append(computeGrad(initial_theta, X, y))

    # initialize theta to a column vector of non-zero values
    initial_theta = np.array([[-24], [0.2], [0.2]])

    # calculate the cost and gradient
    J.append(computeCost(initial_theta, X, y))
    grad.append(computeGrad(initial_theta, X, y))
    
    # determine the optimal parameters of theta using fmin_bfgs
    theta = opt.fmin_bfgs(f=computeCost, x0=initial_theta, fprime=computeGrad, args=(X, y))
    #theta = opt.fmin_tnc(func=computeCost, x0=initial_theta, fprime=computeGrad, args=(X, y))[0]
    
    # optimal cost determined from scipy's fmin_bfgs value of theta
    J.append(computeCost(theta, X, y))
    grad.append(computeGrad(theta, X, y))
    
    # determine the accuracy of the training set results
    p = np.array(predict(theta, X))
    p = np.expand_dims(p, axis=1)
    accuracy = np.mean(p == y) * 100
    
    return J, grad, theta, accuracy

def main():
    J, grad, theta, accuracy = logisticRegression()
    print(f'J = {J}\n grad = {grad}\n theta = {theta}\n accuracy = {accuracy}')
    producePlot(theta)

if __name__ == '__main__':
    main()
