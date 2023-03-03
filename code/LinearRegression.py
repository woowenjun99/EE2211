import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression

######### Replace the data below ################
X = np.array([
    [-10], 
    [-8], 
    [-3], 
    [-1], 
    [2], 
    [8]
    ])
Y = np.array([
    [5], 
    [5], 
    [4], 
    [3], 
    [2], 
    [2]])
######### Replace the data above ################

def calculate_weight(X: np.array, Y: np.array):
    """
    Used to compute the weight vector for Linear Regression
    without any bias

    @param X: The set of vectors for X
    @param Y: The score
    @returns: The weight vector
    """
    return inv(X.T @ X) @ X.T @ Y

def calculate_weight_and_bias(X: np.array, Y: np.array):
    """
    Used to compute the weight and bias for Linear Regression
    with biasness

    @param X: The set of vectors for X
    @param Y: The score
    @returns: The weight vector
    """
    multi_model = LinearRegression()
    multi_model.fit(X, Y)
    weight = multi_model.coef_
    bias = multi_model.intercept_
    return weight, bias


weight = calculate_weight(X, Y)
(w, coeff) = calculate_weight_and_bias(X, Y)
print(weight)
print(f'The weight is {w} and the coefficient is {coeff}')

