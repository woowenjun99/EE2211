"""
This file is used to calculate the polynomial features.
Refer to lecture 6 slides
"""
import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import PolynomialFeatures

######### Replace the data below ################
X = np.array([
    [1, 1],
    [-1, 1],
    [1, -1],
    [-1, -1]
])

y = np.array([
    [1],
    [-1],
    [-1],
    [1]
])

Xnew = np.array([[0.2, 0.5], [-0.9, 0.7]])
######### Replace the data above ################

# Generate polynomial features
order = 2
poly = PolynomialFeatures(order)
P = poly.fit_transform(X)
print("Matrix P")
print(P)
print("Under-determined system")

# dual solution: # of rows < # of columns
w_dual = P.T @ inv(P @ P.T) @ y
print("Unique constrained solution, no ridge")
print(w_dual)

# Testing
print("Prediction")

Pnew = poly.fit_transform(Xnew)
Ynew = Pnew @ w_dual
print(Ynew)
print(np.sign(Ynew))