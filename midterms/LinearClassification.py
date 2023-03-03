import numpy as np
from numpy.linalg import inv

######### Replace the data below ################
X = np.array([
    [1, -7],
    [1, -5],
    [1, 1],
    [1, 5]
    ])

y = np.array([
    [-1],
    [-1],
    [1],
    [1]
])

Xt = np.array([
    [1, -2]
    ])
######### Replace the data above ################

## Linear Regression for classification
w = inv(X.T @ X) @ X.T @ y
print("Estimated w")
print(w)
print("\n")

## Testing
y_predict = Xt @ w
print("Predicted y")
print(y_predict)
print("\n")

y_class_predict = np.sign(y_predict)
print("Predicted y class")
print(y_class_predict)