import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import OneHotEncoder

######### Replace the data below ################
X = np.array([
    [1, 1, 1],
    [1, -1, 1],
    [1, 1, 3],
    [1, 1, 0]
])

y_class = np.array([
    [1],
    [2],
    [1],
    [3]
])

X_test = np.array([
    [1, 0, -1]
])
######### Replace the data above ################

print("One-hot encoding function")
onehot_encoder = OneHotEncoder(sparse=False)
Ytr_onehot = onehot_encoder.fit_transform(y_class)
print(Ytr_onehot, "\n")

## Linear Classification:
print("Estimated W")
W = inv(X.T @ X) @ X.T @Ytr_onehot
print(W, "\n")

## Testing
y_test = X_test @ W
print("Test")
print(y_test, "\n")

## Labelling
yt_class = [[1 if y == max(x) else 0 for y in x] for x in y_test]
print("class label test")
print(yt_class, "\n")

print("Class Label Test using argmax")
print(np.argmax(y_test) + 1)