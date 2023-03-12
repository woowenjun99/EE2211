from numpy import array, power, zeros, eye, matlib, mean
from numpy.linalg import inv
from sklearn.preprocessing import PolynomialFeatures


def CreateRegressors(x: array, max_order: int):
    P: list[array] = []
    for order in range(1, max_order + 1):
        poly = PolynomialFeatures(order)
        current_regressors = poly.fit_transform(x)
        P.append(current_regressors)
    return P


def EstimateRegressionCoefficients(P_list: list[array], y, reg=None):
    w_list = []
    for P in P_list:
        if reg is None:
            num_row, num_col = P.shape
            if num_col > num_row:
                w = P.T @ inv(P @ P.T) @ y
            else:
                w = (inv(P.T @ P) @ P.T) @ y
            w_list.append(w)
        else:
            w_list.append(inv(P.T @ P + reg * eye(P.shape[1])) @ P.T @ y)
    return w_list


def PerformPrediction(P_list, w_list):
    N = P_list[0].shape[0]
    max_order = len(P_list)
    y_predict_mat = zeros([N, max_order])

    for order in range(len(w_list)):
        y_predict = P_list[order] @ w_list[order]
        y_predict_mat[:, order] = y_predict

    return y_predict_mat


#########################################################################
# Input the list of training data for x and y (Only edit the data here) #
#########################################################################
x = array([-10, -8, -3, -1, 2, 7])
y = array([4.18, 2.42, 0.22, 0.12, 0.25, 3.09])
xt = array([-9, -7, -5, -4, -2, 1, 4, 5, 6, 9])
yt = array([3, 1.81, 0.80, 0.25, -0.19, 0.4, 1.24, 1.68, 2.32, 5.05])
max_order = 6
reg = 1

#############################
# Training (w/o regression) #
#############################
P_train_list = CreateRegressors(x.reshape(-1, 1), max_order)
w_list = EstimateRegressionCoefficients(P_train_list, y)
y_train_pred = PerformPrediction(P_train_list, w_list)
train_error = y_train_pred - matlib.repmat(y, max_order, 1).T
train_MSE = mean(power(train_error, 2), 0)
print(f"Training MSE\n{train_MSE.reshape(-1, 1)}\n")

############################
# Testing (w/o regression) #
############################
P_test_list = CreateRegressors(xt.reshape(-1, 1), max_order)
y_test_pred = PerformPrediction(P_test_list, w_list)
test_error = y_test_pred - matlib.repmat(yt, max_order, 1).T
test_MSE = mean(power(test_error, 2), 0)
print(f"Testing MSE\n{test_MSE.reshape(-1, 1)}\n")
