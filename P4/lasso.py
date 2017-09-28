from P4 import lassoData
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np
from P3.ridge import ridgeRegression_Expanded
import pylab as pl

def expand_theta(x):
    x = np.array(x)
    expanded = np.zeros([len(x), 13])
    expanded[:,0] = x.flatten()
    for i in range(1, 13):
        expanded[:, i] = (np.sin(0.4*i*np.pi*x)).flatten()
    return expanded


x_train, y_train = lassoData.lassoTrainData()
x_valid, y_valid = lassoData.lassoValData()
x_test, y_test = lassoData.lassoTestData()

X_train = expand_theta(x_train)
X_valid = expand_theta(x_valid)
X_test = expand_theta(x_test)


### Training

# lambdas = [0.000001, 0.001, 0.01, 0.05, 0.1, 0.5]

# theta_l0 = Lasso(alpha=lambdas[0]).fit(X_train, y_train).coef_
# theta_r0 = ridgeRegression_Expanded(X_train, y_train, lambdas[0])
# theta_l1 = Lasso(alpha=lambdas[1]).fit(X_train, y_train).coef_
# theta_r1 = ridgeRegression_Expanded(X_train, y_train, lambdas[1])
# theta_l2 = Lasso(alpha=lambdas[2]).fit(X_train, y_train).coef_
# theta_r2 = ridgeRegression_Expanded(X_train, y_train, lambdas[2])
# theta_l3 = Lasso(alpha=lambdas[3]).fit(X_train, y_train).coef_
# theta_r3 = ridgeRegression_Expanded(X_train, y_train, lambdas[3])
# theta_l4 = Lasso(alpha=lambdas[4]).fit(X_train, y_train).coef_
# theta_r4 = ridgeRegression_Expanded(X_train, y_train, lambdas[4])
# theta_l5 = Lasso(alpha=lambdas[5]).fit(X_train, y_train).coef_
# theta_r5 = ridgeRegression_Expanded(X_train, y_train, lambdas[5])

# ### tune hyper parameters on validation set
# tune_lasso = []
# tune_lasso.append(np.square(y_valid.flatten()-np.matmul(X_valid, theta_l0)).sum())
# tune_lasso.append(np.square(y_valid.flatten()-np.matmul(X_valid, theta_l1)).sum())
# tune_lasso.append(np.square(y_valid.flatten()-np.matmul(X_valid, theta_l2)).sum())
# tune_lasso.append(np.square(y_valid.flatten()-np.matmul(X_valid, theta_l3)).sum())
# tune_lasso.append(np.square(y_valid.flatten()-np.matmul(X_valid, theta_l4)).sum())
# tune_lasso.append(np.square(y_valid.flatten()-np.matmul(X_valid, theta_l5)).sum())
# #
# tune_ridge = []
# tune_ridge.append(np.square(y_valid-np.matmul(X_valid, theta_r0)).sum())
# tune_ridge.append(np.square(y_valid-np.matmul(X_valid, theta_r1)).sum())
# tune_ridge.append(np.square(y_valid-np.matmul(X_valid, theta_r2)).sum())
# tune_ridge.append(np.square(y_valid-np.matmul(X_valid, theta_r3)).sum())
# tune_ridge.append(np.square(y_valid-np.matmul(X_valid, theta_r4)).sum())
# tune_ridge.append(np.square(y_valid-np.matmul(X_valid, theta_r5)).sum())
#
# ### best lambda for both 0.01
# plt.plot(lambdas, tune_lasso, label="lasso")
# plt.plot(lambdas, tune_ridge, label="ridge")
# plt.xlabel("lambda")
# plt.ylabel("SSE over validation set")
# plt.title("Tuning lambda on validation set")
# plt.legend()
# plt.show('hold')


###
# x_space = np.linspace(-1, 1, 100)
# X_space = expand_theta(x_space)
#
# theta_0 = Lasso(0).fit(X_train, y_train).coef_
# true_theta = np.array([0, 0, 5.6463, 0.7786, 0, 0.8109, 2.6827, 0, 0, 0, 0, 0, 0])
# true_sol = np.matmul(X_space, true_theta)
# lasso_2_sol = np.matmul(X_space, theta_l2)
# ridge_2_sol = np.matmul(X_space, theta_r2)
# lambda_0_sol = np.matmul(X_space, theta_0)

#
#
# plt.plot(x_space, true_sol, label="True function")
# plt.plot(x_space, lasso_2_sol, label="Lasso lambda = 0.01")
# plt.plot(x_space, ridge_2_sol, label="Ridge lambda = 0.01")
# plt.plot(x_space, lambda_0_sol, label="lambda = 0")
#
# plt.plot(x_train, y_train, 'o', label="training")
# plt.plot(x_valid, y_valid, 'o', label="validation")
# plt.plot(x_test, y_test, 'o', label="testing")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Fitting The Data")
# plt.legend()
# plt.show()


theta_0 = Lasso(0).fit(X_train, y_train).coef_
lasso = Lasso(0.1).fit(X_train, y_train).coef_
ridge = ridgeRegression_Expanded(X_train, y_train, 0.1)
true_theta = np.array([0, 0, 5.6463, 0.7786, 0, 0.8109, 2.6827, 0, 0, 0, 0, 0, 0])

# # BAR GRAPHS
plt.subplot(2, 2, 1)
plt.title("W true")
plt.bar(np.arange(len(true_theta)), true_theta)

plt.subplot(2, 2, 2)
plt.title("Estimated w with Lasso")
plt.bar(np.arange(len(lasso)), lasso)

plt.subplot(2, 2, 3)
plt.title("Estimated w with Ridge")
plt.bar(np.arange(len(ridge)), ridge)

plt.subplot(2, 2, 4)
plt.title("Estimated w with lambda = 0")
plt.bar(np.arange(len(theta_0)), theta_0)

plt.subplots_adjust(hspace=0.3)

plt.show("hold")