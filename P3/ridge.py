import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import regressData


def expand(x, m):
    x = np.array(x)
    expanded = np.zeros([len(x), m+1])
    for i in range(m+1):
        expanded[:,i] = (x**i).reshape(-1)
    return expanded


#Implement Ridge Regression
def ridgeRegression(x,y,l,m):
    x_p = expand(x,m)
    x_p = np.matrix(x_p)
    y = np.matrix(y)
    I = np.identity(x_p.shape[1])
    a = (l*I).shape
    beta = np.linalg.inv(x_p.T*x_p + l*I)*x_p.T*y
    return beta

from P2.loadFittingDataP2 import getData
#
# X, Y = getData(True)
# # # Part A
# X_train, Y_train = regressData.getData('curvefittingp2.txt')
#
# theta_0_10 = ridgeRegression(X_train, Y_train, 0, 10)
# theta_1_10 = ridgeRegression(X_train, Y_train, 0.01, 10)
# theta_2_10 = ridgeRegression(X_train, Y_train, 1, 10)
# theta_0_5 = ridgeRegression(X_train, Y_train, 0, 5)
# theta_1_5 = ridgeRegression(X_train, Y_train, 0.01, 5)
# theta_2_5 = ridgeRegression(X_train, Y_train, 1, 5)
# theta_0_2 = ridgeRegression(X_train, Y_train, 0, 2)
# theta_1_2 = ridgeRegression(X_train, Y_train, 0.01, 2)
# theta_2_2 = ridgeRegression(X_train, Y_train, 1, 2)
#
#
# X_test = np.linspace(0, 1, 100)
# sol_0_10 = np.matmul(expand(X_test, 10), theta_0_10)
# sol_1_10 = np.matmul(expand(X_test, 10), theta_1_10)
# sol_2_10 = np.matmul(expand(X_test, 10), theta_2_10)
# sol_0_5 = np.matmul(expand(X_test, 5), theta_0_5)
# sol_1_5 = np.matmul(expand(X_test, 5), theta_1_5)
# sol_2_5 = np.matmul(expand(X_test, 5), theta_2_5)
# sol_0_2 = np.matmul(expand(X_test, 2), theta_0_2)
# sol_1_2 = np.matmul(expand(X_test, 2), theta_1_2)
# sol_2_2 = np.matmul(expand(X_test, 2), theta_2_2)
#
# plt.plot(X_test, sol_0_10, label="Lambda = 0 M = 10")
# plt.plot(X_test, sol_1_10, label="Lambda = 0.01 M = 10")
# plt.plot(X_test, sol_2_10, label="Lambda = 1 M = 10")
# plt.plot(X_test, sol_0_5, label="Lambda = 0 M = 5")
# plt.plot(X_test, sol_1_5, label="Lambda = 0.01 M = 5")
# plt.plot(X_test, sol_2_5, label="Lambda = 1 M = 5")
# plt.plot(X_test, sol_0_2, label="Lambda = 0 M = 2")
# plt.plot(X_test, sol_1_2, label="Lambda = 0.01 M = 2")
# plt.plot(X_test, sol_2_2, label="Lambda = 1 M = 2")
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title("Exploring different lambdas and M values with ridge regression")
# plt.legend()
# plt.show("hold")

X_A_train, Y_A_train = regressData.getData('curvefittingp2.txt')
X_B_train, Y_B_train = regressData.getData('curvefittingp2.txt')
X_val, Y_val = regressData.getData('curvefittingp2.txt')
