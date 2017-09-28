import numpy as np
import matplotlib.pyplot as plt
from P2.main import expand_theta
import pylab as pl
import regressData
from tqdm import tqdm

def expand(x, m):
    x = np.array(x)
    expanded = np.zeros([len(x), m+1])
    for i in range(m+1):
        expanded[:,i] = (x**i).reshape(-1)
    return expanded

#Implement Ridge Regression
def ridgeRegression_Expanded(x,y,l):
    x = np.matrix(x)
    y = np.matrix(y)
    I = np.identity(x.shape[1])
    beta = np.linalg.inv(x.T*x + l*I)*x.T*y
    return beta


#Implement Ridge Regression
def ridgeRegression(x,y,l,m, expander=expand):
    x_p = expander(x, m)
    x_p = np.matrix(x_p)
    y = np.matrix(y)
    I = np.identity(x_p.shape[1])
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

# # # Train A. val, test B
# X_A_train, Y_A_train = regressData.getData('regressA_train.txt')
# X_B_train, Y_B_train = regressData.getData('regressB_train.txt')
# X_val, Y_val = regressData.getData('regress_validate.txt')
# l0 = []
# l1 = []
# l2 = []
# l3 = []
# l4 = []
# l5 = []
#
# for m in tqdm(range(1, 10)):
#     theta_0 = ridgeRegression(X_A_train, Y_A_train, 0, m)
#     theta_2 = ridgeRegression(X_A_train, Y_A_train, 0.01, m)
#     theta_4 = ridgeRegression(X_A_train, Y_A_train, 0.1, m)
#     theta_5 = ridgeRegression(X_A_train, Y_A_train, 1, m)
#
#     l0.append(np.square(Y_val-np.matmul(expand(X_val, m), theta_0)).sum())
#     l2.append(np.square(Y_val-np.matmul(expand(X_val, m), theta_2)).sum())
#     l4.append(np.square(Y_val-np.matmul(expand(X_val, m), theta_4)).sum())
#     l5.append(np.square(Y_val-np.matmul(expand(X_val, m), theta_5)).sum())


# plt.plot(l0, label="lambda = 0")
# plt.plot(l2, label="lambda = 0.01")
# plt.plot(l4, label="lambda = 0.1")
# plt.plot(l5, label="lambda = 1")
# plt.xlabel("M")
# plt.ylabel("SSE over validation set")
# plt.title("Training on A vs validation set")
# plt.legend()
# plt.show("hold")
#
#### BEST PARAMS WERE l = 0 m = 1
#
# X_test = np.linspace(-3, 3, 100)
# theta = ridgeRegression(X_A_train, Y_A_train, 0, 1)
# sol = np.matmul(expand(X_test, 1), theta)
# plt.title("Best mode trained on Train_A and validated on Validate vs Train_B")
# plt.plot(X_test, sol, label="lambda = 0, m = 1")
# plt.plot(X_B_train, Y_B_train, 'o', label = "test data")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show("hold")


# # # Train B. val, test A
# X_A_train, Y_A_train = regressData.getData('regressA_train.txt')
# X_B_train, Y_B_train = regressData.getData('regressB_train.txt')
# X_val, Y_val = regressData.getData('regress_validate.txt')
# l0 = []
# l1 = []
# l2 = []
# l3 = []
# l4 = []
# l5 = []
#
# for m in tqdm(range(1, 10)):
#     theta_0 = ridgeRegression(X_B_train, Y_B_train, 0, m)
#     theta_2 = ridgeRegression(X_B_train, Y_B_train, 0.01, m)
#     theta_4 = ridgeRegression(X_B_train, Y_B_train, 0.1, m)
#     theta_5 = ridgeRegression(X_B_train, Y_B_train, 10, m)
#
#     l0.append(np.square(Y_val-np.matmul(expand(X_val, m), theta_0)).sum())
#     l2.append(np.square(Y_val-np.matmul(expand(X_val, m), theta_2)).sum())
#     l4.append(np.square(Y_val-np.matmul(expand(X_val, m), theta_4)).sum())
#     l5.append(np.square(Y_val-np.matmul(expand(X_val, m), theta_5)).sum())


# plt.plot(l0, label="lambda = 0")
# plt.plot(l2, label="lambda = 0.01")
# plt.plot(l4, label="lambda = 0.1")
# plt.plot(l5, label="lambda = 1")
# plt.xlabel("M")
# plt.ylabel("SSE over validation set")
# plt.title("Training on B vs validation set")
# plt.legend()
# plt.show("hold")

#### BEST PARAMS WERE l = 1 m = 2
#
# X_test = np.linspace(-3, 3, 100)
# theta = ridgeRegression(X_B_train, Y_B_train, 1, 2)
# sol = np.matmul(expand(X_test, 2), theta)
# plt.title("Best mode trained on Train_B and validated on Validate vs Train_A")
# plt.plot(X_test, sol, label="lambda = 1, m = 2")
# plt.plot(X_A_train, Y_A_train, 'o', label = "test data")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show("hold")