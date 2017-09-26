from loadFittingDataP2 import getData
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


X, Y = getData(False)

def expand(x, m):
    x = np.array(x)
    expanded = np.zeros([len(x), m+1])
    for i in range(m+1):
        expanded[:,i] = x**i
    return expanded


def SSE_poly(y, x, theta):
    X = expand(x, len(theta))
    error = y - np.matmul(X, theta)


def maxLikelihoodVector(x,y,m):
    x_panded = expand(x,m)
    x_panded = np.matrix(x_panded)
    y = np.matrix(y)
    w = np.linalg.inv(x_panded.T*x_panded)*x_panded.T*y.T
    return w

maxVec = maxLikelihoodVector(X,Y,3)
X_test = np.linspace(0,1,100)
sol = np.matmul(expand(X_test,3),maxVec)
plt.plot(X_test,sol)

plt.plot(X,Y,'o')

plt.xlabel('x')
plt.ylabel('y')
plt.show()