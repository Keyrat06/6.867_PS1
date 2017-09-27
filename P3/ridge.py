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


X,Y = regressData.getData('curvefittingp2.txt')

#Implement Ridge Regression
def ridgeRegression(x,y,l,m):
    x_p = expand(x,m)
    x_p = np.matrix(x_p)
    y = np.matrix(y)
    I = np.identity(x_p.shape[1])
    a = (l*I).shape
    beta = np.linalg.inv(x_p.T*x_p + l*I)*x_p.T*y
    return beta

r = ridgeRegression(X,Y,.001,10)

X_test = np.linspace(0,1,100)
sol = np.matmul(expand(X_test, 10), r)
plt.plot(X_test,sol)

plt.plot(X,Y,'o')

plt.xlabel('x')
plt.ylabel('y')
plt.show()