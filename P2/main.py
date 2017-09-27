from P2.loadFittingDataP2 import getData
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


# X, Y = getData(False)

def expand(x, m):
    x = np.array(x)
    expanded = np.zeros([len(x), m+1])
    for i in range(m+1):
        expanded[:, i] = (x**i).reshape(-1)
    return expanded

def expand_theta(x, m):
    x = np.array(x)
    expanded = np.zeros([len(x), m+1])
    for i in range(m+1):
        expanded[:, i] = np.cos(i*np.pi*x)
    return expanded

def SSE_poly(y, x, theta):
    X = expand(x, len(theta)-1)
    return SSE_poly_expanded(y, X, theta)

def SSE_poly_expanded(y, X, theta):
    error = y.reshape(1, -1) - np.matmul(X, theta)
    return (error**2).sum()

def d_SSE_poly(y, x, theta):
    X = expand(x, len(theta)-1)
    return d_SSE_poly_expanded(y, X, theta)

def d_SSE_poly_expanded(y, X, theta):
    error = np.matmul(X, theta) - y.reshape(1, -1)
    return 2*(error.T * X).sum(axis=0)

def gradientApprox(f, x, d):
    return (f(x+d)-f(x))/d


#with these params SSE_poly should return 0
# y = np.array([3, 5, 7, 9, 11])
# x = np.array([1, 2, 3, 4, 5])
# theta = np.array([1, 2])
# print (SSE_poly(y, x, theta))
#
# y = np.array([4, 6, 8, 10, 12])
# x = np.array([1, 2, 3, 4, 5])
# theta = np.array([1, 2])
# print(d_SSE_poly(y, x, theta))
#
# f = lambda x: SSE_poly(y, x, theta)
# print(gradientApprox(f, x, 0.00001))


#Batch gradient descent for problem 2.3
def batchGradDescent(y, x, m, step_size, num_iterations, theta = None, expander = expand):
    x = expander(x,m)
    if theta is None:
        theta = np.zeros([len(x[0])])
    errors = []
    for i in range(num_iterations):
        error = SSE_poly_expanded(y, x, theta)
        errors.append(error)

        d_sse_error = d_SSE_poly_expanded(y, x, theta)
        theta -= step_size * d_sse_error


    plt.plot(errors)
    plt.show("hold")
    return theta

#Batch gradient descent for problem 2.3
def stochGradDescent(y, x, m, t_o, k, num_iterations, theta = None, expander = expand):
    x = expander(x,m)
    if theta is None:
        theta = np.zeros(len(x[0]))
    errors = []
    for i in range(num_iterations):
        error = SSE_poly_expanded(y, x, theta)
        errors.append(error)
        step_size = (t_o+i)**(-k)

        for point in range(len(x)):
            theta -= step_size * d_SSE_poly_expanded(y[point], x[point], theta)

    plt.plot(errors)
    plt.show("hold")
    return theta

# m = 5

# theta = batchGradDescent(Y, X, m, .05, 10000)

# theta2 = stochGradDescent(Y, X, m, 100, 0.5, 1000)

# theta3 = batchGradDescent(Y, X, m, .001, 10000, expander=expand_theta)

# theta4 = stochGradDescent(Y, X, m, 20, 0.75, 10000, expander=expand_theta)

# y = np.array([3, 5, 7])
# x = np.array([1, 2, 3])
# theta = np.array([1, 2])
# print (SSE_poly(y, x, theta))
#
# y = np.array([4, 6, 8])
# x = np.array([1, 2, 3])
# theta = np.array([1, 2])
# print(d_SSE_poly(y, x, theta))
#
# f = lambda theta: SSE_poly(y, x, theta)
# print(gradientApprox(f, theta, 0.001))

def maxLikelihoodVector(x,y,m):
    x_panded = expand(x,m)
    x_panded = np.matrix(x_panded)
    y = np.matrix(y)
    w = np.linalg.inv(x_panded.T*x_panded)*x_panded.T*y.T
    return w

# maxVec = maxLikelihoodVector(X,Y,3)
# X_test = np.linspace(0,1,100)
# sol = np.matmul(expand_theta(X_test, m), theta3)
# plt.plot(X_test,sol)
#
# plt.plot(X,Y,'o')
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()